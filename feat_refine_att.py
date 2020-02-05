import torch 
import torch.nn as nn
import torch.nn.functional as F
import networks

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None):
        
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
        	attention = attention * scale
        
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention
    
class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.BatchNorm1d(model_dim)

    def forward(self, key, value, query):

        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm((residual + output).permute(0, 2, 1)).permute(0, 2, 1)

        return output, attention


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.BatchNorm1d(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm((x + output).permute(0, 2, 1)).permute(0, 2, 1)
        return output


class EncoderLayer(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs):

        # self attention
        context, attention = self.attention(inputs, inputs, inputs)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):

    def __init__(self,
               num_layers=6,
               model_dim=512,
               num_heads=8,
               ffn_dim=2048,
               dropout=0.0):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
           range(num_layers)])
        self.fc = nn.Linear(model_dim, model_dim)
        torch.nn.init.xavier_normal_(self.fc.weight, gain=1e-7)
        torch.nn.init.constant_(self.fc.bias, val=1e-7)

        
    def forward(self, x):
        
        for encoder in self.encoder_layers:
            x, _ = encoder(x)
        
        x = self.fc(x)
        return x
        
class ClassifierRefine(nn.Module):
    """
    nKall: number of categories in train-set
    nKnovel: number of categories in an episode
    nFeat: feature dimension at the input of classifier
    q_steps: number of iteration used in weights refinement
    """
    def __init__(self, nKnovel, nFeat):
        super(ClassifierRefine, self).__init__()

        self.nFeat = nFeat
        self.nKnovel = nKnovel
        # bias & scale of classifier
        self.bias = nn.Parameter(torch.FloatTensor(1).fill_(0), requires_grad=True)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(10), requires_grad=True)

        # init_net
        self.favgblock = networks.FeatExemplarAvgBlock(self.nFeat)
        

    def apply_classification_weights(self, features, cls_weights):
        '''
        (B x n x nFeat, B x nKnovel x nFeat) -> B x n x nKnovel
        '''
        features = F.normalize(features, p=2, dim=features.dim()-1, eps=1e-12)
        cls_weights = F.normalize(cls_weights, p=2, dim=cls_weights.dim()-1, eps=1e-12)

        cls_scores = self.scale_cls * torch.baddbmm(1.0, self.bias.view(1, 1, 1), 1.0,
                                                    features, cls_weights.transpose(1,2))
        return cls_scores

    
    def get_classification_weights(self, features_supp=None, labels_supp_1hot=None, features_query=None):
        '''
        features_supp, labels_supp --> self.init_theta
        features_query --> self.refine_theta
        '''
        assert(features_supp is not None and features_query is not None)

        # generate weights for novel categories
        features_supp = F.normalize(features_supp, p=2, dim=features_supp.dim()-1, eps=1e-12)

        cls_weights = self.favgblock(features_supp, labels_supp_1hot) # B x nKnovel x nFeat

        return cls_weights


    def forward(self, features_supp=None, labels_supp=None, features_query=None):
        '''
        features_supp: (B, nKnovel * nExamplar, nFeat)
        labels_supp: (B, nknovel * nExamplar) in [0, nKnovel - 1]
        features_query: (B, nKnovel * nTest, nFeat)
        '''
        labels_supp_1hot = networks.label_to_1hot(labels_supp, self.nKnovel)
        cls_weights = self.get_classification_weights(features_supp, labels_supp_1hot, features_query)
        cls_scores = self.apply_classification_weights(features_query, cls_weights)

        return cls_scores

                
if __name__ == '__main__' : 
    
    net = Encoder(num_layers=6,
                      model_dim=512,
                      num_heads=8,
                      ffn_dim=2048,
                      dropout=0.0)
    net = net.cuda()
    input = torch.randn(1, 10, 512).cuda()
    output = net(input)
    print (output.size())
