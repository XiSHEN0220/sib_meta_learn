import os
import itertools
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils.outils import progress_bar, AverageMeter, accuracy, getCi
from utils.utils import to_device

class Algorithm:
    def __init__(self, args, logger, netFeat, netRefine, netClassifier, optimizer, criterion):
        self.netFeat = netFeat
        self.netRefine = netRefine
        self.netClassifier = netClassifier
        
        self.optimizer = optimizer
        self.criterion = criterion

        self.nbIter = args.nbIter
        self.outDir = args.outDir
        self.nFeat = args.nFeat
        self.batchSize = args.batchSize
        self.inputH = args.inputH
        self.inputW = args.inputW
        self.nEpisode = args.nEpisode
        self.logger = logger
        self.device = torch.device('cuda' if args.cuda else 'cpu')

        # Load pretrained model
        if args.resumeFeatPth :
            if args.cuda:
                param = torch.load(args.resumeFeatPth)
            else:
                param = torch.load(args.resumeFeatPth, map_location='cpu')
            self.netFeat.load_state_dict(param)
            msg = '\nLoading netFeat from {}'.format(args.resumeFeatPth)
            logger.info(msg)

        if args.ckptPth:
            param = torch.load(args.ckptPth)
            self.netFeat.load_state_dict(param['netFeat'])
            self.netRefine.load_state_dict(param['netRefine'])
            self.netClassifier.load_state_dict(param['netClassifier'])
            
            msg = '\nLoading networks from {}'.format(args.ckptPth)
            logger.info(msg)

            self.optimizer = torch.optim.SGD(itertools.chain(*[self.netRefine.parameters(), self.netClassifier.parameters()]),
                                             lr,
                                             momentum=args.momentum,
                                             weight_decay=args.weightDecay,
                                             nesterov=True)


    def validate(self, valLoader, mode='val'):
        if mode == 'test':
            nEpisode = self.nEpisode
            self.logger.info('\n\nTest mode: randomly sample {:d} episodes...'.format(nEpisode))
        elif mode == 'val':
            nEpisode = len(valLoader)
            self.logger.info('\n\nValidation mode: pre-defined {:d} episodes...'.format(nEpisode))
            valLoader = iter(valLoader)
        else:
            raise ValueError('mode is wrong!')

        episodeAccLog = []
        top1 = AverageMeter()

        self.netFeat.eval()
        self.netRefine.eval()
        self.netClassifier.eval()
        
        #for batchIdx, data in enumerate(valLoader):
        for batchIdx in range(nEpisode):
            data = valLoader.getEpisode() if mode == 'test' else next(valLoader)
            data = to_device(data, self.device)

            SupportTensor, SupportLabel, QueryTensor, QueryLabel = \
                    data['SupportTensor'].squeeze(0), data['SupportLabel'].squeeze(0), \
                    data['QueryTensor'].squeeze(0), data['QueryLabel'].squeeze(0)

            with torch.no_grad():
                SupportFeat, QueryFeat = self.netFeat(SupportTensor), self.netFeat(QueryTensor)
                SupportFeat, QueryFeat, SupportLabel = \
                        SupportFeat.unsqueeze(0), QueryFeat.unsqueeze(0), SupportLabel.unsqueeze(0)
                nbSupport, nbQuery = SupportFeat.size()[1], QueryFeat.size()[1]
                
                feat = torch.cat((SupportFeat, QueryFeat), dim=1)
                refine_feat = self.netRefine(feat)
                refine_feat = feat + refine_feat
                refine_support, refine_query = refine_feat.narrow(1, 0, nbSupport), refine_feat.narrow(1, nbSupport, nbQuery)
                clsScore = self.netClassifier(refine_support, SupportLabel, refine_query)
                clsScore = clsScore.squeeze(0)
            QueryLabel = QueryLabel.view(-1)
            acc1 = accuracy(clsScore, QueryLabel, topk=(1,))
            top1.update(acc1[0].item(), clsScore.size()[0])

            msg = 'Top1: {:.3f}%'.format(top1.avg)
            progress_bar(batchIdx, nEpisode, msg)
            episodeAccLog.append(acc1[0].item())
                

        mean, ci95 = getCi(episodeAccLog)
        self.logger.info('Final Perf with 95% confidence intervals: {:.3f}%, {:.3f}%'.format(mean, ci95))
        
        self.netRefine.train()
        self.netClassifier.train()
        return mean, ci95


    def train(self, trainLoader, valLoader, lr=None, coeffGrad=0.0) :
        bestAcc, ci = self.validate(valLoader)
        self.logger.info('Acc improved over validation set from 0% ---> {:.3f} +- {:.3f}%'.format(bestAcc,ci))

        self.netRefine.train()
        self.netFeat.eval()

        losses = AverageMeter()
        top1 = AverageMeter()
        history = {'trainLoss' : [], 'trainAcc' : [], 'valAcc' : []}

        for episode in range(self.nbIter):
            data = trainLoader.getBatch()
            data = to_device(data, self.device)

            with torch.no_grad() :
                SupportTensor, SupportLabel, QueryTensor, QueryLabel = \
                        data['SupportTensor'], data['SupportLabel'], data['QueryTensor'], data['QueryLabel']

                SupportFeat = self.netFeat(SupportTensor.contiguous().view(-1, 3, self.inputW, self.inputH))
                QueryFeat = self.netFeat(QueryTensor.contiguous().view(-1, 3, self.inputW, self.inputH))

                SupportFeat, QueryFeat = SupportFeat.contiguous().view(self.batchSize, -1, self.nFeat), \
                        QueryFeat.contiguous().view(self.batchSize, -1, self.nFeat)

            
            self.optimizer.zero_grad()
            
            nbSupport, nbQuery = SupportFeat.size()[1], QueryFeat.size()[1]
            feat = torch.cat((SupportFeat, QueryFeat), dim=1)
            refine_feat = self.netRefine(feat)
            refine_feat = feat + refine_feat
            refine_support, refine_query = refine_feat.narrow(1, 0, nbSupport), refine_feat.narrow(1, nbSupport, nbQuery)
            clsScore = self.netClassifier(refine_support, SupportLabel, refine_query)
            
            clsScore = clsScore.view(refine_query.size()[0] * refine_query.size()[1], -1)
            QueryLabel = QueryLabel.view(-1)

            loss = self.criterion(clsScore, QueryLabel)

            loss.backward()
            self.optimizer.step()

            acc1 = accuracy(clsScore, QueryLabel, topk=(1, ))
            top1.update(acc1[0].item(), clsScore.size()[0])
            losses.update(loss.item(), QueryFeat.size()[1])
            msg = 'Loss: {:.3f} | Top1: {:.3f}% '.format(losses.avg, top1.avg)
            progress_bar(episode, self.nbIter, msg)

            if episode % 1000 == 999 :
                acc, _ = self.validate(valLoader)

                if acc > bestAcc :
                    msg = 'Acc improved over validation set from {:.3f}% ---> {:.3f}%'.format(bestAcc , acc)
                    self.logger.info(msg)

                    bestAcc = acc
                    self.logger.info('Saving Best')
                    torch.save({
                                'lr': lr,
                                'netFeat': self.netFeat.state_dict(),
                                'netRefine': self.netRefine.state_dict(),
                                'netClassifier': self.netClassifier.state_dict(),
                                
                                }, os.path.join(self.outDir, 'netBest.pth'))

                self.logger.info('Saving Last')
                torch.save({
                            'lr': lr,
                            'netFeat': self.netFeat.state_dict(),
                            'netRefine': self.netRefine.state_dict(),
                            'netClassifier': self.netClassifier.state_dict(),
                            
                            }, os.path.join(self.outDir, 'netLast.pth'))

                msg = 'Iter {:d}, Train Loss {:.3f}, Train Acc {:.3f}%, Val Acc {:.3f}%'.format(
                        episode, losses.avg, top1.avg, acc)
                self.logger.info(msg)
                history['trainLoss'].append(losses.avg)
                history['trainAcc'].append(top1.avg)
                history['valAcc'].append(acc)

                losses = AverageMeter()
                top1 = AverageMeter()

        return bestAcc, acc, history
