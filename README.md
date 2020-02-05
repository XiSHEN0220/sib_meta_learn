# Beyond SIB

## TODO

* Instead of just giving support feature and class-averaged distance to the module, we can give all **pair-wise distances** and **refine features** for labeled data as well as for unlabeled data

* Try **weighted sum** instead of a simple mean

* Apply to **different tasks**!


## Discussion
In short, we need a module to change the feature (or the classifier weight) according to some input. 
As this input should represent a graph representing the pair-wise relationship, 
we can compute some feature distances (cos sim) for each pair of sample, 
or the distance between a class-averaged node and all the samples.

Instead of using a transformer to learn to encode this graph, we should directly apply this computation to the features, which could include labeled or unlabeled data.

Then, since we have a module aware of this pair-wise feature distance, it could output a feature gradient for each sample to make them more semanticly separatable in this feature space.

At last, the classifier weight can be set as the mean value or weighted sum of the labeled sample in each class.


## Dependencies
The code is tested under **Pytorch > 1.0 + Python 3.6** environment. 


## How to use the code
### **Step 0**: Download Mini-ImageNet dataset

``` Bash
cd data
bash download_miniimagenet.sh 
cd ..
```

### **Step 1** (optional): train a WRN-28-10 feature network (aka backbone)
The weights of the feature network is downloaded in step 0, but you may also train from scracth by running

``` Bash
python main_feat.py --outDir miniImageNet_WRN_60Epoch --cuda --dataset miniImageNet --nbEpoch 60
```

### **Step 2**: Few-shot classification on Mini-ImageNet, e.g., 5-way-1-shot:

``` Bash
python main.py --config config/miniImageNet_1shot.yaml --seed 100 --gpu 0
```

## Mini-ImageNet Results

| Setup         | 5-way-1-shot  | 5-way-5-shot |
| ------------- | -------------:| ------------:|
| SIB (K=3)     | 70.700% ± 0.585% | 80.045% ± 0.363%|
| SIB (K=5)     | 70.494 ± 0.619% | 80.192% ± 0.372%|
