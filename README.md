# ClusterODSS：基于特征聚类的目标检测样本筛选方法
用于目标检测数据增强后的样本筛选
## 项目思路：

（1）将需要筛选的数据增强后的数据制作成一个分类数据集

（2）用分类数据集微调ImageNet上的分类模型：swin-Transformer、mobilenet、resnet等

（3）预测增强后的数据，提取最终的特征向量：1x1024

（4）PCA降维：1x1024→1x256

（5）K-Means聚类


## 训练步骤
1. datasets文件夹下存放的图片分为两部分，train里面是训练图片，test里面是测试图片。  
2. 在训练之前需要首先准备好数据集，在train或者test文件里里面创建不同的文件夹，每个文件夹的名称为对应的类别名称，文件夹下面的图片为这个类的图片。文件格式可参考如下：
```
|-datasets
    |-train
        |-cat
            |-123.jpg
            |-234.jpg
        |-dog
            |-345.jpg
            |-456.jpg
        |-...
    |-test
        |-cat
            |-567.jpg
            |-678.jpg
        |-dog
            |-789.jpg
            |-890.jpg
        |-...
```
3. 在准备好数据集后，需要在根目录运行txt_annotation.py生成训练所需的cls_train.txt，运行前需要修改其中的classes，将其修改成自己需要分的类。   
4. 之后修改model_data文件夹下的cls_classes.txt，使其也对应自己需要分的类。  
5. 在train.py里面调整自己要选择的网络和权重后，就可以开始训练了！ 


## 筛选步骤
1. 建立用于存放有效样本和无效样本的文件夹。
2. 修改getFeatures.py中的对应路径。
3. 运行getFeatures.py。

## 效果展示
|  增强方法   | 有效样本  | 无效样本  |
| :----: | :----: | :----: |
|  噪声  |![image](https://user-images.githubusercontent.com/44053847/209775199-09717c7d-2de6-4075-a44b-a1f5c4dcca13.png)|![image](https://user-images.githubusercontent.com/44053847/209775295-8a5e038d-18be-460f-ae1b-de5fff5c426e.png)|
|  CycleGAN  |![image](https://user-images.githubusercontent.com/44053847/209775318-70058a83-6ccb-436a-8031-0a3e3584245d.png)|![image](https://user-images.githubusercontent.com/44053847/209775335-006a7a22-5578-4da0-acc3-6e4c5276147e.png)|


## Reference
https://github.com/keras-team/keras-applications

https://github.com/bubbliiiing/classification-pytorch
