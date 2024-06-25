# 基于Bilinear-VGG的鸟类细粒度分类

(Fine-grained-image-classification-based-on-Bilinear-VGG-for-CUB-200-2011)

## 数据集介绍(Introduction to the dataset)

The Caltech-UCSD Birds-200-2011 Dataset（CUB-200-2011）

加州理工学院-加州大学洛杉矶分校Birds-200-2011（CUB-200-2011）数据集是最广泛使用的细粒度视觉分类任务数据集。该数据集包含属于鸟类的 200 个子类别的 11,788 张图像，其中 5,994 张用于训练，5,794 张用于测试。每张图像都有详细的注释： 1 个子类别标签、15 个部分位置、312 个二进制属性和 1 个边界框。文本信息来自 Reed 等人。他们通过收集精细的自然语言描述来扩展 CUB-200-2011 数据集。他们为每幅图像收集了 10 个单句描述。自然语言描述是通过 Amazon Mechanical Turk（AMT）平台收集的，要求至少 10 个单词，不包含任何子类别和操作信息。

![dataset](https://github.com/ych2862715221/Fine-grained-image-classification-based-on-Bilinear-VGG-for-CUB-200-2011/blob/main/README/dataset.jpg)

## 双线性卷积网络介绍( Introduction to Bilinear Convolutional Networks)

双线性卷积网络示意图如下：

![A Bilinear CNN model for image classification](https://github.com/ych2862715221/Fine-grained-image-classification-based-on-Bilinear-VGG-for-CUB-200-2011/blob/main/README/A%20Bilinear%20CNN%20model%20for%20image%20classification.jpg)

在本项目中我以VGG13作为基础卷积神经网络，用CUB-200-2011数据集就行训练， 其中，我将数据集按 9:1的比例划分为训练集(Train)和验证集(Val)，其中选取样本做验证集是采用均匀划分的方式进行的。在本次训练中只用到数据集的图像和类别标签。

这个数据集是一个细粒度数据集，类间距离较近，可判别区域(discriminative parts)只是图像中的很小一部分。

![细粒度](https://github.com/ych2862715221/Fine-grained-image-classification-based-on-Bilinear-VGG-for-CUB-200-2011/blob/main/README/%E7%BB%86%E7%B2%92%E5%BA%A6.jpg)

其中CNN A 和 CNN B的参数是完全一样的，也就是说两个卷积块输出的特征向量是一样的。在提取完特征后，两两进行一个外积得到一个双线性向量，接上全连接层进行分类。

## 模型训练(Training model)

在训练开始之前，对数据集进行预处理有随机裁剪(Random Resized Crop)，随机翻转(Random Horizontal Flip)，归一化(Normalize)等等。

此外优化器(optimizer)采用的是 SGD方法，其中，每隔100个epoch，learning-rate变为原来的10%，训练300轮。

## 实验结果(Results)

训练集准确率：

![train_acc](https://github.com/ych2862715221/Fine-grained-image-classification-based-on-Bilinear-VGG-for-CUB-200-2011/blob/main/README/train_acc.jpg)

验证集准确率：

![Val_acc](https://github.com/ych2862715221/Fine-grained-image-classification-based-on-Bilinear-VGG-for-CUB-200-2011/blob/main/README/Val_acc.jpg)
