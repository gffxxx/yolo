# Average Precision (AP) 

**平均精度：物体检测模型性能的常用指标**

**所有类别**的平均精度，提供一个单一的数值来比较不同的模型



基于precision-recall metrics  

单独计算每个类别的平均精度（AP），然后取所有类别中这些AP的平均值

## **IoU(Intersection over Union)**：实况和预测边界盒之间的重叠程度

![img](https://i-blog.csdnimg.cn/blog_migrate/0e833fe1f3bf246b8c44e76e8305740f.png)

![img](https://i-blog.csdnimg.cn/blog_migrate/785bd39d41d0a9b7411b65ce93eb5f52.png)

![img](https://i-blog.csdnimg.cn/blog_migrate/4daeeee05017e51cc4d63a4a7e377e11.png)

不会反应两个目标之间的距离

无法精确反应两者重合度大小

## GIOU（Generalized IoU）

<img src="https://i-blog.csdnimg.cn/blog_migrate/03edcfdd474e390c1b27388182c4ab16.png" alt="img"  />

C：两个框的最小外接矩形的面积

 当IOU为0时，意味着A与B没有交集，这个时候两个框离得越远，GIOU越接近-1；两框重合，GIOU=1，所以GIOU的取值为(-1, 1]

1. 当两个框属于包含关系时，GIoU会退化成IoU，无法区分其相对位置关系

   <img src="https://i-blog.csdnimg.cn/blog_migrate/4616b234479e091e99fbb8751bd6f67a.png" alt="4af195ddac92494ab64f3b1d4231870d.png" style="zoom:67%;" />

2. 在两个垂直方向，误差很大

![c7998ce5d28f412487ebddb8f18c312d.png](https://i-blog.csdnimg.cn/blog_migrate/8e5fe0a38db974c1993cf1ce91da6b26.png)



## DIoU

最小化两个BBox中心点的标准化距离<img src="https://i-blog.csdnimg.cn/blog_migrate/ba691dadcc1e1d847b015aee2bb8177b.png" alt="img" style="zoom:67%;" />

![ee2a4bd2bed34d08ad3ab5014fe4048b.png](https://i-blog.csdnimg.cn/blog_migrate/5811a55a383a4bde3b737f6bbd392b81.png)

**纵横比**暂未考虑

## CIoU

![img](https://i-blog.csdnimg.cn/blog_migrate/9b735e6924fcdbf349b724f5b3f8a6a1.png)

α是权重函数，v用来度量宽高比的一致性

![img](https://i-blog.csdnimg.cn/blog_migrate/53313dbff5dd6b1f1ce4c5bbb9b59c6f.png)

![img](https://i-blog.csdnimg.cn/blog_migrate/3d5b4580cb1d07f35799a40a2cd840f8.png)

![image-20240805133848606](C:\Users\51139\AppData\Roaming\Typora\typora-user-images\image-20240805133848606.png)



**VOC数据集**：（20个类别）

1.对于每个类别：改变模型预测的置信度阈值，计算出精确-召回曲线

2.计算每个类别的AP：使用精度-召回曲线的**内插11点抽样**

3.求20个类别AP的平均值



**COCO数据集**：Microsoft COCO (Common Objects in Context)  

1.对于每个类别：改变模型预测的置信度阈值，计算出精确-召回曲线

2.计算每个类别的AP：使用精度-召回曲线的**内插101点抽样**

3.在**不同的交叉联合（IoU）阈值**下计算AP，通常从0.5到0.95，步长为0.05。更高的IoU阈值需要更准确的预测才能被认为是真阳性

4.对于每个IoU阈值，取所有80个类别的AP的平均值

5.最后，通过平均每个IoU阈值计算的AP值来计算总体AP

## NMS（Non-Maximum Suppression）非极大值抑制

物体检测算法中使用的一种后处理技术

作用：减少重叠边界盒的数量，提高整体检测质量

物体检测算法通常会在同一物体周围产生多个具有不同置信度分数的边界框。NMS**过滤掉多余的和不相关的边界盒**，只保留最准确的边界盒。

<img src="https://i-blog.csdnimg.cn/blog_migrate/4d85eb2cf377d6ab1c623cafe9e633e1.png" alt="img" style="zoom:150%;" />

![img](https://i-blog.csdnimg.cn/blog_migrate/1d4348410f4509405a51e5442ff2eee7.png)





# YOLOv1：**You Only Look Once**

Joseph Redmon、Santosh Divvala、Ross Girshick

R-CNN系列：two-stage



one-stage  实时的**端到端**物体检测方法

YOLO 的核心思想就是把目标检测转变成一个**回归问题**，利用整张图作为网络的输入，仅仅经过一个神经网络，得到bounding box（边界框） 的位置及其所属的类别。

将输入图像划分为 S × S 网格，并预测同一类别的 B 个边界框，以及每个网格元素对 C 个不同类别的置信度。

每个边界框的预测由五个值组成：*Pc、bx、by、bh、bw* 

Pc是bounding box的置信度分数，反映了模型对bbox包含物体的置信度以及bbox的精确程度。*bx*和*by*坐标是方框相对于网格单元的中心，*bh*和*bw*是方框相对于整个图像的高度和宽度。

1. 将一幅图像分成 S×S个网格（grid cell），如果某个 object 的中心落在这个网格中，则这个网格就负责预测这个object。

2. 每个网格要预测 B 个bounding box，每个 bounding box 要预测 (x, y, w, h) 和 confidence 共5个值。
3. 每个网格还要预测一个类别信息，记为 C 个类。
4. 总的来说，S×S 个网格，每个网格要预测 B个bounding box ，还要预测 C 个类。网络输出就是一个 S × S × (5×B+C) 的张量。

*S*×*S*×(*B*×5+*C)*

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1dc079481165333e39c459c1ad418553.png)

## 网络架构

24个卷积层+2个全连接层（预测bbox坐标和概率）

除了最后一个层使用线性激活函数外，所有层都使用了leaky rectified linear unit  激活

<img src="C:\Users\51139\AppData\Roaming\Typora\typora-user-images\image-20240801170542225.png" alt="image-20240801170542225" style="zoom:25%;" />

<img src="https://ask.qcloudimg.com/http-save/6834658/1ye6lfzda1.png" alt="img" style="zoom: 33%;" />



增强：

1. 最多为输入图像大小 20% 的随机缩放和平移
2. HSV 色彩空间中上限系数为 1.5 的随机曝光和饱和度

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f6a5794c96a0755366c7b8badbf766a2.png)

![image-20240801164104141](C:\Users\51139\AppData\Roaming\Typora\typora-user-images\image-20240801164104141.png)

使用1×1卷积层来减少特征图的数量并保持相对较低的参数数量



![1](https://i-blog.csdnimg.cn/blog_migrate/1ee7151433d4f8f0be2afa0548db11fe.png)





## 损失函数

 坐标预测损失、置信度预测损失、类别预测损失

λcoord = 5：定位误差比分类误差更大，所以增加对定位误差的惩罚

λnoobj = 0.5：减少了不包含目标的框的置信度预测的损失（许多网格单元不包含任何目标。训练时就会把这些网格里的框的“置信度”分数推到零，这往往超过了包含目标的框的梯度。）



<img src="https://i-blog.csdnimg.cn/blog_migrate/a8f1588c2e1202d6abcf23c3fcfdd3cd.png" alt="img" style="zoom:150%;" />

## **缺陷**

1. 网格单元中最多只能检测到两个相同类别的物体，限制了它预测附近物体的能力（对相互靠近的物体，以及很小的群体检测效果不好）
2. 定位误差较大
3. 对不常见的角度的目标泛化性能偏弱。

# YOLOv2

Joseph Redmon and Ali Farhadi  

适应多种尺寸的图片输入



改进：

1. 对所有卷积层进行批量归一化（BN）可提高收敛性，并作为正则化器减少过拟合

   ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/55501802d1750d41d2b8d7e8624b7da0.png)

   γ 参数实现对标准化后的x再次进行**缩放**，β 参数实现对标准化的x 进行**平移**。不同的是，γ 、β 参数均由反向传播算法自动优化，实现网络层“按需”缩放平移数据的分布的目的。
   
2. **使用Anchor来预测边界盒**

   YOLOv1:全连接层直接预测Bounding Box的坐标

   ​		YOLOv2:引入Anchor机制(K-means**聚类**在训练集中聚类计算出Anchor Box 大小)前筛选得到的具有代表性**先		验框Anchors**，训练时更快收敛

   Anchor：预定义形状

   ![image-20240802090527695](C:\Users\51139\AppData\Roaming\Typora\typora-user-images\image-20240802090527695.png)

   k-means聚类：

   距离度量：<img src="https://i-blog.csdnimg.cn/blog_migrate/cd2741dffc218f48b7438ff49938965b.jpeg" alt="在这里插入图片描述" style="zoom: 50%;" />

   ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d0cc120e26dc1fb575435561007de6a7.png)

   YOLOv2采用5种 Anchor

3. 全卷积，去掉了FC

   Darknet-19  backbone（主干：layer1-23）+detection head（4个卷积层+passthrough layer）

   object classification head：conv（1000filters）+Global Avgpool+softmax

   ![image-20240802085418475](C:\Users\51139\AppData\Roaming\Typora\typora-user-images\image-20240802085418475.png)

   最后一个卷积层输出 13×13 的 feature map

2. 预测 bounding box 与 ground truth 的位置偏移值 t x , t y（相对于该网格左上角坐标的相对偏移量）

​		

<img src="C:\Users\51139\AppData\Roaming\Typora\typora-user-images\image-20240802090850660.png" alt="image-20240802090850660" style="zoom:80%;" />

每个bounding box有五个值*tx* *, ty* *,* *tw* *, th* *, to*






![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/2ecf413382a80e788a69a5bcf824a740.jpeg)



5.**Fine-Grained Features**：细粒度特征

不同层之间的特征融合

Passthrough Layer

先获取前层的26×26的特征图，将其同最后输出的13×13的特征图进行连接

# YOLOv3

darknet-19 →darknet-53

<img src="C:\Users\51139\AppData\Roaming\Typora\typora-user-images\image-20240805092420493.png" alt="image-20240805092420493" style="zoom:150%;" />

<img src="https://i-blog.csdnimg.cn/blog_migrate/afa91428ed91c0554951d5be50d667c6.png" alt="在这里插入图片描述" style="zoom: 200%;" />

**DBL：** 一个卷积层、一个批量归一化层和一个Leaky ReLU

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1ef8f71e5eeab9ff173665f48a4fd6a4.png)



1. **多标签分类**

   softmax层修改为逻辑分类器（主要用到了sigmoid函数）

2. **多尺度预测**

   选择了三种不同shape的Anchors，同时每种Anchors具有三种不同的尺度，一共9中不同大小的Anchors

   ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/be5c8e36ddb29878def3a25639b9bb6f.png)

   ![image-20240805134818222](C:\Users\51139\AppData\Roaming\Typora\typora-user-images\image-20240805134818222.png)

3. **特征金字塔网络（feature parymid network,FPN)**

   设计了3种不同尺度的网络输出Y1、Y2、Y3

   ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e35289f677caab4474bf324a0a25a612.png)

损失函数

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/dbc97a5dda0daef0a2ee9dac5db2b32d.png)

**置信度损失和类别预测均由原来的sum-square error改为了交叉熵的损失计算方法**



# YOLOv4

**YOLOv4 = CSPDarknet53（主干） + SPP附加模块（颈） + PANet路径聚合（颈） + YOLOv3（头部）**

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e094082d336f85c0eadc7d9b25880b8d.png)

CSPNet全称是Cross Stage Partial Network

darknet-53 →CSPDarknet53

![image-20240805135756878](C:\Users\51139\AppData\Roaming\Typora\typora-user-images\image-20240805135756878.png)

**MIsh激活函数代替了原来的Leaky ReLU**



###  SPP（Spatial Pyramid Pooling）

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/febe9a727bd69b7b6ab1a1d6a943d9eb.png)

最大池化的池化核大小分别为13x13、9x9、5x5、1x1

### PANet（Path Aggregation Network)

![image-20240805143236830](C:\Users\51139\AppData\Roaming\Typora\typora-user-images\image-20240805143236830.png)

- **Bottom-up Path Augmentation**主要是考虑**网络浅层特征信息**对于实例分割非常重要，因为浅层特征一般是边缘形状等特征。



**Mosaic 数据增强**：利用四张图片进行拼接



# YOLOv5

输入图像：进行Mosaic数据增强



detection:YOLOv5利用**GIOU_Loss**来代替Smooth L1 Loss函数





# YOLOv8

 CSPDarknet（主干） + PAN-FPN（颈） + Decoupled-Head（输出头部）

**anchor-free**

**解耦头（Decoupled Head）**

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ae93be487fd370c0dacb4e700cfc6f25.png)





<img src="https://i-blog.csdnimg.cn/blog_migrate/716093d7317f9bc62e76867574600c51.jpeg" alt="在这里插入图片描述" style="zoom:150%;" />

