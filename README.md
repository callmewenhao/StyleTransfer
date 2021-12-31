# 风格迁移

> 迁移学习实战——风格迁移👏

## 预备知识

### CNN

> CNN随着层数的加深，在空间尺度的像素信息被丢失，而语义信息在增加

- 每个卷积核提取不同的特征😊

- 1个卷积核对其输入进行卷积操作，得到1个 feature map ，其体现了卷积核从输入中提取的特征

- 不同卷积核提取不同特征，当卷积核提取到对应特征后，feature map 中对应的值特别大🎈
- 浅层卷积核提取︰边缘、颜色、斑块等底层像素特征
- 中层卷积核提取︰条纹、纹路、形状等中层纹理特征
- 高层卷积核提取︰眼睛、轮胎、文字等高层语义特征
- 最后的分类输出层输出最抽象的分类结果

### VGG16 & 19

> VGG16 & 19 的结构如下：

<img src="figures\VGG.png" style="zoom:100%;" />

## 算法基本流程

<img src="figures\ST1.png" style="zoom:50%;" />

## 损失函数

### 内容损失函数

> 比较内容图和生成图经过VGG19后的某个 feature map 的差异

$J_{content}$ 如下：
$$
J_{content} = \frac{1}{2}||a^{[l][c]} - \alpha^{[l][g]}||^{2}
$$
其中，$\alpha^{[l]}$ 是第 $l$ 个中间层的 feature map

### 风格损失函数💥

> Gram Matrix：Gram矩阵是两两向量的内积组成，所以Gram矩阵可以反映出该组向量中各个向量之间的某种关系。

输入图像的 feature map 为 [ c, h, w]。我们经过 flatten（即是将h\*w进行平铺成一维向量）和矩阵转置操作，可以变形为 [ch, h\*w] 和 [h*w, ch] 的矩阵。再对两个作内积得到 Gram Matrices。

<img src="figures\Gram.png" style="zoom:80%;" />

> 比较内容图和生成图经过VGG19后的某个 feature map 的 Gram Matrix 他们越接近越好

第 $l$ 层的风格损失函数：
$$
E_{l} = \frac{1}{4N^{2}_{l}M^{2}_{l}}\sum_{i, j}(G_{ij}^{l}-A_{ij}^{l})^{2}
$$
其中，$N$ 和 $M$ 分别是：C、H*W

总风格损失函数：
$$
L_{style}(\vec{a}, \vec{x}) = \sum_{l=0}^{L}w_{l}E_{l}
$$
其中，$w$ 是每层 loss func 的权重

### 融合损失函数

总的损失函数：
$$
L_{total}(\vec{p}, \vec{a},\vec{x})=\alpha L_{content}(\vec{p}, \vec{x})+\beta L_{style}(\vec{a},\vec{x})
$$
其中，$\vec{p}$ 是内容图， $\vec{a}$ 是风格图，$\vec{x}$ 是生成图

## keras实战

https://keras.io/examples/generative/neural_style_transfer/

## fast-style-transfer

> paper: [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155v1.pdf) 

