# Learning Rate

> Learning Rate（学习率）决定了权值更新的速度，设置得太大会使结果超过最优值，太小会使下降速度过慢。仅靠人为干预调整参数需要不断修改学习率，因此后面$3$种参数

 - *Weight Decay 权值衰减*
 - *Momentum 动量*
 - *Learning Rate Decay 学习率衰减*

都是基于自适应的思路提出的解决方案。

----------
# Weight decay

> weight decay（权值衰减）的使用既不是为了提高收敛精确度也不是为了提高收敛速度，其最终目的是防止过拟合。在损失函数中，weight decay是放在正则项（regularization）前面的一个系数，正则项一般指示模型的复杂度，所以weight decay的作用是调节模型复杂度对损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大。

在实际应用中，为了避免网络的过拟合，必须对损失函数加入一些正则项，在SGD（Stochastic Gradient Descent）中加入$\eta \lambda \omega _{i} $这一正则项对这个loss function进行规范化：

$$\omega_{i}\leftarrow  \omega_{i} - \eta \frac{\partial E}{\partial \omega_{i}} - \eta \lambda \omega _{i} $$

上面这个公式基本思想就是减小不重要的参数对最后结果的影响，网络中有用的权重则不会受到Weight decay影响。

> 在机器学习或者模式识别中，会出现overfitting，而当网络逐渐overfitting时网络权值逐渐变大，因此，为了避免出现overfitting,会给误差函数添加一个惩罚项，常用的惩罚项是所有权重的平方乘以一个衰减常量之和。其用来惩罚大的权值。

----------
# Momentum 

momentum是梯度下降法中一种常用的加速技术。对于一般的SGD，其表达式为$x \leftarrow  x-\alpha \ast dx$,$x$沿负梯度方向下降。而带momentum项的SGD则生成如下形式：


$$v=\beta \ast v -a\ast dx\\$$

$x \leftarrow  x+v$

其中$\beta$ 即momentum系数，通俗的理解上面式子就是，如果上一次的momentum（即v）与这一次的负梯度方向是相同的，那这次下降的幅度就会加大，所以这样做能够达到加速收敛的过程。动量来源于牛顿定律，基本思想是为了找到最优加入“惯性”的影响，当误差曲面中存在平坦区域，SGD就可以更快的学习。

$$\omega_{i}\leftarrow  m\cdot \omega_{i} - \eta \frac{\partial E}{\partial \omega_{i}} $$

----------
# Learning Rate Decay 

该方法是为了提高SGD寻优能力，具体就是每次迭代的时候减少学习率的大小。

$$\eta \left( s \right) =\frac{\eta _{0} }{1+s\cdot \eta _{n}} $$

----------
# normalization（batch normalization）

batch normalization是指在神经网络中激活函数的前面，将$wx+b$按照特征进行normalization，
这样做的好处有三点：

 1. 提高梯度在网络中的流动。Normalization能够使特征全部缩放到[0,1]，这样在反向传播时候的梯度都是在1左右 ，避免了梯度消失现象。
 2. 提升学习速率。归一化后的数据能够快速的达到收敛。
 3. 减少模型训练对初始化的依赖。

----------


 
