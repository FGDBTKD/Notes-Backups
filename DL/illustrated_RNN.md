# Illustrated RNN, RNN variant, Seq2Seq, Attention mechanism

----------
> [参考链接](https://zhuanlan.zhihu.com/p/28054589)

----------

> 本文主要是利用图片的形式，详细地介绍了经典的**RNN**、**RNN几个重要变体**，以及**Seq2Seq模型**、**Attention机制**。

----------

# 从单层网络谈起

首先来了解一下最基本的单层网络，它的结构如图：

![Figure 1](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/17.jpg)

输入是$x$，经过变换$Wx+b$和激活函数$f$得到输出$y$。

----------

# 经典的RNN结构（N ---> N）

在实际应用中，我们还会遇到很多序列形的数据：

![Figure 2](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/1.jpg)

如：

 - 自然语言处理问题。x1可以看做是第一个单词，x2可以看做是第二个单词，依次类推。
 - 语音处理。此时，x1、x2、x3……是每帧的声音信号。
 - 时间序列问题。例如每天的股票价格等等。

> 序列形的数据就不太好用原始的神经网络处理了。
为了建模序列问题，**RNN**引入了隐状态**h（hidden state）**的概念，**h**可以对序列形的数据提取特征，接着再转换为输出。
先从**h1**的计算开始看：

![Figure 3](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/14.jpg)

> 图示中记号的含义是：
圆圈或方块表示的是向量。
一个箭头就表示对该向量做一次变换。如上图中$h_0$和$x_1$分别有一个箭头连接，就表示对$h_0$和$x_1$各做了一次变换。

$h_2$的计算和$h_1$类似。
要注意的是，在计算时，每一步使用的参数$U$、$W$、$b$都是一样的，也就是说每个步骤的参数都是共享的。

![Figure 4](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/8.jpg)

依次计算剩下来的（使用相同的参数$U$、$W$、$b$）：

![Figure 5](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/15.jpg)

这里为了方便起见，只画出序列长度为$4$的情况，实际上，这个计算过程可以无限地持续下去。

目前的**RNN**还没有输出，得到输出值的方法就是直接通过$h$进行计算：

![Figure 6](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/6.jpg)

正如之前所说，一个箭头就表示对对应的向量做一次类似于$f(Wx+b)$的变换，这里的这个箭头就表示对$h_1$进行一次变换，得到输出$y_1$。

剩下的输出类似进行（使用和$y_1$同样的参数$V$和$c$）：

![Figure 7](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/11.jpg)

这就是最经典的**RNN**结构，
它的输入是$x_1, x_2, ...x_n$，输出为$y_1, y_2, ...y_n$，
也就是说，输入和输出序列必须要是等长的。

由于这个限制的存在，经典**RNN**的适用范围比较小，
但也有一些问题适合用经典的**RNN**结构建模，如：

 - 计算视频中每一帧的分类标签。因为要对每一帧进行计算，因此输入和输出序列等长。
 - 输入为字符，输出为下一个字符的概率。这就是著名的**Char RNN**。

----------

# RNN结构（N ---> 1）

> 有时要处理的问题输入是一个序列，输出是一个单独的值而不是序列，应该怎样建模呢？实际上，我们只在最后一个$h$上进行输出变换就可以了：

![Figure 8](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/3.jpg)

> 这种结构通常用来处理序列分类问题。
如输入一段文字判别它所属的类别，
输入一个句子判断其情感倾向，
输入一段视频并判断它的类别等等。

----------

# RNN结构（1 ---> N）

> 输入不是序列而输出为序列的情况怎么处理？我们可以只在序列开始进行输入计算：

![Figure 9](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/10.jpg)

还有一种结构是把输入信息$X$作为每个阶段的输入：

![Figure 10](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/19.jpg)

下图省略了一些$X$的圆圈，是一个等价表示：
 
![Figure 11](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/7.jpg)
 
这种**1 ---> N**的结构可以处理的问题有：

 - 从图像生成文字（image caption），此时输入的$X$就是图像的特征，而输出的$y$序列就是一段句子。
 - 从类别生成语音或音乐等。

----------

# RNN结构（N ---> M）
 
> 下面来介绍**RNN**最重要的一个变种：**N ---> M**。
这种结构又叫**Encoder-Decoder**模型，也可以称之为**Seq2Seq**模型。
原始的**N ---> N RNN**要求序列等长，然而我们遇到的大部分问题序列都是不等长的，如机器翻译中，源语言和目标语言的句子往往并没有相同的长度。
为此，**Encoder-Decoder**结构先将输入数据编码成一个上下文向量$c$：

![Figure 12](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/2.jpg)

得到$c$有多种方式，最简单的方法就是把**Encoder**的最后一个隐状态赋值给$c$，还可以对最后的隐状态做一个变换得到$c$，也可以对所有的隐状态做变换。

拿到$c$之后，就用另一个**RNN**网络对其进行解码，这部分**RNN**网络被称为**Decoder**。
具体做法就是将$c$当做之前的初始状态$h_0$输入到**Decoder**中：

![Figure 13](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/9.jpg)

还有一种做法是将$c$当做每一步的输入：

![Figure 14](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/18.jpg)

由于这种**Encoder-Decoder**结构不限制输入和输出的序列长度，因此应用的范围非常广泛，比如：

 - 机器翻译。**Encoder-Decoder**的最经典应用，事实上这一结构就是在机器翻译领域最先提出的。
 - 文本摘要。输入是一段文本序列，输出是这段文本序列的摘要序列。
 - 阅读理解。将输入的文章和问题分别编码，再对其进行解码得到问题的答案。
 - 语音识别。输入是语音信号序列，输出是文字序列。
 - …………

----------

# Attention机制

> 在**Encoder-Decoder**结构中，**Encoder**把所有的输入序列都编码成一个统一的语义特征$c$再解码，因此， $c$中必须包含原始序列中的所有信息，它的长度就成了限制模型性能的瓶颈。
如机器翻译问题，当要翻译的句子较长时，一个$c$可能存不下那么多信息，就会造成翻译精度的下降。

**Attention机制**通过在每个时间输入不同的$c$来解决这个问题，下图是带有**Attention机制**的**Decoder**：

![Figure 15](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/5.jpg)

每一个$c$会自动去选取与当前所要输出的$y$最合适的上下文信息。
具体来说，我们用 $a_{ij}$ 衡量**Encoder**中第$j$阶段的$h_j$和解码时第$i$阶段的相关性，最终**Decoder**中第$i$阶段的输入的上下文信息 $c_i$ 就来自于所有 $h_j$ 对 $a_{ij}$ 的加权和。

以机器翻译为例（将中文翻译成英文）：

![Figure 16](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/16.jpg)

输入的序列是“我爱中国”，因此，**Encoder**中的$h_1、h_2、h_3、h_4$就可以分别看做是“我”、“爱”、“中”、“国”所代表的信息。
在翻译成英语时，第一个上下文$c_1$应该和“我”这个字最相关，因此对应的 $a_{11}$ 就比较大，而相应的 $a_{12} 、 a_{13} 、 a_{14}$ 就比较小。
$c_2$应该和“爱”最相关，因此对应的 $a_{22}$ 就比较大。
最后的$c_3$和$h_3、h_4$最相关，因此 $a_{33} 、 a_{34}$ 的值就比较大。

至此，关于**Attention模型**，我们就只剩最后一个问题了，那就是：这些权重 $a_{ij}$ 是怎么来的？

事实上， $a_{ij}$ 同样是从模型中学出的，它实际和**Decoder**的第$i-1$阶段的隐状态、**Encoder**第$j$个阶段的隐状态有关。

同样还是拿上面的机器翻译举例， $a_{1j}$ 的计算（此时箭头就表示对$h'$和 $h_j$ 同时做变换）：
 
![Figure 17](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/12.jpg)

$a_{2j}$ 的计算：

![Figure 18](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/13.jpg)

$a_{3j}$ 的计算：

![Figure 19](https://github.com/Text-sentiment-analysis-bjfu/work_log/raw/master/7-19/images/4.jpg)

以上就是**带有Attention的Encoder-Decoder模型**计算的全过程。

----------

# LSTM

> LSTM从外部看和RNN完全一样，因此上面的所有结构对LSTM都是通用的，具体可参考[LSTM内部结构](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)。

