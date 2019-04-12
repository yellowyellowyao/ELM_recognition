# ELM_recognition
## 背景说明：
ELM（边缘局域模，Edge local mode）是等离子体运行中的一种模式，常伴随H-模（高约束模）一起出现。
ELM出现时，Hα信号幅值整体下降，能量约束水平上升，伴随着大量的准周期性的、快速的尖峰状扰动，即 ELM，如下图所示：
<div style="align: center">
<img src="https://github.com/yellowyellowyao/ELM_recognition/blob/master/picture/ELM%E7%A4%BA%E6%84%8F%E5%9B%BE.png" width = "600" height = "400" />
 </div >

## 训练数据
24.19万数据切片（由5200炮数据切片获得）

## 数据处理流程
本项目对数据处理流程如下图所示:
首先对完整的一个通道数据进行切片处理，以获得大量数据及保证识别精度，之后神经网络会识别一个切片中是否含有ELM信号。识别完成后再平移滑动窗口识别下一个数据切片。这样最终会得到一个只含0、1的一维数据。考虑到L-H模的转换是连续的过程，不会出现单独几个切片长度的ELM，因而进行平滑处理得到ELM开始与结束时刻。
<img src=https://github.com/yellowyellowyao/ELM_recognition/blob/master/picture/ELM%E8%AF%86%E5%88%AB%E7%AE%97%E6%B3%95%E7%A4%BA%E6%84%8F%E5%9B%BE.png width = "600" height = "400">


## 网络结构
<img src=https://github.com/yellowyellowyao/ELM_recognition/blob/master/picture/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E7%A4%BA%E6%84%8F%E5%9B%BE.png width = "600" height = "200" />


## 识别效果展示
### 识别效果示意图1：
<img src=https://github.com/yellowyellowyao/ELM_recognition/blob/master/picture/%E8%AF%86%E5%88%AB%E6%95%88%E6%9E%9C%E7%A4%BA%E6%84%8F%E5%9B%BE1.png width = "600" height = "400" />

### 识别效果示意图2：
<img src=https://github.com/yellowyellowyao/ELM_recognition/blob/master/picture/%E8%AF%86%E5%88%AB%E6%95%88%E6%9E%9C%E7%A4%BA%E6%84%8F%E5%9B%BE2.png width = "600" height = "200" />

### 混淆矩阵：
<img src=https://github.com/yellowyellowyao/ELM_recognition/blob/master/picture/%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%98%B5.png width = "400" height = "400" />


## 计算时间分布
<img src=https://github.com/yellowyellowyao/ELM_recognition/blob/master/picture/%E8%AF%86%E5%88%AB%E6%97%B6%E9%97%B4%E6%B5%8B%E8%AF%95%E5%9B%BE.png width = "600" height = "400" />
