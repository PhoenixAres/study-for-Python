# 1. 机器学习的主要任务

监督学习：

分类：将实例数据划分到合适的分类中

回归：用于预测数据值类型
       
无监督学习：

聚类：数据集合分成由类似的对象组成的多个类的过程
        
密度估计：寻找描述数据统计值的过程  
 

# 2. 开发机器学习应用程序的步骤

## 2.1 收集数据

可以通过制作网络爬虫从网站上抽取数据等

## 2.2 准备输入数据

需要为机器学习算法准备特定的数据格式，如一些算法要求目标变量和特征值是字符串类型，而另一些算法可能要求是整数类型

## 2.3 分析输入数据

此步骤主要是人工分析之前得到的数据

最简单的方法是用文本编辑器打开数据文件，查看得到的数据是否为空值

还可以进一步浏览数据，分析是否存在明显的异常值

可以将多维数据压缩到二维或三维，然后图形化数据

## 2.4 训练算法

将前两步得到的格式化的数据输入到算法，从中抽取知识或信息

如果使用无监督算法，由于不存在目标变量值，故而也不需要训练算法

## 2.5 测试算法

对于监督学习，必须已知用于评估算法的目标变量值

对于无监督学习，必须用其他的评测手段来检验算法的成功率

## 2.6 使用算法

# 3. NumPy函数库基础


```python
from numpy import *

random.rand(4, 4)
```




    array([[0.6719326 , 0.64379511, 0.83425991, 0.92481639],
           [0.59553986, 0.59975872, 0.51947045, 0.56702924],
           [0.74516058, 0.22018588, 0.68811214, 0.09055449],
           [0.68828658, 0.36381466, 0.80085059, 0.23872964]])




```python
randMat = random.rand(4, 4)
linalg.pinv(randMat)
```




    array([[-1.53508113,  4.90769988, -6.69841757,  3.08335884],
           [ 1.02638696,  5.93082324, -1.50872219, -3.55703232],
           [ 1.92664425, -7.57125595,  5.85855734, -0.19762387],
           [-0.531746  ,  0.7794424 ,  1.65726608, -1.02036776]])




```python
invRandMat = linalg.pinv(randMat)
dot(randMat, invRandMat)
```




    array([[ 1.00000000e+00, -1.37044166e-15,  3.33547833e-16,
             1.20947586e-15],
           [ 2.42872571e-16,  1.00000000e+00,  1.48729476e-15,
             6.09922987e-16],
           [ 1.14699670e-16, -1.91448404e-15,  1.00000000e+00,
             5.39488428e-16],
           [ 2.75219432e-16, -2.76126016e-16,  1.51508234e-15,
             1.00000000e+00]])




```python
myEye = dot(randMat, invRandMat)
myEye - eye(4)
```




    array([[ 6.66133815e-16, -1.37044166e-15,  3.33547833e-16,
             1.20947586e-15],
           [ 2.42872571e-16, -1.11022302e-15,  1.48729476e-15,
             6.09922987e-16],
           [ 1.14699670e-16, -1.91448404e-15,  1.77635684e-15,
             5.39488428e-16],
           [ 2.75219432e-16, -2.76126016e-16,  1.51508234e-15,
             2.22044605e-16]])


