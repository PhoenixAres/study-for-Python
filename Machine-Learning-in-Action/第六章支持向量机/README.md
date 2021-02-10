# 1. 基于最大间隔分隔数据

支持向量机

优点：泛化错误率低，计算开销不大，结果易解释

缺点：对参数调节和核函数的选择敏感，原始分类器不加修改仅适用于处理二类问题

适用数据类型：数值型和标称型数据

当数据点都在二维平面上，如果可以用一条直线将两组数据点分开，那么此时这组数据成为线性可分(linearly separable)数据

上述将数据集分隔开来的直线称为分隔超平面(separating hyperplane)

当数据集是n维时，则需要一个n-1维的对象来对数据进行分隔，该对象称为超平面(hyperplane)，也就是分类的决策边界。

我们希望能采用这种方式来构建分类器，即如果数据点离决策边界越远，那么其最后的预测结果也就越可信。

我们希望找到离分隔超平面最近的点，确保它们离分隔面的距离尽可能远。这里点到分隔面的距离被称为间隔(margin)。

我们希望间隔尽可能地大，这是因为如果我们犯错或者在有限数据上训练分类器的话，我们希望分类器尽可能健壮。

支持向量(support vector)就是离分隔超平面最近的那些点，接下来要试着最大化支持向量到分隔面的距离。

# 2. 寻找最大间隔

分隔超平面的形式可以写成：

$$ w^Tx+b $$

要计算点A到分隔超平面的距离，就必须给出点到分隔面的法线或垂线的长度，该值为：

$$ \frac{|w^TA+b|}{||w||} $$

这里的常数$b$类似于Logistic回归中的截距$w_0$

## 2.1 分类器求解的优化问题

输入数据给分类器会输出一个类别标签，这相当于一个类似于Sigmoid的函数在作用。

下面将使用类似海维赛德阶跃函数（即单位阶跃函数）的函数对$w^Tx+b$作用得到$f(w^Tx+b)$，其中当u<0时f(u)输出-1，反之则输出+1。

这里标签采用-1和+1，而不是0和1，这是由于-1和+1仅仅相差一个符号，方便数学上的处理。

当计算数据点到分隔面的距离并确定分隔面的放置位置时，间隔通过$label * (w^Tx+b)$来计算，

如果数据点处于正方向（即+1类）并且离分隔超平面很远的位置时，$w^Tx+b$会是一个很大的正数，同时$label * (w^Tx+b)$也会是一个很大的正数。

如果数据点处于负方向（即-1类）并且离分隔超平面很远的位置时，此时由于类别标签为-1，则$label * (w^Tx+b)$仍然是一个很大的正数。

注：其中$label * (w^Tx+b)$被称为点到分隔面的函数间隔，而$label * (w^Tx+b)*\frac{1}{||w||}$称为点到分隔面的几何间隔

我们需要找到具有最小间隔的数据点（即支持向量），找到以后对该间隔最大化：

$$ \underset{w,b}{argmax}\lbrace\underset{n}{min}(label\cdot (w^Tx+b))\cdot \frac{1}{||w||}\rbrace $$

直接求解相当困难，考察上式大括号内的部分，如果令所有支持向量的$label*(w^Tx+b)$都为1，则可以通过求解$||w||^{-1}$的最大值来得到最终解。

但并非所有数据点的$label*(w^Tx+b)$都等于1，只有那些离分隔超平面最近的点得到的值才为1，而离分隔超平面越远，其$label*(w^Tx+b)$的值也就越大。

上述优化问题是一个带约束条件的优化问题，这里的优化条件即为$label*(w^Tx+b)\geq 1.0$。

对于这类优化问题，可以使用拉格朗日乘子法，因此我们可以将超平面写成数据点的形式，于是，优化目标函数写成：

$$ \underset{\alpha}{max}\{ \sum_{i=1}^{m}\alpha - \frac{1}{2}\sum_{i,j=1}^{m}label^{(i)}\cdot label^{(j)}\cdot \alpha_i \cdot \alpha_j  \langle {x^{(i)},x^{(j)}} \rangle\} $$

其中，$\langle {x^{(i)},x^{(j)}} \rangle$表示$x^{(i)}$和$x^{(j)}$两个向量的内积，且上式的约束条件为：

$$ \alpha \geq 0，和 \sum_{i=1}^{m}\alpha_i \cdot label^{(i)} = 0 $$

至此，一切都很完美，但是这里有个假设：数据必须100%线性可分。这时我们可以引入松弛变量(slack variable)来允许有些数据点可以处于分隔面的错误一侧。

这样，我们的优化目标就能保持不变，但是约束条件变为：

$$ C \geq \alpha \geq 0，和 \sum_{i=1}^{m}\alpha_i \cdot label^{(i)} = 0 $$

这里的常数C用于控制“最大化间隔”和“保证大部分点的函数间隔小于1.0”这两个目标的权重。

在优化算法的实现代码中，常数C是一个参数，因此我们就可以通过调节该参数得到不同的结果。

一旦求出所有的alpha，那么分隔超平面就可以通过这些alpha来表达，SVM的主要工作就是求解这些alpha。

## 2.2 SVM应用的一般框架

SVM的一般流程：

（1）收集数据：可以使用任何方法

（2）准备数据：需要数值型数据

（3）分析数据：有助于可视化分隔超平面

（4）训练算法：SVM的大部分时间都源自训练，该过程主要实现两个参数的调优

（5）测试算法：十分简单的计算过程就可以实现

（6）使用算法：几乎所用分类问题都可以用SVM，但SVM本身是一个二类分类器，对多类问题应用SVM需要对代码做一些修改

# 3. SMO高效优化算法

## 3.1 Platt的SMO算法

SMO表示序列最小优化(Sequential Minimal Optimization)。Platt的SMO算法是将大优化问题分解为多个小优化问题来求解的。

SMO算法的目标是求出一系列alpha和b，一旦求出了这些alpha，就很容易计算出权重向量$w$，并得到分隔超平面。

SMO算法的工作原理：每次循环中选择两个alpha进行优化处理。一旦找到一对合适的alpha，那么就增大其中一个同时减小另一个。

这里的“合适”指需要符合一定条件，条件之一是这两个alpha必须要在间隔边界之外，之二是这两个alpha还没有进行过区间化处理或者不在边界上。

## 3.2 应用简化版SMO算法处理小规模数据集

Platt SMO算法中的外循环确定要优化的最佳alpha对，而简化版会跳过这一部分。

首先在数据集上遍历每一个alpha，再在剩下的alpha集合中随机选择另一个alpha，从而构成alpha对。


```python
from numpy import *

def loadDataSet(filename):
    dataMat = []
    labelMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    aj = min(aj, H)
    aj = max(aj, L)
    return aj

dataArr, labelArr = loadDataSet('testSet.txt')
print(labelArr)
```

    [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
    

SMO函数的伪代码大致如下：

    创建一个alpha向量并将其初始化为0向量
    当迭代次数小于最大迭代次数时（外循环）
        对数据集中的每个数据向量（内循环）：
            如果该数据向量可以被优化：
                随机选择另外一个数据向量
                同时优化这两个向量
                如果两个向量都不能优化，退出内循环
        如果所有向量都没有被优化，增加迭代数目，继续下一次循环


```python
def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).T
    b = 0
    m, n = shape(dataMatrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            if labelMat[i] * Ei < -toler and alphas[i] < C or labelMat[i] * Ei > toler and alphas[i] > 0:
                                                                 #如果alpha可以更改进入优化过程
                j = selectJrand(i, m)                            #随机选择第二个alpha
                fXj = float(multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelMat[i] != labelMat[j]:                   #保证alpha在0与C之间
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print('L == H')
                    continue
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - \
                            dataMatrix[i, :] * dataMatrix[i, :].T - \
                            dataMatrix[j, :] * dataMatrix[j, :].T
                if eta >= 0:
                    print('eta >= 0')
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print('j not moving enough')
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])   #对i进行修改，修改量与j相同，但方向相反
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                              labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                              labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:               #设置常数项
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print('iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print('iteration number: %d' % iter)
    return b, alphas


b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
```

    iter: 0 i: 0, pairs changed 1
    L == H
    j not moving enough
    L == H
    L == H
    L == H
    j not moving enough
    j not moving enough
    L == H
    L == H
    L == H
    iter: 0 i: 69, pairs changed 2
    L == H
    j not moving enough
    L == H
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    L == H
    j not moving enough
    iteration number: 0
    j not moving enough
    L == H
    L == H
    L == H
    L == H
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    L == H
    j not moving enough
    j not moving enough
    L == H
    iter: 0 i: 30, pairs changed 1
    j not moving enough
    L == H
    iter: 0 i: 52, pairs changed 2
    L == H
    iter: 0 i: 55, pairs changed 3
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    L == H
    L == H
    iteration number: 0
    j not moving enough
    L == H
    L == H
    L == H
    L == H
    L == H
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    L == H
    L == H
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    iter: 0 i: 69, pairs changed 1
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 55, pairs changed 1
    j not moving enough
    L == H
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    L == H
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    iter: 0 i: 37, pairs changed 1
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    L == H
    L == H
    L == H
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 96, pairs changed 2
    L == H
    iteration number: 0
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 69, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 96, pairs changed 1
    L == H
    iteration number: 0
    iter: 0 i: 0, pairs changed 1
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    iter: 1 i: 30, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 1 i: 87, pairs changed 2
    L == H
    j not moving enough
    j not moving enough
    L == H
    iteration number: 0
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    iter: 0 i: 24, pairs changed 1
    iter: 0 i: 26, pairs changed 2
    j not moving enough
    iter: 0 i: 34, pairs changed 3
    L == H
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    iter: 0 i: 54, pairs changed 1
    iter: 0 i: 55, pairs changed 2
    iter: 0 i: 69, pairs changed 3
    iteration number: 0
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    iter: 0 i: 23, pairs changed 1
    iter: 0 i: 24, pairs changed 2
    iter: 0 i: 27, pairs changed 3
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    iter: 0 i: 96, pairs changed 4
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    L == H
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    iter: 0 i: 97, pairs changed 1
    iteration number: 0
    j not moving enough
    j not moving enough
    iter: 0 i: 8, pairs changed 1
    j not moving enough
    j not moving enough
    iter: 0 i: 17, pairs changed 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 54, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 96, pairs changed 1
    j not moving enough
    iteration number: 0
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    iteration number: 1
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    iter: 1 i: 30, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 69, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    iter: 0 i: 10, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 55, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    L == H
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    iteration number: 2
    L == H
    j not moving enough
    iter: 2 i: 8, pairs changed 1
    iter: 2 i: 10, pairs changed 2
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 2 i: 54, pairs changed 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 24, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 23, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 17, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    iter: 2 i: 10, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 24, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    iter: 0 i: 8, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 57, pairs changed 2
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 23, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 54, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 3 i: 23, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 1 i: 29, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 3 i: 17, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 3 i: 52, pairs changed 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    L == H
    j not moving enough
    iter: 3 i: 17, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 13
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 14
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 15
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 16
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 17
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 18
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 19
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 19 i: 52, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 55, pairs changed 1
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    iter: 2 i: 17, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 1 i: 54, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 1 i: 17, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 13
    j not moving enough
    iter: 13 i: 23, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 2 i: 52, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 3 i: 54, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 55, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 29, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 55, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 4 i: 52, pairs changed 1
    j not moving enough
    iter: 4 i: 55, pairs changed 2
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 13
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 14
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 15
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 16
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 17
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 18
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 19
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 20
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 21
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 21 i: 54, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 52, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 5 i: 54, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 6 i: 55, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 5 i: 54, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 13
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 14
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 15
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 16
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 17
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 18
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 19
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 20
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 20 i: 55, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 13
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 14
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 15
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 16
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 17
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 18
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 19
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 20
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 20 i: 52, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 3 i: 29, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 6 i: 52, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    iter: 1 i: 55, pairs changed 1
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    j not moving enough
    iter: 12 i: 29, pairs changed 1
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    iter: 0 i: 23, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 55, pairs changed 2
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 5 i: 54, pairs changed 1
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    iter: 0 i: 29, pairs changed 1
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    iter: 5 i: 17, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 13
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 14
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 15
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 16
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 17
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 18
    j not moving enough
    j not moving enough
    iter: 18 i: 52, pairs changed 1
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    iter: 0 i: 54, pairs changed 1
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 1 i: 55, pairs changed 1
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 5 i: 54, pairs changed 1
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 4 i: 52, pairs changed 1
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    iter: 0 i: 23, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    L == H
    j not moving enough
    iter: 0 i: 55, pairs changed 1
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    iter: 1 i: 23, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 4 i: 55, pairs changed 1
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 9 i: 54, pairs changed 1
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    iter: 3 i: 29, pairs changed 1
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 13
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 14
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 15
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 16
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 17
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 18
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 19
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 20
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 21
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 22
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 23
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 24
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 25
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 26
    iter: 26 i: 17, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 13
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 14
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 15
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 16
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 17
    j not moving enough
    iter: 17 i: 29, pairs changed 1
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    iter: 10 i: 17, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 13
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 14
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 15
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 16
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 17
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 18
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 19
    j not moving enough
    j not moving enough
    iter: 19 i: 54, pairs changed 1
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    iteration number: 7
    iter: 7 i: 17, pairs changed 1
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    iteration number: 12
    j not moving enough
    j not moving enough
    iteration number: 13
    j not moving enough
    j not moving enough
    iteration number: 14
    j not moving enough
    j not moving enough
    iteration number: 15
    j not moving enough
    j not moving enough
    iteration number: 16
    j not moving enough
    j not moving enough
    iteration number: 17
    j not moving enough
    j not moving enough
    iteration number: 18
    j not moving enough
    j not moving enough
    iteration number: 19
    j not moving enough
    j not moving enough
    iteration number: 20
    iter: 20 i: 54, pairs changed 1
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 13
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 14
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 15
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 16
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    L == H
    iter: 16 i: 55, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 13
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 14
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 15
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 16
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 17
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 18
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 19
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 20
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 21
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 22
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 23
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 24
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 25
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 26
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 27
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 28
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 29
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 30
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 31
    j not moving enough
    iter: 31 i: 17, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 9 i: 55, pairs changed 1
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 13
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 14
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 15
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 16
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 17
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 18
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 19
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 20
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 21
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 22
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 23
    j not moving enough
    j not moving enough
    iter: 23 i: 17, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 0 i: 54, pairs changed 1
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iter: 5 i: 29, pairs changed 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 0
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 1
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 2
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 3
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 4
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 5
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 6
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 7
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 8
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 9
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 10
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 11
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 12
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 13
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 14
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 15
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 16
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 17
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 18
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 19
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 20
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 21
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 22
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 23
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 24
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 25
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 26
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 27
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 28
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 29
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 30
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 31
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 32
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 33
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 34
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 35
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 36
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 37
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 38
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 39
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    j not moving enough
    iteration number: 40
    


```python
b
```




    matrix([[-3.84848578]])




```python
alphas[alphas > 0]
```




    matrix([[1.08420217e-16, 4.72712147e-17, 1.27867992e-01, 2.36356074e-17,
             2.36356074e-17, 2.42270373e-01, 1.08420217e-16, 2.02962647e-16,
             2.36356074e-17, 3.70135180e-01]])




```python
shape(alphas[alphas > 0])
```




    (1, 10)




```python
for i in range(100):
    if alphas[i] > 0.0:
        print(dataArr[i], labelArr[i])
```

    [3.542485, 1.977398] -1.0
    [2.326297, 0.265213] -1.0
    [4.658191, 3.507396] -1.0
    [3.223038, -0.552392] -1.0
    [2.301095, -0.533988] -1.0
    [3.457096, -0.082216] -1.0
    [3.023938, -0.057392] -1.0
    [2.893743, -1.643468] -1.0
    [1.870457, -1.04042] -1.0
    [6.080573, 0.418886] 1.0
    

# 4. 利用完整的Platt SMO算法加速优化

Platt SMO算法是通过一个外循环来选择第一个alpha值的，并且其选择过程会在两种方式之间进行交替：

一种是在所有数据集上进行单遍扫描，另一种则是在非边界alpha中实现单遍扫描。

在选择第一个alpha值后，算法会通过一个内循环来选择第二个alpha值，在优化过程中，会通过最大化步长的方式获得第二个alpha值。

我们会建立一个全局的缓存用于保存误差值，并从中选择使得步长或者说Ei-Ej最大的alpha值。


```python
class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))        #误差缓存

def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i, oS, Ei):           #内循环中的启发式方法
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:          #选择具有最大步长的j
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if (oS.labelMat[i] * Ei < -oS.tol and oS.alphas[i] < oS.C) or (oS.labelMat[i] * Ei > oS.tol and oS.alphas[i] > 0):
        j, Ej = selectJ(i, oS, Ei)            # 第二个alpha选择中的启发式方法
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print('L == H')
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print('eta >= 0')
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)                           #更新误差缓存
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print('j not moving enough')
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)                           #更新误差缓存
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - \
             oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j] < oS.C:  # 设置常数项
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0
    
def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).T, C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        if entireSet:                 #遍历所有值
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print('fullSet, iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1
        else:                         #遍历非边界值
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print('non-bound, iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print('iteration number: %d' % iter)
    return oS.b, oS.alphas

dataArr, labelArr = loadDataSet('testSet.txt')
b, alphas = smoP(dataArr, labelArr, 0.6, 0.001, 40)
```

    fullSet, iter: 0 i: 0, pairs changed 1
    fullSet, iter: 0 i: 1, pairs changed 1
    fullSet, iter: 0 i: 2, pairs changed 1
    fullSet, iter: 0 i: 3, pairs changed 1
    fullSet, iter: 0 i: 4, pairs changed 2
    fullSet, iter: 0 i: 5, pairs changed 3
    fullSet, iter: 0 i: 6, pairs changed 4
    fullSet, iter: 0 i: 7, pairs changed 4
    j not moving enough
    fullSet, iter: 0 i: 8, pairs changed 4
    fullSet, iter: 0 i: 9, pairs changed 4
    fullSet, iter: 0 i: 10, pairs changed 5
    fullSet, iter: 0 i: 11, pairs changed 5
    fullSet, iter: 0 i: 12, pairs changed 5
    fullSet, iter: 0 i: 13, pairs changed 5
    fullSet, iter: 0 i: 14, pairs changed 5
    fullSet, iter: 0 i: 15, pairs changed 5
    fullSet, iter: 0 i: 16, pairs changed 5
    fullSet, iter: 0 i: 17, pairs changed 5
    L == H
    fullSet, iter: 0 i: 18, pairs changed 5
    fullSet, iter: 0 i: 19, pairs changed 5
    fullSet, iter: 0 i: 20, pairs changed 5
    fullSet, iter: 0 i: 21, pairs changed 5
    fullSet, iter: 0 i: 22, pairs changed 5
    fullSet, iter: 0 i: 23, pairs changed 6
    j not moving enough
    fullSet, iter: 0 i: 24, pairs changed 6
    fullSet, iter: 0 i: 25, pairs changed 7
    L == H
    fullSet, iter: 0 i: 26, pairs changed 7
    fullSet, iter: 0 i: 27, pairs changed 7
    fullSet, iter: 0 i: 28, pairs changed 7
    fullSet, iter: 0 i: 29, pairs changed 8
    fullSet, iter: 0 i: 30, pairs changed 8
    fullSet, iter: 0 i: 31, pairs changed 8
    fullSet, iter: 0 i: 32, pairs changed 8
    fullSet, iter: 0 i: 33, pairs changed 8
    fullSet, iter: 0 i: 34, pairs changed 8
    fullSet, iter: 0 i: 35, pairs changed 8
    fullSet, iter: 0 i: 36, pairs changed 8
    fullSet, iter: 0 i: 37, pairs changed 8
    fullSet, iter: 0 i: 38, pairs changed 8
    fullSet, iter: 0 i: 39, pairs changed 8
    fullSet, iter: 0 i: 40, pairs changed 8
    fullSet, iter: 0 i: 41, pairs changed 8
    fullSet, iter: 0 i: 42, pairs changed 8
    fullSet, iter: 0 i: 43, pairs changed 8
    fullSet, iter: 0 i: 44, pairs changed 8
    fullSet, iter: 0 i: 45, pairs changed 8
    L == H
    fullSet, iter: 0 i: 46, pairs changed 8
    fullSet, iter: 0 i: 47, pairs changed 8
    fullSet, iter: 0 i: 48, pairs changed 8
    fullSet, iter: 0 i: 49, pairs changed 8
    fullSet, iter: 0 i: 50, pairs changed 8
    fullSet, iter: 0 i: 51, pairs changed 8
    fullSet, iter: 0 i: 52, pairs changed 8
    fullSet, iter: 0 i: 53, pairs changed 8
    L == H
    fullSet, iter: 0 i: 54, pairs changed 8
    L == H
    fullSet, iter: 0 i: 55, pairs changed 8
    fullSet, iter: 0 i: 56, pairs changed 8
    L == H
    fullSet, iter: 0 i: 57, pairs changed 8
    fullSet, iter: 0 i: 58, pairs changed 8
    fullSet, iter: 0 i: 59, pairs changed 8
    fullSet, iter: 0 i: 60, pairs changed 8
    fullSet, iter: 0 i: 61, pairs changed 8
    fullSet, iter: 0 i: 62, pairs changed 8
    fullSet, iter: 0 i: 63, pairs changed 8
    fullSet, iter: 0 i: 64, pairs changed 8
    fullSet, iter: 0 i: 65, pairs changed 8
    fullSet, iter: 0 i: 66, pairs changed 8
    fullSet, iter: 0 i: 67, pairs changed 8
    fullSet, iter: 0 i: 68, pairs changed 8
    L == H
    fullSet, iter: 0 i: 69, pairs changed 8
    fullSet, iter: 0 i: 70, pairs changed 8
    fullSet, iter: 0 i: 71, pairs changed 8
    fullSet, iter: 0 i: 72, pairs changed 8
    fullSet, iter: 0 i: 73, pairs changed 8
    fullSet, iter: 0 i: 74, pairs changed 8
    fullSet, iter: 0 i: 75, pairs changed 8
    fullSet, iter: 0 i: 76, pairs changed 8
    fullSet, iter: 0 i: 77, pairs changed 8
    fullSet, iter: 0 i: 78, pairs changed 8
    fullSet, iter: 0 i: 79, pairs changed 8
    fullSet, iter: 0 i: 80, pairs changed 8
    fullSet, iter: 0 i: 81, pairs changed 8
    fullSet, iter: 0 i: 82, pairs changed 8
    fullSet, iter: 0 i: 83, pairs changed 8
    fullSet, iter: 0 i: 84, pairs changed 8
    fullSet, iter: 0 i: 85, pairs changed 8
    fullSet, iter: 0 i: 86, pairs changed 8
    fullSet, iter: 0 i: 87, pairs changed 8
    fullSet, iter: 0 i: 88, pairs changed 8
    fullSet, iter: 0 i: 89, pairs changed 8
    fullSet, iter: 0 i: 90, pairs changed 8
    fullSet, iter: 0 i: 91, pairs changed 8
    fullSet, iter: 0 i: 92, pairs changed 8
    fullSet, iter: 0 i: 93, pairs changed 8
    fullSet, iter: 0 i: 94, pairs changed 8
    fullSet, iter: 0 i: 95, pairs changed 8
    fullSet, iter: 0 i: 96, pairs changed 8
    fullSet, iter: 0 i: 97, pairs changed 8
    fullSet, iter: 0 i: 98, pairs changed 8
    fullSet, iter: 0 i: 99, pairs changed 8
    iteration number: 1
    j not moving enough
    non-bound, iter: 1 i: 0, pairs changed 0
    non-bound, iter: 1 i: 4, pairs changed 1
    non-bound, iter: 1 i: 5, pairs changed 2
    non-bound, iter: 1 i: 10, pairs changed 3
    j not moving enough
    non-bound, iter: 1 i: 23, pairs changed 3
    non-bound, iter: 1 i: 25, pairs changed 4
    non-bound, iter: 1 i: 26, pairs changed 5
    j not moving enough
    non-bound, iter: 1 i: 29, pairs changed 5
    iteration number: 2
    j not moving enough
    non-bound, iter: 2 i: 0, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 10, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 23, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 29, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 54, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 55, pairs changed 0
    iteration number: 3
    j not moving enough
    fullSet, iter: 3 i: 0, pairs changed 0
    fullSet, iter: 3 i: 1, pairs changed 0
    fullSet, iter: 3 i: 2, pairs changed 0
    fullSet, iter: 3 i: 3, pairs changed 0
    fullSet, iter: 3 i: 4, pairs changed 0
    fullSet, iter: 3 i: 5, pairs changed 0
    fullSet, iter: 3 i: 6, pairs changed 0
    fullSet, iter: 3 i: 7, pairs changed 0
    fullSet, iter: 3 i: 8, pairs changed 0
    fullSet, iter: 3 i: 9, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 10, pairs changed 0
    fullSet, iter: 3 i: 11, pairs changed 0
    fullSet, iter: 3 i: 12, pairs changed 0
    fullSet, iter: 3 i: 13, pairs changed 0
    fullSet, iter: 3 i: 14, pairs changed 0
    fullSet, iter: 3 i: 15, pairs changed 0
    fullSet, iter: 3 i: 16, pairs changed 0
    fullSet, iter: 3 i: 17, pairs changed 0
    fullSet, iter: 3 i: 18, pairs changed 0
    fullSet, iter: 3 i: 19, pairs changed 0
    fullSet, iter: 3 i: 20, pairs changed 0
    fullSet, iter: 3 i: 21, pairs changed 0
    fullSet, iter: 3 i: 22, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 23, pairs changed 0
    fullSet, iter: 3 i: 24, pairs changed 0
    fullSet, iter: 3 i: 25, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 26, pairs changed 0
    fullSet, iter: 3 i: 27, pairs changed 0
    fullSet, iter: 3 i: 28, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 29, pairs changed 0
    fullSet, iter: 3 i: 30, pairs changed 0
    fullSet, iter: 3 i: 31, pairs changed 0
    fullSet, iter: 3 i: 32, pairs changed 0
    fullSet, iter: 3 i: 33, pairs changed 0
    fullSet, iter: 3 i: 34, pairs changed 0
    fullSet, iter: 3 i: 35, pairs changed 0
    fullSet, iter: 3 i: 36, pairs changed 0
    fullSet, iter: 3 i: 37, pairs changed 0
    fullSet, iter: 3 i: 38, pairs changed 0
    fullSet, iter: 3 i: 39, pairs changed 0
    fullSet, iter: 3 i: 40, pairs changed 0
    fullSet, iter: 3 i: 41, pairs changed 0
    fullSet, iter: 3 i: 42, pairs changed 0
    fullSet, iter: 3 i: 43, pairs changed 0
    fullSet, iter: 3 i: 44, pairs changed 0
    fullSet, iter: 3 i: 45, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 46, pairs changed 0
    fullSet, iter: 3 i: 47, pairs changed 0
    fullSet, iter: 3 i: 48, pairs changed 0
    fullSet, iter: 3 i: 49, pairs changed 0
    fullSet, iter: 3 i: 50, pairs changed 0
    fullSet, iter: 3 i: 51, pairs changed 0
    fullSet, iter: 3 i: 52, pairs changed 0
    fullSet, iter: 3 i: 53, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 54, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 55, pairs changed 0
    fullSet, iter: 3 i: 56, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 57, pairs changed 0
    fullSet, iter: 3 i: 58, pairs changed 0
    fullSet, iter: 3 i: 59, pairs changed 0
    fullSet, iter: 3 i: 60, pairs changed 0
    fullSet, iter: 3 i: 61, pairs changed 0
    fullSet, iter: 3 i: 62, pairs changed 0
    fullSet, iter: 3 i: 63, pairs changed 0
    fullSet, iter: 3 i: 64, pairs changed 0
    fullSet, iter: 3 i: 65, pairs changed 0
    fullSet, iter: 3 i: 66, pairs changed 0
    fullSet, iter: 3 i: 67, pairs changed 0
    fullSet, iter: 3 i: 68, pairs changed 0
    L == H
    fullSet, iter: 3 i: 69, pairs changed 0
    fullSet, iter: 3 i: 70, pairs changed 0
    fullSet, iter: 3 i: 71, pairs changed 0
    fullSet, iter: 3 i: 72, pairs changed 0
    fullSet, iter: 3 i: 73, pairs changed 0
    fullSet, iter: 3 i: 74, pairs changed 0
    fullSet, iter: 3 i: 75, pairs changed 0
    fullSet, iter: 3 i: 76, pairs changed 0
    fullSet, iter: 3 i: 77, pairs changed 0
    fullSet, iter: 3 i: 78, pairs changed 0
    fullSet, iter: 3 i: 79, pairs changed 0
    fullSet, iter: 3 i: 80, pairs changed 0
    fullSet, iter: 3 i: 81, pairs changed 0
    fullSet, iter: 3 i: 82, pairs changed 0
    fullSet, iter: 3 i: 83, pairs changed 0
    fullSet, iter: 3 i: 84, pairs changed 0
    fullSet, iter: 3 i: 85, pairs changed 0
    fullSet, iter: 3 i: 86, pairs changed 0
    fullSet, iter: 3 i: 87, pairs changed 0
    fullSet, iter: 3 i: 88, pairs changed 0
    fullSet, iter: 3 i: 89, pairs changed 0
    fullSet, iter: 3 i: 90, pairs changed 0
    fullSet, iter: 3 i: 91, pairs changed 0
    fullSet, iter: 3 i: 92, pairs changed 0
    fullSet, iter: 3 i: 93, pairs changed 0
    fullSet, iter: 3 i: 94, pairs changed 0
    fullSet, iter: 3 i: 95, pairs changed 0
    fullSet, iter: 3 i: 96, pairs changed 0
    fullSet, iter: 3 i: 97, pairs changed 0
    fullSet, iter: 3 i: 98, pairs changed 0
    fullSet, iter: 3 i: 99, pairs changed 0
    iteration number: 4
    


```python
def calcWs(alphas, dataArr, classLabels):
    X = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w

ws = calcWs(alphas, dataArr, labelArr)
ws
```




    array([[ 0.57421375],
           [-0.17224914]])




```python
dataMat = mat(dataArr)
print(dataMat[0] * mat(ws) + b)
print(labelArr[0])
```

    [[-1.48292458]]
    -1.0
    


```python
print(dataMat[2] * mat(ws) + b)
print(labelArr[2])
print(dataMat[1] * mat(ws) + b)
print(labelArr[1])
```

    [[1.4318766]]
    1.0
    [[-1.88331193]]
    -1.0
    

# 5. 在复杂数据上应用核函数

## 5.1 利用核函数将数据映射到高维空间

当二维数据点处于一个圆中，人类的大脑可以意识到这一点。然而，对于分类器而言，它只能识别分类器的结果是否大于0。

此时，需要将低维特征空间映射到高维空间，通过核函数实现。

核函数(kernel)可以看成一个包装器(wrapper)或者是接口(interface)，它能将数据从某个很难处理的形式转换成为另一个较容易处理的形式。

通俗来说，我们在高维空间解决线性问题，等价于在低维空间解决非线性问题。

SVM优化中，所有的运算都可以写成内积(inner product，也称点积)的形式，我们可以把内积运算替换成核函数，而不需要做简化处理。

将内积替换为核函数的方法称为核技巧(kernel trick)或者核变电(kernel substation)。

## 5.2 径向基核函数

径向基函数是一个采用向量作为自变量的函数，能够基于向量距离运算输出一个标量。这个距离可以是从<0,0>向量或者其他向量开始计算的距离。

径向基函数的高斯版本，具体公式如下：

$$ k(x,y) = exp(\frac {-||x-y||^2} {2\sigma^2})$$ 

其中，$\sigma$是用户定义的用于确定到达率(reach)或者说函数值跌落到0的速度参数。

高斯核函数将数据从特征空间映射到更高维的空间，具体来说这里是映射到一个无穷维的空间。

使用高斯核函数并不需要理解数据是如何表现的，依然可以得到一个理想的结果。


```python
def kernelTrans(X, A, kTup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if kTup[0] == 'lin':
        K = X * A.T
    elif kTup[0] == 'rbf':
        for j in range(m):
            deltaRow = X[j, :] - A
            K[j] = deltaRow * deltaRow.T
        K = exp(K / (-1 * kTup[1]**2))       #元素间的除法
    else:
        raise NameError('Houston we Have a Problem -- That Kernel is not recognized')
    return K

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler, kTup):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))        #误差缓存
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernelTrans(self.X, self.X[i, :], kTup)
            
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * oS.K[:, k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if (oS.labelMat[i] * Ei < -oS.tol and oS.alphas[i] < oS.C) or (oS.labelMat[i] * Ei > oS.tol and oS.alphas[i] > 0):
        j, Ej = selectJ(i, oS, Ei)            # 第二个alpha选择中的启发式方法
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print('L == H')
            return 0
        eta = 2.0 * oS.K[i, j] - oS.K[i, i] - oS.K[j, j]
        if eta >= 0:
            print('eta >= 0')
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)                           #更新误差缓存
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print('j not moving enough')
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)                           #更新误差缓存
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - \
                         oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - \
                         oS.labelMat[j] * (oS.alphas[j] - alphaJold) * oS.K[j, j]
        if 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j] < oS.C:  # 设置常数项
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter, kTup = ('lin', 0)):
    oS = optStruct(mat(dataMatIn), mat(classLabels).T, C, toler, kTup)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        if entireSet:                 #遍历所有值
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print('fullSet, iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1
        else:                         #遍历非边界值
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print('non-bound, iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print('iteration number: %d' % iter)
    return oS.b, oS.alphas
```

## 5.3 在测试中使用核函数


```python
def testRbf(k1 = 1.3):
    dataArr, labelArr = loadDataSet('testSetRBF.txt')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1))
    datMat = mat(dataArr)
    labelMat = mat(labelArr).T
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]                 #构建支持向量矩阵
    labelSV = labelMat[svInd]
    print('there are %d Support Vectors' % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print('the training error rate is: %f' % (float(errorCount) / m))
    dataArr, labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).T
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], ('rbf', k1))
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print('the test error rate is: %f' % (float(errorCount) / m))
    
testRbf()
```

    fullSet, iter: 0 i: 0, pairs changed 1
    fullSet, iter: 0 i: 1, pairs changed 2
    fullSet, iter: 0 i: 2, pairs changed 3
    fullSet, iter: 0 i: 3, pairs changed 4
    fullSet, iter: 0 i: 4, pairs changed 5
    fullSet, iter: 0 i: 5, pairs changed 6
    fullSet, iter: 0 i: 6, pairs changed 7
    fullSet, iter: 0 i: 7, pairs changed 7
    fullSet, iter: 0 i: 8, pairs changed 8
    fullSet, iter: 0 i: 9, pairs changed 8
    fullSet, iter: 0 i: 10, pairs changed 9
    fullSet, iter: 0 i: 11, pairs changed 10
    L == H
    fullSet, iter: 0 i: 12, pairs changed 10
    L == H
    fullSet, iter: 0 i: 13, pairs changed 10
    L == H
    fullSet, iter: 0 i: 14, pairs changed 10
    fullSet, iter: 0 i: 15, pairs changed 11
    fullSet, iter: 0 i: 16, pairs changed 12
    fullSet, iter: 0 i: 17, pairs changed 12
    fullSet, iter: 0 i: 18, pairs changed 13
    fullSet, iter: 0 i: 19, pairs changed 14
    fullSet, iter: 0 i: 20, pairs changed 14
    fullSet, iter: 0 i: 21, pairs changed 15
    fullSet, iter: 0 i: 22, pairs changed 15
    fullSet, iter: 0 i: 23, pairs changed 15
    fullSet, iter: 0 i: 24, pairs changed 16
    j not moving enough
    fullSet, iter: 0 i: 25, pairs changed 16
    fullSet, iter: 0 i: 26, pairs changed 16
    j not moving enough
    fullSet, iter: 0 i: 27, pairs changed 16
    fullSet, iter: 0 i: 28, pairs changed 17
    fullSet, iter: 0 i: 29, pairs changed 18
    fullSet, iter: 0 i: 30, pairs changed 18
    fullSet, iter: 0 i: 31, pairs changed 19
    fullSet, iter: 0 i: 32, pairs changed 19
    fullSet, iter: 0 i: 33, pairs changed 19
    fullSet, iter: 0 i: 34, pairs changed 20
    fullSet, iter: 0 i: 35, pairs changed 20
    fullSet, iter: 0 i: 36, pairs changed 21
    fullSet, iter: 0 i: 37, pairs changed 21
    fullSet, iter: 0 i: 38, pairs changed 21
    fullSet, iter: 0 i: 39, pairs changed 21
    fullSet, iter: 0 i: 40, pairs changed 21
    fullSet, iter: 0 i: 41, pairs changed 22
    fullSet, iter: 0 i: 42, pairs changed 23
    L == H
    fullSet, iter: 0 i: 43, pairs changed 23
    fullSet, iter: 0 i: 44, pairs changed 23
    fullSet, iter: 0 i: 45, pairs changed 24
    fullSet, iter: 0 i: 46, pairs changed 25
    fullSet, iter: 0 i: 47, pairs changed 25
    fullSet, iter: 0 i: 48, pairs changed 25
    fullSet, iter: 0 i: 49, pairs changed 25
    j not moving enough
    fullSet, iter: 0 i: 50, pairs changed 25
    fullSet, iter: 0 i: 51, pairs changed 25
    fullSet, iter: 0 i: 52, pairs changed 25
    L == H
    fullSet, iter: 0 i: 53, pairs changed 25
    fullSet, iter: 0 i: 54, pairs changed 25
    fullSet, iter: 0 i: 55, pairs changed 25
    fullSet, iter: 0 i: 56, pairs changed 26
    fullSet, iter: 0 i: 57, pairs changed 26
    fullSet, iter: 0 i: 58, pairs changed 26
    fullSet, iter: 0 i: 59, pairs changed 26
    j not moving enough
    fullSet, iter: 0 i: 60, pairs changed 26
    fullSet, iter: 0 i: 61, pairs changed 27
    fullSet, iter: 0 i: 62, pairs changed 28
    fullSet, iter: 0 i: 63, pairs changed 28
    fullSet, iter: 0 i: 64, pairs changed 28
    fullSet, iter: 0 i: 65, pairs changed 28
    fullSet, iter: 0 i: 66, pairs changed 28
    fullSet, iter: 0 i: 67, pairs changed 28
    fullSet, iter: 0 i: 68, pairs changed 28
    fullSet, iter: 0 i: 69, pairs changed 28
    fullSet, iter: 0 i: 70, pairs changed 28
    fullSet, iter: 0 i: 71, pairs changed 28
    fullSet, iter: 0 i: 72, pairs changed 28
    fullSet, iter: 0 i: 73, pairs changed 28
    fullSet, iter: 0 i: 74, pairs changed 29
    fullSet, iter: 0 i: 75, pairs changed 29
    fullSet, iter: 0 i: 76, pairs changed 29
    fullSet, iter: 0 i: 77, pairs changed 29
    fullSet, iter: 0 i: 78, pairs changed 29
    fullSet, iter: 0 i: 79, pairs changed 29
    j not moving enough
    fullSet, iter: 0 i: 80, pairs changed 29
    fullSet, iter: 0 i: 81, pairs changed 29
    L == H
    fullSet, iter: 0 i: 82, pairs changed 29
    fullSet, iter: 0 i: 83, pairs changed 29
    fullSet, iter: 0 i: 84, pairs changed 29
    fullSet, iter: 0 i: 85, pairs changed 29
    fullSet, iter: 0 i: 86, pairs changed 29
    fullSet, iter: 0 i: 87, pairs changed 29
    fullSet, iter: 0 i: 88, pairs changed 29
    fullSet, iter: 0 i: 89, pairs changed 29
    fullSet, iter: 0 i: 90, pairs changed 29
    fullSet, iter: 0 i: 91, pairs changed 29
    fullSet, iter: 0 i: 92, pairs changed 29
    fullSet, iter: 0 i: 93, pairs changed 29
    fullSet, iter: 0 i: 94, pairs changed 29
    fullSet, iter: 0 i: 95, pairs changed 29
    L == H
    fullSet, iter: 0 i: 96, pairs changed 29
    fullSet, iter: 0 i: 97, pairs changed 29
    fullSet, iter: 0 i: 98, pairs changed 29
    L == H
    fullSet, iter: 0 i: 99, pairs changed 29
    iteration number: 1
    j not moving enough
    non-bound, iter: 1 i: 0, pairs changed 0
    j not moving enough
    non-bound, iter: 1 i: 1, pairs changed 0
    non-bound, iter: 1 i: 2, pairs changed 1
    j not moving enough
    non-bound, iter: 1 i: 3, pairs changed 1
    j not moving enough
    non-bound, iter: 1 i: 6, pairs changed 1
    non-bound, iter: 1 i: 8, pairs changed 2
    j not moving enough
    non-bound, iter: 1 i: 10, pairs changed 2
    non-bound, iter: 1 i: 11, pairs changed 3
    j not moving enough
    non-bound, iter: 1 i: 14, pairs changed 3
    non-bound, iter: 1 i: 15, pairs changed 4
    j not moving enough
    non-bound, iter: 1 i: 16, pairs changed 4
    j not moving enough
    non-bound, iter: 1 i: 18, pairs changed 4
    j not moving enough
    non-bound, iter: 1 i: 19, pairs changed 4
    j not moving enough
    non-bound, iter: 1 i: 21, pairs changed 4
    j not moving enough
    non-bound, iter: 1 i: 27, pairs changed 4
    j not moving enough
    non-bound, iter: 1 i: 28, pairs changed 4
    non-bound, iter: 1 i: 29, pairs changed 5
    j not moving enough
    non-bound, iter: 1 i: 31, pairs changed 5
    j not moving enough
    non-bound, iter: 1 i: 34, pairs changed 5
    j not moving enough
    non-bound, iter: 1 i: 36, pairs changed 5
    j not moving enough
    non-bound, iter: 1 i: 41, pairs changed 5
    j not moving enough
    non-bound, iter: 1 i: 42, pairs changed 5
    j not moving enough
    non-bound, iter: 1 i: 45, pairs changed 5
    j not moving enough
    non-bound, iter: 1 i: 46, pairs changed 5
    j not moving enough
    non-bound, iter: 1 i: 56, pairs changed 5
    j not moving enough
    non-bound, iter: 1 i: 61, pairs changed 5
    j not moving enough
    non-bound, iter: 1 i: 62, pairs changed 5
    non-bound, iter: 1 i: 74, pairs changed 6
    iteration number: 2
    non-bound, iter: 2 i: 0, pairs changed 1
    j not moving enough
    non-bound, iter: 2 i: 2, pairs changed 1
    j not moving enough
    non-bound, iter: 2 i: 3, pairs changed 1
    j not moving enough
    non-bound, iter: 2 i: 6, pairs changed 1
    j not moving enough
    non-bound, iter: 2 i: 10, pairs changed 1
    j not moving enough
    non-bound, iter: 2 i: 14, pairs changed 1
    non-bound, iter: 2 i: 15, pairs changed 2
    j not moving enough
    non-bound, iter: 2 i: 16, pairs changed 2
    j not moving enough
    non-bound, iter: 2 i: 18, pairs changed 2
    j not moving enough
    non-bound, iter: 2 i: 19, pairs changed 2
    j not moving enough
    non-bound, iter: 2 i: 21, pairs changed 2
    j not moving enough
    non-bound, iter: 2 i: 27, pairs changed 2
    j not moving enough
    non-bound, iter: 2 i: 28, pairs changed 2
    non-bound, iter: 2 i: 29, pairs changed 3
    j not moving enough
    non-bound, iter: 2 i: 31, pairs changed 3
    j not moving enough
    non-bound, iter: 2 i: 34, pairs changed 3
    j not moving enough
    non-bound, iter: 2 i: 36, pairs changed 3
    j not moving enough
    non-bound, iter: 2 i: 41, pairs changed 3
    j not moving enough
    non-bound, iter: 2 i: 42, pairs changed 3
    j not moving enough
    non-bound, iter: 2 i: 45, pairs changed 3
    j not moving enough
    non-bound, iter: 2 i: 46, pairs changed 3
    j not moving enough
    non-bound, iter: 2 i: 56, pairs changed 3
    j not moving enough
    non-bound, iter: 2 i: 61, pairs changed 3
    j not moving enough
    non-bound, iter: 2 i: 62, pairs changed 3
    j not moving enough
    non-bound, iter: 2 i: 74, pairs changed 3
    iteration number: 3
    j not moving enough
    non-bound, iter: 3 i: 2, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 3, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 6, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 10, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 14, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 15, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 16, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 18, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 19, pairs changed 0
    non-bound, iter: 3 i: 21, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 27, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 28, pairs changed 0
    non-bound, iter: 3 i: 29, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 31, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 34, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 36, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 41, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 42, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 45, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 46, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 56, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 61, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 62, pairs changed 0
    j not moving enough
    non-bound, iter: 3 i: 74, pairs changed 0
    iteration number: 4
    j not moving enough
    fullSet, iter: 4 i: 0, pairs changed 0
    fullSet, iter: 4 i: 1, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 2, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 3, pairs changed 0
    fullSet, iter: 4 i: 4, pairs changed 0
    fullSet, iter: 4 i: 5, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 6, pairs changed 0
    fullSet, iter: 4 i: 7, pairs changed 0
    fullSet, iter: 4 i: 8, pairs changed 0
    fullSet, iter: 4 i: 9, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 10, pairs changed 0
    fullSet, iter: 4 i: 11, pairs changed 0
    fullSet, iter: 4 i: 12, pairs changed 0
    fullSet, iter: 4 i: 13, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 14, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 15, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 16, pairs changed 0
    L == H
    fullSet, iter: 4 i: 17, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 18, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 19, pairs changed 0
    fullSet, iter: 4 i: 20, pairs changed 0
    fullSet, iter: 4 i: 21, pairs changed 0
    fullSet, iter: 4 i: 22, pairs changed 0
    fullSet, iter: 4 i: 23, pairs changed 0
    fullSet, iter: 4 i: 24, pairs changed 0
    fullSet, iter: 4 i: 25, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 26, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 27, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 28, pairs changed 0
    fullSet, iter: 4 i: 29, pairs changed 0
    L == H
    fullSet, iter: 4 i: 30, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 31, pairs changed 0
    fullSet, iter: 4 i: 32, pairs changed 0
    fullSet, iter: 4 i: 33, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 34, pairs changed 0
    fullSet, iter: 4 i: 35, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 36, pairs changed 0
    fullSet, iter: 4 i: 37, pairs changed 0
    L == H
    fullSet, iter: 4 i: 38, pairs changed 0
    fullSet, iter: 4 i: 39, pairs changed 0
    fullSet, iter: 4 i: 40, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 41, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 42, pairs changed 0
    L == H
    fullSet, iter: 4 i: 43, pairs changed 0
    fullSet, iter: 4 i: 44, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 45, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 46, pairs changed 0
    fullSet, iter: 4 i: 47, pairs changed 0
    L == H
    fullSet, iter: 4 i: 48, pairs changed 0
    fullSet, iter: 4 i: 49, pairs changed 0
    fullSet, iter: 4 i: 50, pairs changed 0
    fullSet, iter: 4 i: 51, pairs changed 0
    fullSet, iter: 4 i: 52, pairs changed 0
    L == H
    fullSet, iter: 4 i: 53, pairs changed 0
    fullSet, iter: 4 i: 54, pairs changed 0
    fullSet, iter: 4 i: 55, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 56, pairs changed 0
    fullSet, iter: 4 i: 57, pairs changed 0
    L == H
    fullSet, iter: 4 i: 58, pairs changed 0
    fullSet, iter: 4 i: 59, pairs changed 0
    fullSet, iter: 4 i: 60, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 61, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 62, pairs changed 0
    fullSet, iter: 4 i: 63, pairs changed 0
    fullSet, iter: 4 i: 64, pairs changed 0
    fullSet, iter: 4 i: 65, pairs changed 0
    L == H
    fullSet, iter: 4 i: 66, pairs changed 0
    fullSet, iter: 4 i: 67, pairs changed 0
    fullSet, iter: 4 i: 68, pairs changed 0
    fullSet, iter: 4 i: 69, pairs changed 0
    fullSet, iter: 4 i: 70, pairs changed 0
    fullSet, iter: 4 i: 71, pairs changed 0
    fullSet, iter: 4 i: 72, pairs changed 0
    fullSet, iter: 4 i: 73, pairs changed 0
    j not moving enough
    fullSet, iter: 4 i: 74, pairs changed 0
    fullSet, iter: 4 i: 75, pairs changed 0
    L == H
    fullSet, iter: 4 i: 76, pairs changed 0
    fullSet, iter: 4 i: 77, pairs changed 0
    L == H
    fullSet, iter: 4 i: 78, pairs changed 0
    fullSet, iter: 4 i: 79, pairs changed 0
    fullSet, iter: 4 i: 80, pairs changed 0
    fullSet, iter: 4 i: 81, pairs changed 0
    L == H
    fullSet, iter: 4 i: 82, pairs changed 0
    fullSet, iter: 4 i: 83, pairs changed 0
    fullSet, iter: 4 i: 84, pairs changed 0
    L == H
    fullSet, iter: 4 i: 85, pairs changed 0
    fullSet, iter: 4 i: 86, pairs changed 0
    L == H
    fullSet, iter: 4 i: 87, pairs changed 0
    fullSet, iter: 4 i: 88, pairs changed 0
    fullSet, iter: 4 i: 89, pairs changed 0
    L == H
    fullSet, iter: 4 i: 90, pairs changed 0
    L == H
    fullSet, iter: 4 i: 91, pairs changed 0
    L == H
    fullSet, iter: 4 i: 92, pairs changed 0
    fullSet, iter: 4 i: 93, pairs changed 0
    fullSet, iter: 4 i: 94, pairs changed 0
    fullSet, iter: 4 i: 95, pairs changed 0
    fullSet, iter: 4 i: 96, pairs changed 0
    fullSet, iter: 4 i: 97, pairs changed 0
    fullSet, iter: 4 i: 98, pairs changed 0
    fullSet, iter: 4 i: 99, pairs changed 0
    iteration number: 5
    there are 24 Support Vectors
    the training error rate is: 0.070000
    the test error rate is: 0.050000
    

支持向量的数目存在一个最优值。SVM的优点在于它能对数据进行高效分类。

如果支持向量太少，就可能会得到一个很差的决策边界；如果支持向量太多，也就相当于每次都利用整个数据集进行分类，这种分类方法称为k近邻。

# 6. 示例：手写识别问题回顾

示例：基于SVM的手写识别

（1）收集数据：提供的文本文件

（2）准备数据：基于二值图像构造向量

（3）分析数据：对图像向量进行目测

（4）训练算法：采用两种不同的核函数，并对径向基核函数采用不同的设置来运行SMO算法

（5）测试算法：编写一个函数来测试不同的核函数并计算错误率

（6）使用算法：一个图像识别的完整应用还需要一些图像处理的知识


```python
from os import listdir

def img2vector(filename):
    returnVect = zeros((1, 1024))
    with open(filename) as fp:
        for i in range(32):
            lineStr = fp.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
        return returnVect

def loadImages(dirName):
    hwLabels = []
    trainingFileList = listdir(dirName)
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels

def testDigits(kTup = ('rbf', 10)):
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat = mat(dataArr)
    labelMat = mat(labelArr).T
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print('there are %d Support Vectors' % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print('the training error rate is %f' % (float(errorCount) / m))
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).T
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]):
            errorCount += 1
    print('the test error rate is %f' % (float(errorCount) / m))
    
testDigits(('rbf', 20))
```

    fullSet, iter: 0 i: 0, pairs changed 1
    fullSet, iter: 0 i: 1, pairs changed 2
    fullSet, iter: 0 i: 2, pairs changed 3
    fullSet, iter: 0 i: 3, pairs changed 4
    fullSet, iter: 0 i: 4, pairs changed 5
    fullSet, iter: 0 i: 5, pairs changed 6
    fullSet, iter: 0 i: 6, pairs changed 6
    fullSet, iter: 0 i: 7, pairs changed 7
    fullSet, iter: 0 i: 8, pairs changed 7
    fullSet, iter: 0 i: 9, pairs changed 7
    fullSet, iter: 0 i: 10, pairs changed 7
    fullSet, iter: 0 i: 11, pairs changed 7
    fullSet, iter: 0 i: 12, pairs changed 8
    fullSet, iter: 0 i: 13, pairs changed 9
    fullSet, iter: 0 i: 14, pairs changed 10
    j not moving enough
    fullSet, iter: 0 i: 15, pairs changed 10
    fullSet, iter: 0 i: 16, pairs changed 10
    fullSet, iter: 0 i: 17, pairs changed 10
    fullSet, iter: 0 i: 18, pairs changed 10
    fullSet, iter: 0 i: 19, pairs changed 10
    fullSet, iter: 0 i: 20, pairs changed 10
    j not moving enough
    fullSet, iter: 0 i: 21, pairs changed 10
    fullSet, iter: 0 i: 22, pairs changed 10
    fullSet, iter: 0 i: 23, pairs changed 10
    fullSet, iter: 0 i: 24, pairs changed 11
    fullSet, iter: 0 i: 25, pairs changed 12
    fullSet, iter: 0 i: 26, pairs changed 12
    fullSet, iter: 0 i: 27, pairs changed 12
    fullSet, iter: 0 i: 28, pairs changed 12
    fullSet, iter: 0 i: 29, pairs changed 12
    j not moving enough
    fullSet, iter: 0 i: 30, pairs changed 12
    fullSet, iter: 0 i: 31, pairs changed 12
    fullSet, iter: 0 i: 32, pairs changed 12
    fullSet, iter: 0 i: 33, pairs changed 12
    fullSet, iter: 0 i: 34, pairs changed 12
    fullSet, iter: 0 i: 35, pairs changed 12
    fullSet, iter: 0 i: 36, pairs changed 12
    j not moving enough
    fullSet, iter: 0 i: 37, pairs changed 12
    fullSet, iter: 0 i: 38, pairs changed 12
    fullSet, iter: 0 i: 39, pairs changed 12
    fullSet, iter: 0 i: 40, pairs changed 12
    fullSet, iter: 0 i: 41, pairs changed 12
    fullSet, iter: 0 i: 42, pairs changed 13
    fullSet, iter: 0 i: 43, pairs changed 13
    fullSet, iter: 0 i: 44, pairs changed 14
    fullSet, iter: 0 i: 45, pairs changed 14
    fullSet, iter: 0 i: 46, pairs changed 14
    fullSet, iter: 0 i: 47, pairs changed 15
    fullSet, iter: 0 i: 48, pairs changed 15
    fullSet, iter: 0 i: 49, pairs changed 16
    fullSet, iter: 0 i: 50, pairs changed 16
    fullSet, iter: 0 i: 51, pairs changed 16
    fullSet, iter: 0 i: 52, pairs changed 16
    fullSet, iter: 0 i: 53, pairs changed 16
    fullSet, iter: 0 i: 54, pairs changed 16
    fullSet, iter: 0 i: 55, pairs changed 16
    fullSet, iter: 0 i: 56, pairs changed 16
    fullSet, iter: 0 i: 57, pairs changed 16
    fullSet, iter: 0 i: 58, pairs changed 16
    fullSet, iter: 0 i: 59, pairs changed 16
    fullSet, iter: 0 i: 60, pairs changed 16
    fullSet, iter: 0 i: 61, pairs changed 16
    fullSet, iter: 0 i: 62, pairs changed 16
    fullSet, iter: 0 i: 63, pairs changed 17
    fullSet, iter: 0 i: 64, pairs changed 17
    fullSet, iter: 0 i: 65, pairs changed 18
    fullSet, iter: 0 i: 66, pairs changed 19
    fullSet, iter: 0 i: 67, pairs changed 19
    fullSet, iter: 0 i: 68, pairs changed 19
    fullSet, iter: 0 i: 69, pairs changed 19
    fullSet, iter: 0 i: 70, pairs changed 19
    fullSet, iter: 0 i: 71, pairs changed 19
    fullSet, iter: 0 i: 72, pairs changed 19
    fullSet, iter: 0 i: 73, pairs changed 19
    fullSet, iter: 0 i: 74, pairs changed 19
    fullSet, iter: 0 i: 75, pairs changed 19
    fullSet, iter: 0 i: 76, pairs changed 19
    fullSet, iter: 0 i: 77, pairs changed 20
    fullSet, iter: 0 i: 78, pairs changed 20
    fullSet, iter: 0 i: 79, pairs changed 20
    fullSet, iter: 0 i: 80, pairs changed 20
    fullSet, iter: 0 i: 81, pairs changed 21
    fullSet, iter: 0 i: 82, pairs changed 21
    fullSet, iter: 0 i: 83, pairs changed 21
    fullSet, iter: 0 i: 84, pairs changed 21
    L == H
    fullSet, iter: 0 i: 85, pairs changed 21
    fullSet, iter: 0 i: 86, pairs changed 21
    fullSet, iter: 0 i: 87, pairs changed 21
    fullSet, iter: 0 i: 88, pairs changed 21
    fullSet, iter: 0 i: 89, pairs changed 21
    fullSet, iter: 0 i: 90, pairs changed 21
    fullSet, iter: 0 i: 91, pairs changed 21
    fullSet, iter: 0 i: 92, pairs changed 21
    fullSet, iter: 0 i: 93, pairs changed 21
    fullSet, iter: 0 i: 94, pairs changed 21
    fullSet, iter: 0 i: 95, pairs changed 21
    fullSet, iter: 0 i: 96, pairs changed 21
    fullSet, iter: 0 i: 97, pairs changed 21
    fullSet, iter: 0 i: 98, pairs changed 21
    fullSet, iter: 0 i: 99, pairs changed 21
    fullSet, iter: 0 i: 100, pairs changed 21
    fullSet, iter: 0 i: 101, pairs changed 21
    fullSet, iter: 0 i: 102, pairs changed 21
    fullSet, iter: 0 i: 103, pairs changed 21
    fullSet, iter: 0 i: 104, pairs changed 21
    L == H
    fullSet, iter: 0 i: 105, pairs changed 21
    fullSet, iter: 0 i: 106, pairs changed 21
    fullSet, iter: 0 i: 107, pairs changed 21
    fullSet, iter: 0 i: 108, pairs changed 21
    fullSet, iter: 0 i: 109, pairs changed 21
    fullSet, iter: 0 i: 110, pairs changed 21
    fullSet, iter: 0 i: 111, pairs changed 21
    fullSet, iter: 0 i: 112, pairs changed 21
    fullSet, iter: 0 i: 113, pairs changed 21
    fullSet, iter: 0 i: 114, pairs changed 21
    fullSet, iter: 0 i: 115, pairs changed 21
    fullSet, iter: 0 i: 116, pairs changed 21
    fullSet, iter: 0 i: 117, pairs changed 21
    fullSet, iter: 0 i: 118, pairs changed 21
    fullSet, iter: 0 i: 119, pairs changed 21
    fullSet, iter: 0 i: 120, pairs changed 21
    fullSet, iter: 0 i: 121, pairs changed 21
    fullSet, iter: 0 i: 122, pairs changed 21
    fullSet, iter: 0 i: 123, pairs changed 21
    fullSet, iter: 0 i: 124, pairs changed 21
    fullSet, iter: 0 i: 125, pairs changed 21
    fullSet, iter: 0 i: 126, pairs changed 21
    L == H
    fullSet, iter: 0 i: 127, pairs changed 21
    fullSet, iter: 0 i: 128, pairs changed 21
    fullSet, iter: 0 i: 129, pairs changed 21
    fullSet, iter: 0 i: 130, pairs changed 21
    fullSet, iter: 0 i: 131, pairs changed 21
    fullSet, iter: 0 i: 132, pairs changed 21
    fullSet, iter: 0 i: 133, pairs changed 21
    fullSet, iter: 0 i: 134, pairs changed 21
    fullSet, iter: 0 i: 135, pairs changed 21
    fullSet, iter: 0 i: 136, pairs changed 21
    fullSet, iter: 0 i: 137, pairs changed 21
    fullSet, iter: 0 i: 138, pairs changed 21
    fullSet, iter: 0 i: 139, pairs changed 21
    fullSet, iter: 0 i: 140, pairs changed 21
    fullSet, iter: 0 i: 141, pairs changed 21
    fullSet, iter: 0 i: 142, pairs changed 21
    fullSet, iter: 0 i: 143, pairs changed 21
    fullSet, iter: 0 i: 144, pairs changed 21
    fullSet, iter: 0 i: 145, pairs changed 21
    fullSet, iter: 0 i: 146, pairs changed 21
    fullSet, iter: 0 i: 147, pairs changed 21
    fullSet, iter: 0 i: 148, pairs changed 21
    fullSet, iter: 0 i: 149, pairs changed 21
    fullSet, iter: 0 i: 150, pairs changed 21
    fullSet, iter: 0 i: 151, pairs changed 21
    fullSet, iter: 0 i: 152, pairs changed 21
    fullSet, iter: 0 i: 153, pairs changed 21
    fullSet, iter: 0 i: 154, pairs changed 21
    fullSet, iter: 0 i: 155, pairs changed 21
    fullSet, iter: 0 i: 156, pairs changed 21
    fullSet, iter: 0 i: 157, pairs changed 21
    fullSet, iter: 0 i: 158, pairs changed 21
    fullSet, iter: 0 i: 159, pairs changed 21
    fullSet, iter: 0 i: 160, pairs changed 21
    fullSet, iter: 0 i: 161, pairs changed 21
    fullSet, iter: 0 i: 162, pairs changed 21
    fullSet, iter: 0 i: 163, pairs changed 21
    fullSet, iter: 0 i: 164, pairs changed 21
    fullSet, iter: 0 i: 165, pairs changed 21
    fullSet, iter: 0 i: 166, pairs changed 21
    fullSet, iter: 0 i: 167, pairs changed 21
    fullSet, iter: 0 i: 168, pairs changed 21
    fullSet, iter: 0 i: 169, pairs changed 21
    fullSet, iter: 0 i: 170, pairs changed 21
    fullSet, iter: 0 i: 171, pairs changed 21
    fullSet, iter: 0 i: 172, pairs changed 21
    fullSet, iter: 0 i: 173, pairs changed 21
    fullSet, iter: 0 i: 174, pairs changed 21
    fullSet, iter: 0 i: 175, pairs changed 21
    fullSet, iter: 0 i: 176, pairs changed 21
    fullSet, iter: 0 i: 177, pairs changed 21
    fullSet, iter: 0 i: 178, pairs changed 21
    fullSet, iter: 0 i: 179, pairs changed 21
    fullSet, iter: 0 i: 180, pairs changed 21
    fullSet, iter: 0 i: 181, pairs changed 21
    fullSet, iter: 0 i: 182, pairs changed 21
    fullSet, iter: 0 i: 183, pairs changed 21
    fullSet, iter: 0 i: 184, pairs changed 21
    fullSet, iter: 0 i: 185, pairs changed 21
    fullSet, iter: 0 i: 186, pairs changed 21
    fullSet, iter: 0 i: 187, pairs changed 21
    fullSet, iter: 0 i: 188, pairs changed 21
    fullSet, iter: 0 i: 189, pairs changed 21
    fullSet, iter: 0 i: 190, pairs changed 21
    fullSet, iter: 0 i: 191, pairs changed 21
    fullSet, iter: 0 i: 192, pairs changed 21
    fullSet, iter: 0 i: 193, pairs changed 21
    fullSet, iter: 0 i: 194, pairs changed 21
    fullSet, iter: 0 i: 195, pairs changed 21
    fullSet, iter: 0 i: 196, pairs changed 21
    fullSet, iter: 0 i: 197, pairs changed 21
    fullSet, iter: 0 i: 198, pairs changed 22
    fullSet, iter: 0 i: 199, pairs changed 23
    fullSet, iter: 0 i: 200, pairs changed 24
    fullSet, iter: 0 i: 201, pairs changed 25
    fullSet, iter: 0 i: 202, pairs changed 26
    fullSet, iter: 0 i: 203, pairs changed 27
    fullSet, iter: 0 i: 204, pairs changed 28
    fullSet, iter: 0 i: 205, pairs changed 29
    fullSet, iter: 0 i: 206, pairs changed 29
    fullSet, iter: 0 i: 207, pairs changed 30
    fullSet, iter: 0 i: 208, pairs changed 31
    fullSet, iter: 0 i: 209, pairs changed 32
    fullSet, iter: 0 i: 210, pairs changed 33
    fullSet, iter: 0 i: 211, pairs changed 34
    fullSet, iter: 0 i: 212, pairs changed 34
    fullSet, iter: 0 i: 213, pairs changed 34
    fullSet, iter: 0 i: 214, pairs changed 34
    fullSet, iter: 0 i: 215, pairs changed 35
    fullSet, iter: 0 i: 216, pairs changed 36
    fullSet, iter: 0 i: 217, pairs changed 36
    j not moving enough
    fullSet, iter: 0 i: 218, pairs changed 36
    fullSet, iter: 0 i: 219, pairs changed 36
    fullSet, iter: 0 i: 220, pairs changed 36
    fullSet, iter: 0 i: 221, pairs changed 36
    fullSet, iter: 0 i: 222, pairs changed 37
    fullSet, iter: 0 i: 223, pairs changed 37
    fullSet, iter: 0 i: 224, pairs changed 37
    fullSet, iter: 0 i: 225, pairs changed 38
    fullSet, iter: 0 i: 226, pairs changed 38
    fullSet, iter: 0 i: 227, pairs changed 38
    fullSet, iter: 0 i: 228, pairs changed 38
    fullSet, iter: 0 i: 229, pairs changed 38
    fullSet, iter: 0 i: 230, pairs changed 38
    fullSet, iter: 0 i: 231, pairs changed 38
    j not moving enough
    fullSet, iter: 0 i: 232, pairs changed 38
    fullSet, iter: 0 i: 233, pairs changed 39
    fullSet, iter: 0 i: 234, pairs changed 40
    fullSet, iter: 0 i: 235, pairs changed 41
    fullSet, iter: 0 i: 236, pairs changed 42
    j not moving enough
    fullSet, iter: 0 i: 237, pairs changed 42
    fullSet, iter: 0 i: 238, pairs changed 43
    fullSet, iter: 0 i: 239, pairs changed 43
    fullSet, iter: 0 i: 240, pairs changed 43
    fullSet, iter: 0 i: 241, pairs changed 43
    fullSet, iter: 0 i: 242, pairs changed 44
    fullSet, iter: 0 i: 243, pairs changed 44
    fullSet, iter: 0 i: 244, pairs changed 44
    fullSet, iter: 0 i: 245, pairs changed 45
    j not moving enough
    fullSet, iter: 0 i: 246, pairs changed 45
    fullSet, iter: 0 i: 247, pairs changed 45
    j not moving enough
    fullSet, iter: 0 i: 248, pairs changed 45
    fullSet, iter: 0 i: 249, pairs changed 46
    fullSet, iter: 0 i: 250, pairs changed 46
    fullSet, iter: 0 i: 251, pairs changed 46
    fullSet, iter: 0 i: 252, pairs changed 46
    fullSet, iter: 0 i: 253, pairs changed 46
    fullSet, iter: 0 i: 254, pairs changed 46
    fullSet, iter: 0 i: 255, pairs changed 46
    fullSet, iter: 0 i: 256, pairs changed 46
    fullSet, iter: 0 i: 257, pairs changed 46
    fullSet, iter: 0 i: 258, pairs changed 46
    fullSet, iter: 0 i: 259, pairs changed 46
    fullSet, iter: 0 i: 260, pairs changed 46
    fullSet, iter: 0 i: 261, pairs changed 46
    fullSet, iter: 0 i: 262, pairs changed 46
    fullSet, iter: 0 i: 263, pairs changed 46
    fullSet, iter: 0 i: 264, pairs changed 46
    fullSet, iter: 0 i: 265, pairs changed 46
    fullSet, iter: 0 i: 266, pairs changed 46
    fullSet, iter: 0 i: 267, pairs changed 46
    fullSet, iter: 0 i: 268, pairs changed 46
    fullSet, iter: 0 i: 269, pairs changed 46
    fullSet, iter: 0 i: 270, pairs changed 46
    fullSet, iter: 0 i: 271, pairs changed 46
    fullSet, iter: 0 i: 272, pairs changed 46
    fullSet, iter: 0 i: 273, pairs changed 46
    fullSet, iter: 0 i: 274, pairs changed 46
    fullSet, iter: 0 i: 275, pairs changed 46
    fullSet, iter: 0 i: 276, pairs changed 46
    fullSet, iter: 0 i: 277, pairs changed 46
    fullSet, iter: 0 i: 278, pairs changed 46
    fullSet, iter: 0 i: 279, pairs changed 46
    fullSet, iter: 0 i: 280, pairs changed 46
    fullSet, iter: 0 i: 281, pairs changed 46
    fullSet, iter: 0 i: 282, pairs changed 46
    fullSet, iter: 0 i: 283, pairs changed 46
    fullSet, iter: 0 i: 284, pairs changed 47
    fullSet, iter: 0 i: 285, pairs changed 47
    fullSet, iter: 0 i: 286, pairs changed 47
    L == H
    fullSet, iter: 0 i: 287, pairs changed 47
    fullSet, iter: 0 i: 288, pairs changed 47
    L == H
    fullSet, iter: 0 i: 289, pairs changed 47
    L == H
    fullSet, iter: 0 i: 290, pairs changed 47
    L == H
    fullSet, iter: 0 i: 291, pairs changed 47
    L == H
    fullSet, iter: 0 i: 292, pairs changed 47
    L == H
    fullSet, iter: 0 i: 293, pairs changed 47
    fullSet, iter: 0 i: 294, pairs changed 47
    fullSet, iter: 0 i: 295, pairs changed 47
    L == H
    fullSet, iter: 0 i: 296, pairs changed 47
    fullSet, iter: 0 i: 297, pairs changed 47
    L == H
    fullSet, iter: 0 i: 298, pairs changed 47
    fullSet, iter: 0 i: 299, pairs changed 47
    fullSet, iter: 0 i: 300, pairs changed 47
    L == H
    fullSet, iter: 0 i: 301, pairs changed 47
    L == H
    fullSet, iter: 0 i: 302, pairs changed 47
    L == H
    fullSet, iter: 0 i: 303, pairs changed 47
    L == H
    fullSet, iter: 0 i: 304, pairs changed 47
    L == H
    fullSet, iter: 0 i: 305, pairs changed 47
    L == H
    fullSet, iter: 0 i: 306, pairs changed 47
    L == H
    fullSet, iter: 0 i: 307, pairs changed 47
    fullSet, iter: 0 i: 308, pairs changed 47
    fullSet, iter: 0 i: 309, pairs changed 47
    L == H
    fullSet, iter: 0 i: 310, pairs changed 47
    L == H
    fullSet, iter: 0 i: 311, pairs changed 47
    fullSet, iter: 0 i: 312, pairs changed 47
    fullSet, iter: 0 i: 313, pairs changed 47
    fullSet, iter: 0 i: 314, pairs changed 47
    fullSet, iter: 0 i: 315, pairs changed 47
    L == H
    fullSet, iter: 0 i: 316, pairs changed 47
    L == H
    fullSet, iter: 0 i: 317, pairs changed 47
    L == H
    fullSet, iter: 0 i: 318, pairs changed 47
    L == H
    fullSet, iter: 0 i: 319, pairs changed 47
    fullSet, iter: 0 i: 320, pairs changed 48
    L == H
    fullSet, iter: 0 i: 321, pairs changed 48
    L == H
    fullSet, iter: 0 i: 322, pairs changed 48
    L == H
    fullSet, iter: 0 i: 323, pairs changed 48
    L == H
    fullSet, iter: 0 i: 324, pairs changed 48
    L == H
    fullSet, iter: 0 i: 325, pairs changed 48
    L == H
    fullSet, iter: 0 i: 326, pairs changed 48
    L == H
    fullSet, iter: 0 i: 327, pairs changed 48
    L == H
    fullSet, iter: 0 i: 328, pairs changed 48
    L == H
    fullSet, iter: 0 i: 329, pairs changed 48
    L == H
    fullSet, iter: 0 i: 330, pairs changed 48
    L == H
    fullSet, iter: 0 i: 331, pairs changed 48
    L == H
    fullSet, iter: 0 i: 332, pairs changed 48
    L == H
    fullSet, iter: 0 i: 333, pairs changed 48
    L == H
    fullSet, iter: 0 i: 334, pairs changed 48
    L == H
    fullSet, iter: 0 i: 335, pairs changed 48
    L == H
    fullSet, iter: 0 i: 336, pairs changed 48
    L == H
    fullSet, iter: 0 i: 337, pairs changed 48
    L == H
    fullSet, iter: 0 i: 338, pairs changed 48
    L == H
    fullSet, iter: 0 i: 339, pairs changed 48
    L == H
    fullSet, iter: 0 i: 340, pairs changed 48
    L == H
    fullSet, iter: 0 i: 341, pairs changed 48
    L == H
    fullSet, iter: 0 i: 342, pairs changed 48
    L == H
    fullSet, iter: 0 i: 343, pairs changed 48
    L == H
    fullSet, iter: 0 i: 344, pairs changed 48
    L == H
    fullSet, iter: 0 i: 345, pairs changed 48
    L == H
    fullSet, iter: 0 i: 346, pairs changed 48
    fullSet, iter: 0 i: 347, pairs changed 48
    L == H
    fullSet, iter: 0 i: 348, pairs changed 48
    L == H
    fullSet, iter: 0 i: 349, pairs changed 48
    L == H
    fullSet, iter: 0 i: 350, pairs changed 48
    L == H
    fullSet, iter: 0 i: 351, pairs changed 48
    L == H
    fullSet, iter: 0 i: 352, pairs changed 48
    L == H
    fullSet, iter: 0 i: 353, pairs changed 48
    L == H
    fullSet, iter: 0 i: 354, pairs changed 48
    fullSet, iter: 0 i: 355, pairs changed 48
    L == H
    fullSet, iter: 0 i: 356, pairs changed 48
    L == H
    fullSet, iter: 0 i: 357, pairs changed 48
    L == H
    fullSet, iter: 0 i: 358, pairs changed 48
    L == H
    fullSet, iter: 0 i: 359, pairs changed 48
    fullSet, iter: 0 i: 360, pairs changed 48
    L == H
    fullSet, iter: 0 i: 361, pairs changed 48
    L == H
    fullSet, iter: 0 i: 362, pairs changed 48
    L == H
    fullSet, iter: 0 i: 363, pairs changed 48
    L == H
    fullSet, iter: 0 i: 364, pairs changed 48
    L == H
    fullSet, iter: 0 i: 365, pairs changed 48
    L == H
    fullSet, iter: 0 i: 366, pairs changed 48
    L == H
    fullSet, iter: 0 i: 367, pairs changed 48
    L == H
    fullSet, iter: 0 i: 368, pairs changed 48
    L == H
    fullSet, iter: 0 i: 369, pairs changed 48
    L == H
    fullSet, iter: 0 i: 370, pairs changed 48
    L == H
    fullSet, iter: 0 i: 371, pairs changed 48
    L == H
    fullSet, iter: 0 i: 372, pairs changed 48
    L == H
    fullSet, iter: 0 i: 373, pairs changed 48
    L == H
    fullSet, iter: 0 i: 374, pairs changed 48
    L == H
    fullSet, iter: 0 i: 375, pairs changed 48
    L == H
    fullSet, iter: 0 i: 376, pairs changed 48
    L == H
    fullSet, iter: 0 i: 377, pairs changed 48
    L == H
    fullSet, iter: 0 i: 378, pairs changed 48
    L == H
    fullSet, iter: 0 i: 379, pairs changed 48
    L == H
    fullSet, iter: 0 i: 380, pairs changed 48
    L == H
    fullSet, iter: 0 i: 381, pairs changed 48
    L == H
    fullSet, iter: 0 i: 382, pairs changed 48
    L == H
    fullSet, iter: 0 i: 383, pairs changed 48
    L == H
    fullSet, iter: 0 i: 384, pairs changed 48
    L == H
    fullSet, iter: 0 i: 385, pairs changed 48
    L == H
    fullSet, iter: 0 i: 386, pairs changed 48
    L == H
    fullSet, iter: 0 i: 387, pairs changed 48
    L == H
    fullSet, iter: 0 i: 388, pairs changed 48
    L == H
    fullSet, iter: 0 i: 389, pairs changed 48
    L == H
    fullSet, iter: 0 i: 390, pairs changed 48
    L == H
    fullSet, iter: 0 i: 391, pairs changed 48
    L == H
    fullSet, iter: 0 i: 392, pairs changed 48
    L == H
    fullSet, iter: 0 i: 393, pairs changed 48
    L == H
    fullSet, iter: 0 i: 394, pairs changed 48
    L == H
    fullSet, iter: 0 i: 395, pairs changed 48
    L == H
    fullSet, iter: 0 i: 396, pairs changed 48
    L == H
    fullSet, iter: 0 i: 397, pairs changed 48
    L == H
    fullSet, iter: 0 i: 398, pairs changed 48
    L == H
    fullSet, iter: 0 i: 399, pairs changed 48
    L == H
    fullSet, iter: 0 i: 400, pairs changed 48
    L == H
    fullSet, iter: 0 i: 401, pairs changed 48
    iteration number: 1
    non-bound, iter: 1 i: 1, pairs changed 1
    non-bound, iter: 1 i: 3, pairs changed 2
    j not moving enough
    non-bound, iter: 1 i: 5, pairs changed 2
    non-bound, iter: 1 i: 7, pairs changed 3
    non-bound, iter: 1 i: 13, pairs changed 4
    non-bound, iter: 1 i: 14, pairs changed 5
    non-bound, iter: 1 i: 21, pairs changed 6
    j not moving enough
    non-bound, iter: 1 i: 24, pairs changed 6
    j not moving enough
    non-bound, iter: 1 i: 25, pairs changed 6
    j not moving enough
    non-bound, iter: 1 i: 42, pairs changed 6
    non-bound, iter: 1 i: 47, pairs changed 7
    non-bound, iter: 1 i: 49, pairs changed 8
    j not moving enough
    non-bound, iter: 1 i: 63, pairs changed 8
    non-bound, iter: 1 i: 65, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 66, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 77, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 81, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 85, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 105, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 127, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 198, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 199, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 200, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 201, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 203, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 204, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 205, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 207, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 208, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 210, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 211, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 215, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 216, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 225, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 233, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 234, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 235, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 236, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 238, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 242, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 245, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 249, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 284, pairs changed 9
    j not moving enough
    non-bound, iter: 1 i: 317, pairs changed 9
    iteration number: 2
    non-bound, iter: 2 i: 1, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 3, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 5, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 7, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 13, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 14, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 21, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 24, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 25, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 42, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 47, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 49, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 63, pairs changed 0
    non-bound, iter: 2 i: 65, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 66, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 77, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 81, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 85, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 105, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 127, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 198, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 199, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 200, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 201, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 203, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 204, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 205, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 207, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 208, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 210, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 211, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 215, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 216, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 225, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 233, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 234, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 235, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 236, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 238, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 242, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 245, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 249, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 284, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 305, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 311, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 317, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 330, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 332, pairs changed 0
    j not moving enough
    non-bound, iter: 2 i: 337, pairs changed 0
    iteration number: 3
    fullSet, iter: 3 i: 0, pairs changed 0
    fullSet, iter: 3 i: 1, pairs changed 0
    fullSet, iter: 3 i: 2, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 3, pairs changed 0
    fullSet, iter: 3 i: 4, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 5, pairs changed 0
    fullSet, iter: 3 i: 6, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 7, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 8, pairs changed 0
    fullSet, iter: 3 i: 9, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 10, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 11, pairs changed 0
    L == H
    fullSet, iter: 3 i: 12, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 13, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 14, pairs changed 0
    L == H
    fullSet, iter: 3 i: 15, pairs changed 0
    L == H
    fullSet, iter: 3 i: 16, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 17, pairs changed 0
    L == H
    fullSet, iter: 3 i: 18, pairs changed 0
    fullSet, iter: 3 i: 19, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 20, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 21, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 22, pairs changed 0
    L == H
    fullSet, iter: 3 i: 23, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 24, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 25, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 26, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 27, pairs changed 0
    fullSet, iter: 3 i: 28, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 29, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 30, pairs changed 0
    fullSet, iter: 3 i: 31, pairs changed 0
    fullSet, iter: 3 i: 32, pairs changed 0
    fullSet, iter: 3 i: 33, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 34, pairs changed 0
    fullSet, iter: 3 i: 35, pairs changed 0
    fullSet, iter: 3 i: 36, pairs changed 0
    fullSet, iter: 3 i: 37, pairs changed 0
    fullSet, iter: 3 i: 38, pairs changed 0
    L == H
    fullSet, iter: 3 i: 39, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 40, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 41, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 42, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 43, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 44, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 45, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 46, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 47, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 48, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 49, pairs changed 0
    fullSet, iter: 3 i: 50, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 51, pairs changed 0
    fullSet, iter: 3 i: 52, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 53, pairs changed 0
    L == H
    fullSet, iter: 3 i: 54, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 55, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 56, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 57, pairs changed 0
    fullSet, iter: 3 i: 58, pairs changed 0
    fullSet, iter: 3 i: 59, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 60, pairs changed 0
    fullSet, iter: 3 i: 61, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 62, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 63, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 64, pairs changed 0
    fullSet, iter: 3 i: 65, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 66, pairs changed 0
    fullSet, iter: 3 i: 67, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 68, pairs changed 0
    fullSet, iter: 3 i: 69, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 70, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 71, pairs changed 0
    L == H
    fullSet, iter: 3 i: 72, pairs changed 0
    fullSet, iter: 3 i: 73, pairs changed 0
    fullSet, iter: 3 i: 74, pairs changed 0
    L == H
    fullSet, iter: 3 i: 75, pairs changed 0
    fullSet, iter: 3 i: 76, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 77, pairs changed 0
    fullSet, iter: 3 i: 78, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 79, pairs changed 0
    fullSet, iter: 3 i: 80, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 81, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 82, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 83, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 84, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 85, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 86, pairs changed 0
    L == H
    fullSet, iter: 3 i: 87, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 88, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 89, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 90, pairs changed 0
    fullSet, iter: 3 i: 91, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 92, pairs changed 0
    fullSet, iter: 3 i: 93, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 94, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 95, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 96, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 97, pairs changed 0
    L == H
    fullSet, iter: 3 i: 98, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 99, pairs changed 0
    fullSet, iter: 3 i: 100, pairs changed 0
    L == H
    fullSet, iter: 3 i: 101, pairs changed 0
    fullSet, iter: 3 i: 102, pairs changed 0
    fullSet, iter: 3 i: 103, pairs changed 0
    fullSet, iter: 3 i: 104, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 105, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 106, pairs changed 0
    fullSet, iter: 3 i: 107, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 108, pairs changed 0
    fullSet, iter: 3 i: 109, pairs changed 0
    fullSet, iter: 3 i: 110, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 111, pairs changed 0
    fullSet, iter: 3 i: 112, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 113, pairs changed 0
    fullSet, iter: 3 i: 114, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 115, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 116, pairs changed 0
    L == H
    fullSet, iter: 3 i: 117, pairs changed 0
    fullSet, iter: 3 i: 118, pairs changed 0
    L == H
    fullSet, iter: 3 i: 119, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 120, pairs changed 0
    fullSet, iter: 3 i: 121, pairs changed 0
    fullSet, iter: 3 i: 122, pairs changed 0
    L == H
    fullSet, iter: 3 i: 123, pairs changed 0
    L == H
    fullSet, iter: 3 i: 124, pairs changed 0
    L == H
    fullSet, iter: 3 i: 125, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 126, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 127, pairs changed 0
    L == H
    fullSet, iter: 3 i: 128, pairs changed 0
    L == H
    fullSet, iter: 3 i: 129, pairs changed 0
    L == H
    fullSet, iter: 3 i: 130, pairs changed 0
    L == H
    fullSet, iter: 3 i: 131, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 132, pairs changed 0
    fullSet, iter: 3 i: 133, pairs changed 0
    fullSet, iter: 3 i: 134, pairs changed 0
    fullSet, iter: 3 i: 135, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 136, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 137, pairs changed 0
    fullSet, iter: 3 i: 138, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 139, pairs changed 0
    fullSet, iter: 3 i: 140, pairs changed 0
    fullSet, iter: 3 i: 141, pairs changed 0
    fullSet, iter: 3 i: 142, pairs changed 0
    fullSet, iter: 3 i: 143, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 144, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 145, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 146, pairs changed 0
    fullSet, iter: 3 i: 147, pairs changed 0
    fullSet, iter: 3 i: 148, pairs changed 0
    fullSet, iter: 3 i: 149, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 150, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 151, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 152, pairs changed 0
    fullSet, iter: 3 i: 153, pairs changed 0
    fullSet, iter: 3 i: 154, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 155, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 156, pairs changed 0
    L == H
    fullSet, iter: 3 i: 157, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 158, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 159, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 160, pairs changed 0
    L == H
    fullSet, iter: 3 i: 161, pairs changed 0
    L == H
    fullSet, iter: 3 i: 162, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 163, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 164, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 165, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 166, pairs changed 0
    fullSet, iter: 3 i: 167, pairs changed 0
    fullSet, iter: 3 i: 168, pairs changed 0
    fullSet, iter: 3 i: 169, pairs changed 0
    fullSet, iter: 3 i: 170, pairs changed 0
    fullSet, iter: 3 i: 171, pairs changed 0
    L == H
    fullSet, iter: 3 i: 172, pairs changed 0
    L == H
    fullSet, iter: 3 i: 173, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 174, pairs changed 0
    fullSet, iter: 3 i: 175, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 176, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 177, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 178, pairs changed 0
    L == H
    fullSet, iter: 3 i: 179, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 180, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 181, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 182, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 183, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 184, pairs changed 0
    L == H
    fullSet, iter: 3 i: 185, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 186, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 187, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 188, pairs changed 0
    L == H
    fullSet, iter: 3 i: 189, pairs changed 0
    fullSet, iter: 3 i: 190, pairs changed 0
    fullSet, iter: 3 i: 191, pairs changed 0
    fullSet, iter: 3 i: 192, pairs changed 0
    fullSet, iter: 3 i: 193, pairs changed 0
    fullSet, iter: 3 i: 194, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 195, pairs changed 0
    fullSet, iter: 3 i: 196, pairs changed 0
    fullSet, iter: 3 i: 197, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 198, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 199, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 200, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 201, pairs changed 0
    fullSet, iter: 3 i: 202, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 203, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 204, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 205, pairs changed 0
    fullSet, iter: 3 i: 206, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 207, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 208, pairs changed 0
    fullSet, iter: 3 i: 209, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 210, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 211, pairs changed 0
    fullSet, iter: 3 i: 212, pairs changed 0
    fullSet, iter: 3 i: 213, pairs changed 0
    fullSet, iter: 3 i: 214, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 215, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 216, pairs changed 0
    fullSet, iter: 3 i: 217, pairs changed 0
    fullSet, iter: 3 i: 218, pairs changed 0
    fullSet, iter: 3 i: 219, pairs changed 0
    fullSet, iter: 3 i: 220, pairs changed 0
    fullSet, iter: 3 i: 221, pairs changed 0
    fullSet, iter: 3 i: 222, pairs changed 0
    fullSet, iter: 3 i: 223, pairs changed 0
    fullSet, iter: 3 i: 224, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 225, pairs changed 0
    fullSet, iter: 3 i: 226, pairs changed 0
    fullSet, iter: 3 i: 227, pairs changed 0
    fullSet, iter: 3 i: 228, pairs changed 0
    fullSet, iter: 3 i: 229, pairs changed 0
    fullSet, iter: 3 i: 230, pairs changed 0
    fullSet, iter: 3 i: 231, pairs changed 0
    fullSet, iter: 3 i: 232, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 233, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 234, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 235, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 236, pairs changed 0
    fullSet, iter: 3 i: 237, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 238, pairs changed 0
    fullSet, iter: 3 i: 239, pairs changed 0
    fullSet, iter: 3 i: 240, pairs changed 0
    fullSet, iter: 3 i: 241, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 242, pairs changed 0
    fullSet, iter: 3 i: 243, pairs changed 0
    fullSet, iter: 3 i: 244, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 245, pairs changed 0
    fullSet, iter: 3 i: 246, pairs changed 0
    fullSet, iter: 3 i: 247, pairs changed 0
    fullSet, iter: 3 i: 248, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 249, pairs changed 0
    fullSet, iter: 3 i: 250, pairs changed 0
    fullSet, iter: 3 i: 251, pairs changed 0
    fullSet, iter: 3 i: 252, pairs changed 0
    fullSet, iter: 3 i: 253, pairs changed 0
    fullSet, iter: 3 i: 254, pairs changed 0
    fullSet, iter: 3 i: 255, pairs changed 0
    fullSet, iter: 3 i: 256, pairs changed 0
    fullSet, iter: 3 i: 257, pairs changed 0
    fullSet, iter: 3 i: 258, pairs changed 0
    fullSet, iter: 3 i: 259, pairs changed 0
    fullSet, iter: 3 i: 260, pairs changed 0
    fullSet, iter: 3 i: 261, pairs changed 0
    fullSet, iter: 3 i: 262, pairs changed 0
    fullSet, iter: 3 i: 263, pairs changed 0
    fullSet, iter: 3 i: 264, pairs changed 0
    fullSet, iter: 3 i: 265, pairs changed 0
    fullSet, iter: 3 i: 266, pairs changed 0
    fullSet, iter: 3 i: 267, pairs changed 0
    fullSet, iter: 3 i: 268, pairs changed 0
    fullSet, iter: 3 i: 269, pairs changed 0
    fullSet, iter: 3 i: 270, pairs changed 0
    fullSet, iter: 3 i: 271, pairs changed 0
    fullSet, iter: 3 i: 272, pairs changed 0
    fullSet, iter: 3 i: 273, pairs changed 0
    fullSet, iter: 3 i: 274, pairs changed 0
    fullSet, iter: 3 i: 275, pairs changed 0
    fullSet, iter: 3 i: 276, pairs changed 0
    fullSet, iter: 3 i: 277, pairs changed 0
    fullSet, iter: 3 i: 278, pairs changed 0
    fullSet, iter: 3 i: 279, pairs changed 0
    fullSet, iter: 3 i: 280, pairs changed 0
    fullSet, iter: 3 i: 281, pairs changed 0
    fullSet, iter: 3 i: 282, pairs changed 0
    fullSet, iter: 3 i: 283, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 284, pairs changed 0
    fullSet, iter: 3 i: 285, pairs changed 0
    fullSet, iter: 3 i: 286, pairs changed 0
    L == H
    fullSet, iter: 3 i: 287, pairs changed 0
    fullSet, iter: 3 i: 288, pairs changed 0
    fullSet, iter: 3 i: 289, pairs changed 0
    fullSet, iter: 3 i: 290, pairs changed 0
    fullSet, iter: 3 i: 291, pairs changed 0
    fullSet, iter: 3 i: 292, pairs changed 0
    L == H
    fullSet, iter: 3 i: 293, pairs changed 0
    fullSet, iter: 3 i: 294, pairs changed 0
    fullSet, iter: 3 i: 295, pairs changed 0
    fullSet, iter: 3 i: 296, pairs changed 0
    fullSet, iter: 3 i: 297, pairs changed 0
    fullSet, iter: 3 i: 298, pairs changed 0
    fullSet, iter: 3 i: 299, pairs changed 0
    fullSet, iter: 3 i: 300, pairs changed 0
    fullSet, iter: 3 i: 301, pairs changed 0
    fullSet, iter: 3 i: 302, pairs changed 0
    fullSet, iter: 3 i: 303, pairs changed 0
    fullSet, iter: 3 i: 304, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 305, pairs changed 0
    L == H
    fullSet, iter: 3 i: 306, pairs changed 0
    fullSet, iter: 3 i: 307, pairs changed 0
    fullSet, iter: 3 i: 308, pairs changed 0
    fullSet, iter: 3 i: 309, pairs changed 0
    fullSet, iter: 3 i: 310, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 311, pairs changed 0
    fullSet, iter: 3 i: 312, pairs changed 0
    fullSet, iter: 3 i: 313, pairs changed 0
    fullSet, iter: 3 i: 314, pairs changed 0
    fullSet, iter: 3 i: 315, pairs changed 0
    fullSet, iter: 3 i: 316, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 317, pairs changed 0
    L == H
    fullSet, iter: 3 i: 318, pairs changed 0
    L == H
    fullSet, iter: 3 i: 319, pairs changed 0
    fullSet, iter: 3 i: 320, pairs changed 0
    fullSet, iter: 3 i: 321, pairs changed 0
    L == H
    fullSet, iter: 3 i: 322, pairs changed 0
    L == H
    fullSet, iter: 3 i: 323, pairs changed 0
    fullSet, iter: 3 i: 324, pairs changed 0
    fullSet, iter: 3 i: 325, pairs changed 0
    fullSet, iter: 3 i: 326, pairs changed 0
    fullSet, iter: 3 i: 327, pairs changed 0
    fullSet, iter: 3 i: 328, pairs changed 0
    fullSet, iter: 3 i: 329, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 330, pairs changed 0
    fullSet, iter: 3 i: 331, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 332, pairs changed 0
    fullSet, iter: 3 i: 333, pairs changed 0
    L == H
    fullSet, iter: 3 i: 334, pairs changed 0
    fullSet, iter: 3 i: 335, pairs changed 0
    fullSet, iter: 3 i: 336, pairs changed 0
    j not moving enough
    fullSet, iter: 3 i: 337, pairs changed 0
    fullSet, iter: 3 i: 338, pairs changed 0
    fullSet, iter: 3 i: 339, pairs changed 0
    fullSet, iter: 3 i: 340, pairs changed 0
    fullSet, iter: 3 i: 341, pairs changed 0
    fullSet, iter: 3 i: 342, pairs changed 0
    fullSet, iter: 3 i: 343, pairs changed 0
    fullSet, iter: 3 i: 344, pairs changed 0
    fullSet, iter: 3 i: 345, pairs changed 0
    fullSet, iter: 3 i: 346, pairs changed 0
    fullSet, iter: 3 i: 347, pairs changed 0
    fullSet, iter: 3 i: 348, pairs changed 0
    fullSet, iter: 3 i: 349, pairs changed 0
    fullSet, iter: 3 i: 350, pairs changed 0
    fullSet, iter: 3 i: 351, pairs changed 0
    fullSet, iter: 3 i: 352, pairs changed 0
    fullSet, iter: 3 i: 353, pairs changed 0
    fullSet, iter: 3 i: 354, pairs changed 0
    fullSet, iter: 3 i: 355, pairs changed 0
    fullSet, iter: 3 i: 356, pairs changed 0
    fullSet, iter: 3 i: 357, pairs changed 0
    fullSet, iter: 3 i: 358, pairs changed 0
    fullSet, iter: 3 i: 359, pairs changed 0
    fullSet, iter: 3 i: 360, pairs changed 0
    fullSet, iter: 3 i: 361, pairs changed 0
    fullSet, iter: 3 i: 362, pairs changed 0
    fullSet, iter: 3 i: 363, pairs changed 0
    fullSet, iter: 3 i: 364, pairs changed 0
    fullSet, iter: 3 i: 365, pairs changed 0
    fullSet, iter: 3 i: 366, pairs changed 0
    fullSet, iter: 3 i: 367, pairs changed 0
    fullSet, iter: 3 i: 368, pairs changed 0
    fullSet, iter: 3 i: 369, pairs changed 0
    fullSet, iter: 3 i: 370, pairs changed 0
    fullSet, iter: 3 i: 371, pairs changed 0
    fullSet, iter: 3 i: 372, pairs changed 0
    fullSet, iter: 3 i: 373, pairs changed 0
    fullSet, iter: 3 i: 374, pairs changed 0
    fullSet, iter: 3 i: 375, pairs changed 0
    fullSet, iter: 3 i: 376, pairs changed 0
    fullSet, iter: 3 i: 377, pairs changed 0
    fullSet, iter: 3 i: 378, pairs changed 0
    fullSet, iter: 3 i: 379, pairs changed 0
    fullSet, iter: 3 i: 380, pairs changed 0
    fullSet, iter: 3 i: 381, pairs changed 0
    fullSet, iter: 3 i: 382, pairs changed 0
    fullSet, iter: 3 i: 383, pairs changed 0
    fullSet, iter: 3 i: 384, pairs changed 0
    fullSet, iter: 3 i: 385, pairs changed 0
    fullSet, iter: 3 i: 386, pairs changed 0
    fullSet, iter: 3 i: 387, pairs changed 0
    fullSet, iter: 3 i: 388, pairs changed 0
    fullSet, iter: 3 i: 389, pairs changed 0
    fullSet, iter: 3 i: 390, pairs changed 0
    fullSet, iter: 3 i: 391, pairs changed 0
    fullSet, iter: 3 i: 392, pairs changed 0
    fullSet, iter: 3 i: 393, pairs changed 0
    fullSet, iter: 3 i: 394, pairs changed 0
    fullSet, iter: 3 i: 395, pairs changed 0
    fullSet, iter: 3 i: 396, pairs changed 0
    fullSet, iter: 3 i: 397, pairs changed 0
    fullSet, iter: 3 i: 398, pairs changed 0
    fullSet, iter: 3 i: 399, pairs changed 0
    fullSet, iter: 3 i: 400, pairs changed 0
    L == H
    fullSet, iter: 3 i: 401, pairs changed 0
    iteration number: 4
    there are 49 Support Vectors
    the training error rate is 0.004975
    the test error rate is 0.016129
    


```python

```
