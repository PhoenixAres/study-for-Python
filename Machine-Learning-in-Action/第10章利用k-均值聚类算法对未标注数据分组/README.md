# 1. K-均值聚类算法

聚类(Clustering)是一种无监督的学习，它将相似的对象归到同一个簇中。

簇识别(cluster identification)，可以给出聚类结果的含义。假定有一些数据，现在将相似数据归到一起，簇识别会告诉我们这些簇到底都是些什么。

聚类与分类的最大不同在于，分类的目标事先已知，而聚类不一样，因此聚类被称为无监督分类(unsupervised classification)。

K-均值是发现给定数据集的k个簇的算法，簇个数k是用户给定的，每一个簇通过质心(centroid)，即簇中所有点的中心来描述。

K-均值聚类

优点：容易实现

缺点：可能收敛到局部最小值，在大规模数据集上收敛较慢

适用数据类型：数值型数据

K-均值聚类的一般流程：

（1）收集数据：使用任意方法

（2）准备数据：需要数值型数据来计算距离，也可以将标称型数据映射为二值型数据再用于距离计算

（3）分析数据：使用任意方法

（4）训练算法：不适用于无监督学习，即无监督学习没有训练过程

（5）测试算法：应用聚类算法、观察结果。可以使用量化的误差指标如误差平方和来评价算法的结果

（6）使用算法：可以用于所希望的任何应用。通常情况下，簇质心可以代表整个簇的数据来做出决策

K-均值算法的工作流程是这样的：

首先，随机确定k个初始点作为质心，然后为每个点找距其最近的质心，并将其分配给该质心所对应的簇，最后将该簇的质心更新为该簇所有点的平均值。

伪代码如下：

    创建k个点作为起始质心（经常是随机选择）
    当任意一个点的簇分配结果发生改变时：
        对数据集中的每个数据点
            对每个质心
                计算质心与数据点之间的距离
            将数据点分配到距其最近的簇
        对每一个簇，计算簇中所有点的均值并将均值作为质心


```python
from numpy import *

def loadDataSet(filename):
    dataMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine))
            dataMat.append(fltLine)
    return dataMat

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = zeros((k, n))
    for j in range(n):               #构建簇质心
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k)
    return centroids

datMat = array(loadDataSet('testSet.txt'))
min(datMat[:, 0])
```




    -5.379713




```python
min(datMat[:, 1])
```




    -4.232586




```python
max(datMat[:, 1])
```




    5.1904




```python
max(datMat[:, 0])
```




    4.838138




```python
randCent(datMat, 2)
```




    array([[ 1.937516  , -1.12242328],
           [-1.76744708,  2.82623485]])




```python
distEclud(datMat[0], datMat[1])
```




    5.184632816681332




```python
def kMeans(dataSet, k, distMeas = distEclud, createCent = randCent):
    m = shape(dataSet)[0]
    clusterAssment = zeros((m, 2))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):                                       #寻找最近的质心
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        print(centroids)
        for cent in range(k):                                      #更新质心的位置
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0] == cent)[0]]
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

datMat = array(loadDataSet('testSet.txt'))
myCentroids, clustAssing = kMeans(datMat, 4)
```

    [[ 4.55058227  2.31156768]
     [-1.07269954  3.23551465]
     [-4.43443397  0.94108917]
     [ 1.11099621  1.32311344]]
    [[ 3.92807891  1.38550982]
     [-1.74777617  3.20930561]
     [-3.54251791 -2.066412  ]
     [ 2.19428421 -0.75487632]]
    [[ 2.82411853  2.92788124]
     [-1.94392522  2.96291883]
     [-3.38237045 -2.9473363 ]
     [ 2.80293085 -2.7315146 ]]
    [[ 2.6265299   3.10868015]
     [-2.46154315  2.78737555]
     [-3.38237045 -2.9473363 ]
     [ 2.80293085 -2.7315146 ]]
    

# 2. 使用后处理来提高聚类性能

K-均值算法收敛但聚类效果较差的原因是，K-均值算法收敛到了局部最小值，而非全局最小值（局部最小值指结果还可以但并非最好结果，全局最小值是可能的最好结果）。

一种用于度量聚类效果的指标是SSE(Sum of Squared Error，误差平方和)，SSE值越小表示数据点越接近于它们的质心，聚类的效果也越好。

因为对误差取了平方，因此更加重视那些远离中心的点，一种肯定可以降低SSE值的方法是增加簇的个数，但这违背了聚类的目标，聚类的目标是在保持簇数目不变的情况下提高簇的质量。

那么如何改进？可以对生成的簇进行后处理，一种方法是将具有最大SSE的值的簇划分成两个簇。为了保持簇总数不变，可以将某两个簇进行合并。

合并簇有两种可以量化的方法：

第一种是合并最近的质心，通过计算所有质心之间的距离，然后合并距离最近的两个点。

第二种是合并两个使得SSE增幅最小的质心，必须在所有可能的两个簇上重复计算这两个簇合并后的总SSE值，直到找到合并最佳的两个簇为止。

# 3. 二分K-均值算法

为克服K-均值算法收敛于局部最小值的问题，有人提出了二分K-均值算法(bisecting K-means)。

该算法首先将所有点作为一个簇，然后将该簇一分为二，之后选择其中一个簇进行划分，选择哪一个簇进行划分取决于对其划分是否可以最大程度降低SSE值。

上述基于SSE的划分过程不断重复，直到得到用户指定的簇数目为止。

二分K-均值算法的伪代码如下：

    将所有点看成一个簇
    当簇数目小于k时
        对每一个簇
            计算总误差
            在给定的簇上进行K-均值聚类（k=2）
            计算将该簇一分为二之后的总误差
        选择使得误差最小的那个簇进行划分操作
        
另一种做法是选择SSE最大的簇进行划分，直至簇数目达到用户指定的数目为止。


```python
def biKmeans(dataSet, k, distMeas = distEclud):
    m = shape(dataSet)[0]
    clusterAssment = zeros((m, 2))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList = [centroid0]                         #创建一个初始簇
    for j in range(m):
        clusterAssment[j, 1] = distMeas(array(centroid0), dataSet[j, :])**2
    while len(centList) < k:
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:, 0] == i)[0], :]    #尝试划分每一簇
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:, 0] != i)[0], 1])
            print('sseSplit, and notSplit: ', sseSplit, sseNotSplit)
            if sseSplit + sseNotSplit < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:, 0] == 1)[0], 0] = len(centList)      #更新簇的分配结果
        bestClustAss[nonzero(bestClustAss[:, 0] == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :]
        centList.append(bestNewCents[1, :])
        clusterAssment[nonzero(clusterAssment[:, 0] == bestCentToSplit)[0], :] = bestClustAss
    return array(centList), clusterAssment

datMat3 = array(loadDataSet('testSet2.txt'))
centList, myNewAssments = biKmeans(datMat3, 3)
centList
```

    [[ 4.33420273  1.76997734]
     [-1.17210054  4.39828475]]
    [[ 1.73007723  0.1226741 ]
     [-2.04552273  2.32798613]]
    [[ 1.23710375  0.17480612]
     [-2.94737575  3.3263781 ]]
    sseSplit, and notSplit:  570.7227574246755 0.0
    the bestCentToSplit is:  0
    the len of bestClustAss is:  60
    [[0.30608236 0.01696057]
     [1.75374656 2.2505281 ]]
    [[-0.45965615 -2.7782156 ]
     [ 2.93386365  3.12782785]]
    sseSplit, and notSplit:  68.68654812621844 38.06295063565756
    [[-4.16478856  4.68882189]
     [-1.55734022  1.94022161]]
    [[-3.85186957  4.07932457]
     [-2.46034062  2.92094538]]
    [[-4.095738    4.4204886 ]
     [-2.56458833  2.9616746 ]]
    sseSplit, and notSplit:  21.290859679422137 532.6598067890178
    the bestCentToSplit is:  0
    the len of bestClustAss is:  40
    




    array([[-0.45965615, -2.7782156 ],
           [-2.94737575,  3.3263781 ],
           [ 2.93386365,  3.12782785]])


