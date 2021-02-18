# 1. 复杂数据的局部性建模

第3章决策树中的树构建算法是ID3，ID3算法存在两大问题：

其一，是切分方式过于迅速，具体来说，如果一个特征有4种取值，那么数据将被切成4份。一旦按某特征切分后，该特征在之后的算法执行过程中不会再起作用。

那么另一种做法是二元切分法，即每次把数据集切成两份，如果数据的某特征值等于切分所要求的值，则进入左子树，否则进入右子树。

其二，是不能直接处理连续型特征，再将连续型特征转换成离散型的过程中，会破坏连续变量的内在性质。

而二元切分法则易于处理，如果特征值大于给定值就走左子树，否则走右子树。

CART(Classification And Regression Trees，分类回归树)是十分著名的树构建算法，它使用二元切分来处理连续型变量。

如果利用其它方法代替香农熵，来度量集合的无组织程度，就可以使用树构建算法来完成回归。

树回归

优点：可以对复杂和非线性的数据建模

缺点：结果不易理解

适用数据类型：数值型和标称型数据

树回归的一般方法：

（1）收集数据：采用任意方法

（2）准备数据：需要数值型数据，标称型数据应该映射成二值型数据

（3）分析数据：绘出数据的二维可视化显示结果，以字典方式生成树

（4）训练算法：大部分时间花费在叶节点树模型的构建上

（5）测试算法：使用测试数据上的$R^2$值来分析模型的效果

（6）使用算法：使用训练出的树做预测

# 2. 连续和离散型特征的树的构建

使用字典来存储树，该字典将包含以下4个元素：

（1）待切分的特征

（2）待切分的特征值

（3）右子树，当不需要再切分时，也可以是单个值

（4）左子树，与右子树类似

函数createTree()的伪代码大致如下：

    找到最佳的待切分特征：
        如果该节点不能再分，将该节点存为叶节点
        执行二元切分
        在左子树调用createTree()方法
        在右子树调用createTree()方法


```python
from numpy import *

def loadDataSet(filename):
    dataMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine))  #将每行映射成浮点数
            dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

testMat = eye(4)
testMat
```




    array([[1., 0., 0., 0.],
           [0., 1., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])




```python
mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
mat0
```




    array([[0., 1., 0., 0.]])




```python
mat1
```




    array([[1., 0., 0., 0.],
           [0., 0., 1., 0.],
           [0., 0., 0., 1.]])



# 3. 将CART算法用于回归

回归树假设叶节点是常数值，这种策略认为数据中的复杂关系可以用树的结构来概括。

为成功构建以分段常数为叶节点的树，需要度量出数据的一致性，这里的方法是：

首先计算所有数据的均值，然后计算每条数据到均值的差值，为了对正负差值同等看待，一般使用绝对值或平方值。

与统计学中常用的方差计算不同的是，这里需要的是平方误差的总值（总方差），可以通过均方差乘以数据集中的样本点个数得到。

## 3.1 构建树

函数chooseBestSplit()的目标是找到数据集切分的最佳位置，它遍历所有的特征及其可能的取值来找到使误差最小化的切分阈值。伪代码如下：

    对每个特征：
        对每个特征值：
            将数据集切分成两份
            计算切分的误差
            如果当前误差小于当前最小误差，那么将当前切分设定为最佳切分并更新最小误差
    返回最佳切分的特征和阈值


```python
def regLeaf(dataSet):
    return mean(dataSet[:, -1])

def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]

def chooseBestSplit(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist())) == 1:         #如果所有值相等则退出
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if S - bestS < tolS:                   #如果误差减少不大则退出
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):       #如果切分出的数据集很小则退出
        return None, leafType(dataSet)
    return bestIndex, bestValue

def createTree(dataSet, leafType = regLeaf, errType = regErr, ops = (1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat == None:
        return val         #满足停止条件时返回叶节点值
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree
```

函数chooseBestSplit()一开始为ops设定了tolS和tolN两个值，它们是用户指定的参数，用于控制函数的停止时机。

其中变量tolS是容许的误差下降值，tolN是切分的最少样本数。

## 3.2 运行代码


```python
myDat = loadDataSet('ex00.txt')
myMat = array(myDat)
createTree(myMat)
```




    {'spInd': 0,
     'spVal': 0.48813,
     'left': 1.0180967672413792,
     'right': -0.04465028571428572}




```python
myDat1 = loadDataSet('ex0.txt')
myMat1 = array(myDat1)
createTree(myMat1)
```




    {'spInd': 1,
     'spVal': 0.39435,
     'left': {'spInd': 1,
      'spVal': 0.582002,
      'left': {'spInd': 1,
       'spVal': 0.797583,
       'left': 3.9871632,
       'right': 2.9836209534883724},
      'right': 1.980035071428571},
     'right': {'spInd': 1,
      'spVal': 0.197834,
      'left': 1.0289583666666666,
      'right': -0.023838155555555553}}



# 4. 树剪枝

一棵树如果节点过多，表明该模型可能对数据进行了“过拟合”，可以使用交叉验证来发现过拟合现象。

通过降低决策树的复杂度来避免过拟合的过程，称为剪枝(pruning)。

在函数chooseBestSplit()中的提前终止条件，实际上就是一种预剪枝(prepruning)，另一种剪枝需要使用测试集和训练集，称作后剪枝(postpruning)。

## 4.1 预剪枝

上述树构建算法对输入的参数tolS和tolN非常敏感：


```python
myDat2 = loadDataSet('ex2.txt')
myMat2 = array(myDat2)
print(createTree(myMat2))
```

    {'spInd': 0, 'spVal': 0.499171, 'left': {'spInd': 0, 'spVal': 0.729397, 'left': {'spInd': 0, 'spVal': 0.952833, 'left': {'spInd': 0, 'spVal': 0.958512, 'left': 105.24862350000001, 'right': 112.42895575000001}, 'right': {'spInd': 0, 'spVal': 0.759504, 'left': {'spInd': 0, 'spVal': 0.790312, 'left': {'spInd': 0, 'spVal': 0.833026, 'left': {'spInd': 0, 'spVal': 0.944221, 'left': 87.3103875, 'right': {'spInd': 0, 'spVal': 0.85497, 'left': {'spInd': 0, 'spVal': 0.910975, 'left': 96.452867, 'right': {'spInd': 0, 'spVal': 0.892999, 'left': 104.825409, 'right': {'spInd': 0, 'spVal': 0.872883, 'left': 95.181793, 'right': 102.25234449999999}}}, 'right': 95.27584316666666}}, 'right': {'spInd': 0, 'spVal': 0.811602, 'left': 81.110152, 'right': 88.78449880000001}}, 'right': 102.35780185714285}, 'right': 78.08564325}}, 'right': {'spInd': 0, 'spVal': 0.640515, 'left': {'spInd': 0, 'spVal': 0.666452, 'left': {'spInd': 0, 'spVal': 0.706961, 'left': 114.554706, 'right': {'spInd': 0, 'spVal': 0.698472, 'left': 104.82495374999999, 'right': 108.92921799999999}}, 'right': 114.1516242857143}, 'right': {'spInd': 0, 'spVal': 0.613004, 'left': 93.67344971428572, 'right': {'spInd': 0, 'spVal': 0.582311, 'left': 123.2101316, 'right': {'spInd': 0, 'spVal': 0.553797, 'left': 97.20018024999999, 'right': {'spInd': 0, 'spVal': 0.51915, 'left': {'spInd': 0, 'spVal': 0.543843, 'left': 109.38961049999999, 'right': 110.979946}, 'right': 101.73699325000001}}}}}}, 'right': {'spInd': 0, 'spVal': 0.457563, 'left': {'spInd': 0, 'spVal': 0.467383, 'left': 12.50675925, 'right': 3.4331330000000007}, 'right': {'spInd': 0, 'spVal': 0.126833, 'left': {'spInd': 0, 'spVal': 0.373501, 'left': {'spInd': 0, 'spVal': 0.437652, 'left': -12.558604833333334, 'right': {'spInd': 0, 'spVal': 0.412516, 'left': 14.38417875, 'right': {'spInd': 0, 'spVal': 0.385021, 'left': -0.8923554999999995, 'right': 3.6584772500000016}}}, 'right': {'spInd': 0, 'spVal': 0.335182, 'left': {'spInd': 0, 'spVal': 0.350725, 'left': -15.08511175, 'right': -22.693879600000002}, 'right': {'spInd': 0, 'spVal': 0.324274, 'left': 15.05929075, 'right': {'spInd': 0, 'spVal': 0.297107, 'left': -19.9941552, 'right': {'spInd': 0, 'spVal': 0.166765, 'left': {'spInd': 0, 'spVal': 0.202161, 'left': {'spInd': 0, 'spVal': 0.217214, 'left': {'spInd': 0, 'spVal': 0.228473, 'left': {'spInd': 0, 'spVal': 0.25807, 'left': 0.40377471428571476, 'right': -13.070501}, 'right': 6.770429}, 'right': -11.822278500000001}, 'right': 3.4496025}, 'right': {'spInd': 0, 'spVal': 0.156067, 'left': -12.1079725, 'right': -6.247900000000001}}}}}}, 'right': {'spInd': 0, 'spVal': 0.084661, 'left': 6.509843285714284, 'right': {'spInd': 0, 'spVal': 0.044737, 'left': -2.544392714285715, 'right': 4.091626}}}}}
    


```python
createTree(myMat2, ops=(10000, 4))
```




    {'spInd': 0,
     'spVal': 0.499171,
     'left': 101.35815937735848,
     'right': -2.637719329787234}



可以看到，停止条件tolS对误差的数量级十分敏感。然而，通过不断修改停止条件来得到合理结果并不是很好的方法。

## 4.2 后剪枝

首先指定参数，使得构建出的树足够大，足够复杂，便于剪枝。

接下来从上而下找到叶节点，用测试集来判断将这些叶节点合并是否能降低测试误差。如果是的话就合并。

伪代码如下：

    基于已有的树切分测试数据：
        如果存在任一子集是一棵树，则在该子集递归剪枝过程
        计算将当前两个叶节点合并后的误差
        计算不合并的误差
        如果合并会降低误差的话，就将叶节点合并


```python
def isTree(obj):
    return type(obj) == dict

def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0

def prune(tree, testData):
    if shape(testData)[0] == 0:     #没有测试数据则对树进行塌陷处理
        return getMean(tree)
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:, -1] - tree['left'], 2)) + sum(power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
#             print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree

myDat2 = loadDataSet('ex2.txt')
myMat2 = array(myDat2)
myTree = createTree(myMat2, ops=(0, 1))
myDatTest = loadDataSet('ex2test.txt')
myMat2Test = array(myDatTest)
print(prune(myTree, myMat2Test))
```

    {'spInd': 0, 'spVal': 0.499171, 'left': {'spInd': 0, 'spVal': 0.729397, 'left': {'spInd': 0, 'spVal': 0.952833, 'left': {'spInd': 0, 'spVal': 0.965969, 'left': 92.5239915, 'right': {'spInd': 0, 'spVal': 0.956951, 'left': {'spInd': 0, 'spVal': 0.958512, 'left': {'spInd': 0, 'spVal': 0.960398, 'left': 112.386764, 'right': 123.559747}, 'right': 135.837013}, 'right': 111.2013225}}, 'right': {'spInd': 0, 'spVal': 0.759504, 'left': {'spInd': 0, 'spVal': 0.763328, 'left': {'spInd': 0, 'spVal': 0.769043, 'left': {'spInd': 0, 'spVal': 0.790312, 'left': {'spInd': 0, 'spVal': 0.806158, 'left': {'spInd': 0, 'spVal': 0.815215, 'left': {'spInd': 0, 'spVal': 0.833026, 'left': {'spInd': 0, 'spVal': 0.841547, 'left': {'spInd': 0, 'spVal': 0.841625, 'left': {'spInd': 0, 'spVal': 0.944221, 'left': {'spInd': 0, 'spVal': 0.948822, 'left': 96.41885225, 'right': 69.318649}, 'right': {'spInd': 0, 'spVal': 0.85497, 'left': {'spInd': 0, 'spVal': 0.936524, 'left': 110.03503850000001, 'right': {'spInd': 0, 'spVal': 0.934853, 'left': 65.548418, 'right': {'spInd': 0, 'spVal': 0.925782, 'left': 115.753994, 'right': {'spInd': 0, 'spVal': 0.910975, 'left': {'spInd': 0, 'spVal': 0.912161, 'left': 94.3961145, 'right': 85.005351}, 'right': {'spInd': 0, 'spVal': 0.901444, 'left': {'spInd': 0, 'spVal': 0.908629, 'left': 106.814667, 'right': 118.513475}, 'right': {'spInd': 0, 'spVal': 0.901421, 'left': 87.300625, 'right': {'spInd': 0, 'spVal': 0.892999, 'left': {'spInd': 0, 'spVal': 0.900699, 'left': 100.133819, 'right': 108.094934}, 'right': {'spInd': 0, 'spVal': 0.888426, 'left': 82.436686, 'right': {'spInd': 0, 'spVal': 0.872199, 'left': 98.54454949999999, 'right': 106.16859550000001}}}}}}}}}, 'right': {'spInd': 0, 'spVal': 0.84294, 'left': {'spInd': 0, 'spVal': 0.847219, 'left': 89.20993, 'right': 76.240984}, 'right': 95.893131}}}, 'right': 60.552308}, 'right': 124.87935300000001}, 'right': {'spInd': 0, 'spVal': 0.823848, 'left': 76.723835, 'right': {'spInd': 0, 'spVal': 0.819722, 'left': 59.342323, 'right': 70.054508}}}, 'right': {'spInd': 0, 'spVal': 0.811602, 'left': 118.319942, 'right': {'spInd': 0, 'spVal': 0.811363, 'left': 99.841379, 'right': 112.981216}}}, 'right': 73.49439925}, 'right': {'spInd': 0, 'spVal': 0.786865, 'left': 114.4008695, 'right': 102.26514075}}, 'right': 64.041941}, 'right': 115.199195}, 'right': 78.08564325}}, 'right': {'spInd': 0, 'spVal': 0.640515, 'left': {'spInd': 0, 'spVal': 0.642373, 'left': {'spInd': 0, 'spVal': 0.642707, 'left': {'spInd': 0, 'spVal': 0.665329, 'left': {'spInd': 0, 'spVal': 0.706961, 'left': {'spInd': 0, 'spVal': 0.70889, 'left': {'spInd': 0, 'spVal': 0.716211, 'left': 110.90283, 'right': {'spInd': 0, 'spVal': 0.710234, 'left': 103.345308, 'right': 108.553919}}, 'right': 135.416767}, 'right': {'spInd': 0, 'spVal': 0.698472, 'left': {'spInd': 0, 'spVal': 0.69892, 'left': {'spInd': 0, 'spVal': 0.699873, 'left': {'spInd': 0, 'spVal': 0.70639, 'left': 106.180427, 'right': 105.062147}, 'right': 115.586605}, 'right': 92.470636}, 'right': {'spInd': 0, 'spVal': 0.689099, 'left': 120.521925, 'right': {'spInd': 0, 'spVal': 0.666452, 'left': 101.91115275, 'right': 112.78136649999999}}}}, 'right': {'spInd': 0, 'spVal': 0.661073, 'left': 121.980607, 'right': {'spInd': 0, 'spVal': 0.652462, 'left': 115.687524, 'right': 112.715799}}}, 'right': 82.500766}, 'right': 140.613941}, 'right': {'spInd': 0, 'spVal': 0.613004, 'left': {'spInd': 0, 'spVal': 0.623909, 'left': {'spInd': 0, 'spVal': 0.628061, 'left': {'spInd': 0, 'spVal': 0.637999, 'left': 82.713621, 'right': {'spInd': 0, 'spVal': 0.632691, 'left': 91.656617, 'right': 93.645293}}, 'right': {'spInd': 0, 'spVal': 0.624827, 'left': 117.628346, 'right': 105.970743}}, 'right': 82.04976400000001}, 'right': {'spInd': 0, 'spVal': 0.606417, 'left': 168.180746, 'right': {'spInd': 0, 'spVal': 0.513332, 'left': {'spInd': 0, 'spVal': 0.533511, 'left': {'spInd': 0, 'spVal': 0.548539, 'left': {'spInd': 0, 'spVal': 0.553797, 'left': {'spInd': 0, 'spVal': 0.560301, 'left': {'spInd': 0, 'spVal': 0.599142, 'left': 93.521396, 'right': {'spInd': 0, 'spVal': 0.589806, 'left': 130.378529, 'right': {'spInd': 0, 'spVal': 0.582311, 'left': 111.9849935, 'right': {'spInd': 0, 'spVal': 0.571214, 'left': 82.589328, 'right': {'spInd': 0, 'spVal': 0.569327, 'left': 114.872056, 'right': 108.435392}}}}}, 'right': 82.903945}, 'right': 129.0624485}, 'right': {'spInd': 0, 'spVal': 0.546601, 'left': 83.114502, 'right': {'spInd': 0, 'spVal': 0.537834, 'left': 97.3405265, 'right': 90.995536}}}, 'right': {'spInd': 0, 'spVal': 0.51915, 'left': {'spInd': 0, 'spVal': 0.531944, 'left': 129.766743, 'right': 124.795495}, 'right': 116.176162}}, 'right': {'spInd': 0, 'spVal': 0.508548, 'left': 101.075609, 'right': {'spInd': 0, 'spVal': 0.508542, 'left': 93.292829, 'right': 96.403373}}}}}}}, 'right': {'spInd': 0, 'spVal': 0.457563, 'left': {'spInd': 0, 'spVal': 0.465561, 'left': {'spInd': 0, 'spVal': 0.467383, 'left': {'spInd': 0, 'spVal': 0.483803, 'left': {'spInd': 0, 'spVal': 0.487381, 'left': 8.53677, 'right': 27.729263}, 'right': 5.224234}, 'right': {'spInd': 0, 'spVal': 0.46568, 'left': -9.712925, 'right': -23.777531}}, 'right': {'spInd': 0, 'spVal': 0.463241, 'left': 30.051931, 'right': 17.171057}}, 'right': {'spInd': 0, 'spVal': 0.455761, 'left': -34.044555, 'right': {'spInd': 0, 'spVal': 0.126833, 'left': {'spInd': 0, 'spVal': 0.130626, 'left': {'spInd': 0, 'spVal': 0.382037, 'left': {'spInd': 0, 'spVal': 0.388789, 'left': {'spInd': 0, 'spVal': 0.437652, 'left': -4.1911745, 'right': {'spInd': 0, 'spVal': 0.412516, 'left': {'spInd': 0, 'spVal': 0.418943, 'left': {'spInd': 0, 'spVal': 0.426711, 'left': {'spInd': 0, 'spVal': 0.428582, 'left': 19.745224, 'right': 15.224266}, 'right': -21.594268}, 'right': 44.161493}, 'right': {'spInd': 0, 'spVal': 0.403228, 'left': -26.419289, 'right': 0.6359300000000001}}}, 'right': 23.197474}, 'right': {'spInd': 0, 'spVal': 0.335182, 'left': {'spInd': 0, 'spVal': 0.370042, 'left': {'spInd': 0, 'spVal': 0.378965, 'left': -29.007783, 'right': {'spInd': 0, 'spVal': 0.373501, 'left': {'spInd': 0, 'spVal': 0.377383, 'left': 13.583555, 'right': 5.241196}, 'right': -8.228297}}, 'right': {'spInd': 0, 'spVal': 0.35679, 'left': -32.124495, 'right': {'spInd': 0, 'spVal': 0.350725, 'left': -9.9938275, 'right': -26.851234812500003}}}, 'right': {'spInd': 0, 'spVal': 0.324274, 'left': 22.286959625, 'right': {'spInd': 0, 'spVal': 0.309133, 'left': {'spInd': 0, 'spVal': 0.310956, 'left': -20.3973335, 'right': -49.939516}, 'right': {'spInd': 0, 'spVal': 0.131833, 'left': {'spInd': 0, 'spVal': 0.138619, 'left': {'spInd': 0, 'spVal': 0.156067, 'left': {'spInd': 0, 'spVal': 0.166765, 'left': {'spInd': 0, 'spVal': 0.193282, 'left': {'spInd': 0, 'spVal': 0.211633, 'left': {'spInd': 0, 'spVal': 0.228473, 'left': {'spInd': 0, 'spVal': 0.25807, 'left': {'spInd': 0, 'spVal': 0.284794, 'left': {'spInd': 0, 'spVal': 0.300318, 'left': 8.814725, 'right': {'spInd': 0, 'spVal': 0.297107, 'left': -18.051318, 'right': {'spInd': 0, 'spVal': 0.295993, 'left': -1.798377, 'right': {'spInd': 0, 'spVal': 0.290749, 'left': -14.988279, 'right': -14.391613}}}}, 'right': {'spInd': 0, 'spVal': 0.273863, 'left': 35.623746, 'right': {'spInd': 0, 'spVal': 0.264926, 'left': -9.457556, 'right': {'spInd': 0, 'spVal': 0.264639, 'left': 5.280579, 'right': 2.557923}}}}, 'right': {'spInd': 0, 'spVal': 0.228628, 'left': {'spInd': 0, 'spVal': 0.228751, 'left': -9.601409499999999, 'right': -30.812912}, 'right': -2.266273}}, 'right': 6.099239}, 'right': {'spInd': 0, 'spVal': 0.202161, 'left': -16.42737025, 'right': -2.6781805}}, 'right': 9.5773855}, 'right': {'spInd': 0, 'spVal': 0.156273, 'left': {'spInd': 0, 'spVal': 0.164134, 'left': {'spInd': 0, 'spVal': 0.166431, 'left': -14.740059, 'right': -6.512506}, 'right': -27.405211}, 'right': 0.225886}}, 'right': {'spInd': 0, 'spVal': 0.13988, 'left': 7.557349, 'right': 7.336784}}, 'right': -29.087463}, 'right': 22.478291}}}}}, 'right': -39.524461}, 'right': {'spInd': 0, 'spVal': 0.124723, 'left': 22.891675, 'right': {'spInd': 0, 'spVal': 0.085111, 'left': {'spInd': 0, 'spVal': 0.108801, 'left': 6.196516, 'right': {'spInd': 0, 'spVal': 0.10796, 'left': -16.106164, 'right': {'spInd': 0, 'spVal': 0.085873, 'left': -1.293195, 'right': -10.137104}}}, 'right': {'spInd': 0, 'spVal': 0.084661, 'left': 37.820659, 'right': {'spInd': 0, 'spVal': 0.080061, 'left': -24.132226, 'right': {'spInd': 0, 'spVal': 0.068373, 'left': 15.824970500000001, 'right': {'spInd': 0, 'spVal': 0.061219, 'left': -15.160836, 'right': {'spInd': 0, 'spVal': 0.044737, 'left': {'spInd': 0, 'spVal': 0.053764, 'left': {'spInd': 0, 'spVal': 0.055862, 'left': 6.695567, 'right': -3.131497}, 'right': -13.731698}, 'right': 4.091626}}}}}}}}}}}
    

可以看到，大量的节点已经被剪掉了，但没有像预期的那样剪枝成两部分，这说明后剪枝可能不如预剪枝有效。为了寻求最佳模型，两种剪枝技术可以同时使用。

# 5. 模型树

用树来对数据建模，除了把叶节点简单地设定为常数值之外，还有一种方法就是把叶节点设定为分段线性函数。

这里的分段线性(piecewise linear)是指模型由多个线性片段组成。

决策树相比与其他机器学习算法的优势之一是在于结果更易理解。很显然，两条直线比很多节点组成一棵大树更容易解释。

模型树的可解释性是它优于回归树的特点之一，此外，模型树也具有更高的预测准确度。


```python
def linearSolve(dataSet):
    m, n = shape(dataSet)
    X = ones((m, n))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = array([dataSet[:, -1]]).T      #将X与Y中的数据格式化
    xTx = dot(X.T, X)
    ws = dot(linalg.pinv(xTx), dot(X.T, Y))
    return ws, X, Y

def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = dot(X, ws)
    return sum(power(Y - yHat, 2))

myMat2 = array(loadDataSet('exp2.txt'))
print(createTree(myMat2, modelLeaf, modelErr, (1, 10)))
```

    {'spInd': 0, 'spVal': 0.285477, 'left': array([[1.69855694e-03],
           [1.19647739e+01]]), 'right': array([[3.46877936],
           [1.18521743]])}
    

可以看到，该代码以0.285477为界创建了两个模型，createTree()生成的两个线性模型分别是y=3.468+1.1852x和y=0.0016985+11.96477x，

而数据实际模型是由模型y=3.5+1.0x和y=0+12x再加上高斯噪声生成的，该代码生成的模型与真实模型非常相近。

# 6. 示例：树回归与标准回归的比较


```python
def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[0]
    X = ones((1, n+1))
    X[:, 1:n+1] = inDat
    return float(dot(X, model))

def treeForeCast(tree, inData, modelEval = regTreeEval):
    if not isTree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)

def createForeCast(tree, testData, modelEval = regTreeEval):
    m = len(testData)
    yHat = zeros((m, 1))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, array([testData[i]]).T, modelEval)
    return yHat
```

先利用该数据创建一棵回归树：


```python
trainMat = array(loadDataSet('bikeSpeedVsIq_train.txt'))
testMat = array(loadDataSet('bikeSpeedVsIq_test.txt'))
myTree = createTree(trainMat, ops=(1, 20))
yHat = createForeCast(myTree, testMat[:, 0])
corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
```




    0.964085231822215



再创建一棵模型树：


```python
myTree = createTree(trainMat, modelLeaf, modelErr, (1, 20))
yHat = createForeCast(myTree, testMat[:, 0], modelTreeEval)
corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
```




    0.9760412191380615



我们知道，$R^2$值越接近1.0越好，所以从上面结果看出，模型树的结果比回归树好。


```python
ws, X, Y = linearSolve(trainMat)
ws
```




    array([[37.58916794],
           [ 6.18978355]])




```python
for i in range(shape(testMat)[0]):
    yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1]
```




    0.9434684235674766



上述结果是标准的线性回归给出的结果，可以看到，该方法在$R^2$值上的表现不如上面两种树回归的方法。
