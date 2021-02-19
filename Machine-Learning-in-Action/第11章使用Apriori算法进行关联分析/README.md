# 1. 关联分析

从大规模数据集中寻找物品间的隐含关系被称作关联分析(association analysis)或者关联规则学习(association rule learning)。

这些关系可以有两种形式：频繁项集或者关联规则，频繁项集(frequent item sets)是经常出现在一块的物品的集合，关联规则(association rules)暗示两种物品之间可能存在很强的关系。

一个项集的支持度(support)被定义为数据集中包含该项集的记录所占的比例，支持度是针对项集而言的，因此可以定义一个最小支持度，而只保留满足最小支持度的项集。

可信度或置信度(confidence)是针对关联规则来定义的，是量化关联分析是否成功的方法。

# 2. Apriori原理

Apriori原理是说如果某个项集是频繁的，那么它的所有子集也是频繁的，我们经常运用其逆否命题，即：

如果一个项集是非频繁集，那么它的所有超集也是非频繁的。使用该原理就可以避免项集数目的指数增长，从而在合理时间内计算出频繁项集。

Apriori算法：

优点：易编码实现

缺点：在大数据集上可能较慢

适用数据类型：数值型或者标称型数据

Apriori算法的一般过程：

（1）收集数据：使用任意方法

（2）准备数据：任何数据类型都可以，因为只保存集合

（3）分析数据：使用任意方法

（4）训练算法：使用Apriori算法来找到频繁项集

（5）测试算法：不需要测试过程

（6）使用算法：用于发现频繁项集以及物品之间的关联规则

# 3. 使用Apriori算法来发现频繁集

Apriori算法的两个输入参数分别是最小支持度和数据集，该算法首先会生成所有单个物品的项集列表，接着扫描交易记录去掉不满足最小支持度要求的项集。

然后，对剩下来的集合进行组合以生成包含两个元素的项集，接下来，再重新扫描交易记录，去掉不满足最小支持度的项集，重复该过程直至所有项集被去掉。

## 3.1 生成候选项集

伪代码如下：

    对数据集中的每条交易记录tran
    对每个候选项集can：
        检查一下can是否是tran的子集：
        如果是，则增加can的计数值
    对每个候选项集：
    如果其支持度不低于最小值，则保留该项集
    返回所有频繁项集列表


```python
from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))      #对C1中每个项构建一个不变集合

def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                ssCnt[can] = ssCnt.get(can, 0) + 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems      #计算所有项集的支持度
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

dataSet = loadDataSet()
dataSet
```




    [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]




```python
C1 = createC1(dataSet)
C1
```




    [frozenset({1}),
     frozenset({2}),
     frozenset({3}),
     frozenset({4}),
     frozenset({5})]




```python
D = list(map(set, dataSet))
D
```




    [{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]




```python
L1, suppData0 = scanD(D, C1, 0.5)
L1
```




    [frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})]



上述4个项集构成了L1列表，该列表中的每个单物品项集至少出现在50%以上的记录中，由于物品4没有达到最小支持度，所以被丢弃。

## 3.2 组织完整的Apriori算法

伪代码如下：

    当集合中项的个数大于0时
        构建一个k个项组成的候选项集的列表
        检查数据以确认每个项集都是频繁的
        保留频繁项集并构建k+1项组成的候选项集的列表


```python
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            if L1 == L2:                        #前k-2个项相同时，将两个集合合并
                retList.append(Lk[i] | Lk[j])
    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while len(L[k-2]) > 0:
        Ck = aprioriGen(L[k-2], k)
        Lk, supK = scanD(D, Ck, minSupport)     #扫描数据集，从Ck得到Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData

dataSet = loadDataSet()
L, suppData = apriori(dataSet)
L
```




    [[frozenset({5}), frozenset({2}), frozenset({3}), frozenset({1})],
     [frozenset({2, 3}), frozenset({3, 5}), frozenset({2, 5}), frozenset({1, 3})],
     [frozenset({2, 3, 5})],
     []]




```python
aprioriGen(L[0], 2)
```




    [frozenset({2, 5}),
     frozenset({3, 5}),
     frozenset({1, 5}),
     frozenset({2, 3}),
     frozenset({1, 2}),
     frozenset({1, 3})]




```python
L, suppData = apriori(dataSet, minSupport=0.7)
L
```




    [[frozenset({5}), frozenset({2}), frozenset({3})], [frozenset({2, 5})], []]



# 4. 从频繁项集中挖掘关联规则

一条规则$P -> H$的可信度定义为：

$$ \frac {support(P | H)} {support(P)} $$

其中，P | H 表示集合P与H的并集

为找到感兴趣的规则，我们先生成一个可能的规则列表，然后测试每条规则的可信度，如果可信度不满足最小要求，则去掉该规则。

类似于频繁项集的生成，如果某条规则并不满足最小可信度要求，那么该规则的所有子集也不会满足最小可信度的要求。


```python
def calcConf(freqSet, H, supportData, br1, minConf = 0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            br1.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, br1, minConf = 0.7):
    m = len(H[0])
    if len(freqSet) > m + 1:          #尝试进一步合并
        Hmp1 = aprioriGen(H, m + 1)   #创建Hm+1条新候选规则
        Hmp1 = calcConf(freqSet, Hmp1, supportData, br1, minConf)
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, Hmp1, supportData, br1, minConf)

def generateRules(L, supportData, minConf = 0.7):
    bigRuleList = []
    for i in range(1, len(L)):    #只获取有两个或更多元素的集合
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

dataSet = loadDataSet()
L, suppData = apriori(dataSet, minSupport=0.5)
rules = generateRules(L, suppData, minConf=0.7)
rules
```

    frozenset({5}) --> frozenset({2}) conf: 1.0
    frozenset({2}) --> frozenset({5}) conf: 1.0
    frozenset({1}) --> frozenset({3}) conf: 1.0
    




    [(frozenset({5}), frozenset({2}), 1.0),
     (frozenset({2}), frozenset({5}), 1.0),
     (frozenset({1}), frozenset({3}), 1.0)]



下面降低可信度阈值之后看一下结果：


```python
rules = generateRules(L, suppData, minConf=0.5)
rules
```

    frozenset({3}) --> frozenset({2}) conf: 0.6666666666666666
    frozenset({2}) --> frozenset({3}) conf: 0.6666666666666666
    frozenset({5}) --> frozenset({3}) conf: 0.6666666666666666
    frozenset({3}) --> frozenset({5}) conf: 0.6666666666666666
    frozenset({5}) --> frozenset({2}) conf: 1.0
    frozenset({2}) --> frozenset({5}) conf: 1.0
    frozenset({3}) --> frozenset({1}) conf: 0.6666666666666666
    frozenset({1}) --> frozenset({3}) conf: 1.0
    frozenset({5}) --> frozenset({2, 3}) conf: 0.6666666666666666
    frozenset({3}) --> frozenset({2, 5}) conf: 0.6666666666666666
    frozenset({2}) --> frozenset({3, 5}) conf: 0.6666666666666666
    




    [(frozenset({3}), frozenset({2}), 0.6666666666666666),
     (frozenset({2}), frozenset({3}), 0.6666666666666666),
     (frozenset({5}), frozenset({3}), 0.6666666666666666),
     (frozenset({3}), frozenset({5}), 0.6666666666666666),
     (frozenset({5}), frozenset({2}), 1.0),
     (frozenset({2}), frozenset({5}), 1.0),
     (frozenset({3}), frozenset({1}), 0.6666666666666666),
     (frozenset({1}), frozenset({3}), 1.0),
     (frozenset({5}), frozenset({2, 3}), 0.6666666666666666),
     (frozenset({3}), frozenset({2, 5}), 0.6666666666666666),
     (frozenset({2}), frozenset({3, 5}), 0.6666666666666666)]



可以看到，一旦降低可信度阈值，就可以获得更多的规则。

# 5. 示例：发现毒蘑菇的相似特征

有时候我们并不想寻找所有的频繁项集，而只对包含某个特定元素项的项集感兴趣。

例如，我们会寻找毒蘑菇的一些公共特征，利用这些特征就能避免吃到那些有毒的蘑菇。

在数据集中，第一个特征表示有毒或者可食用，如果样本有毒，则值为2，如果可食用，则值为1。

为了找到毒蘑菇中存在的公共特征，可以寻找包含特征值为2的频繁项集。


```python
mushDatSet = []
with open('mushroom.dat') as f:
    mushDatSet = [line.split() for line in f.readlines()]
L, suppData = apriori(mushDatSet, minSupport=0.3)
for item in L[1]:
    if item.intersection('2'):
        print(item)
```

    frozenset({'28', '2'})
    frozenset({'2', '53'})
    frozenset({'23', '2'})
    frozenset({'34', '2'})
    frozenset({'36', '2'})
    frozenset({'2', '59'})
    frozenset({'2', '63'})
    frozenset({'67', '2'})
    frozenset({'2', '76'})
    frozenset({'85', '2'})
    frozenset({'2', '86'})
    frozenset({'90', '2'})
    frozenset({'2', '93'})
    frozenset({'39', '2'})
    

也可以对更大的项集重复上述过程：


```python
for item in L[3]:
    if item.intersection('2'):
        print(item)
```

接下来，需要观察这些特征，以便了解蘑菇的特征。


```python

```
