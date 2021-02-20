# 1. FP树：用于编码数据集的有效方式

FP-growth算法将数据存储在一种称为FP树的紧凑数据结构中，FP代表频繁模式(Frequent Pattern)，它通过链表(link)来连接相似元素。

同搜索树不同的是，一个元素项可以在一棵FP树中出现多次，FP树会存储项集的出现频率，而每个项集会以路径的方式存储在树中。

存在相似元素的集合会共享树的一部分，只有当集合完全不同时，树才会分叉。

树节点上给出集合中的单个元素及其在序列中的出现次数，路径会给出该序列的出现次数。

相似项之间的链接即节点链接(node link)，用于快速发现相似项的位置。

FP-growth算法

优点：一般要快于Apriori

缺点：实现比较困难，在某些数据集上性能会下降

适用数据类型：标称型数据

FP-growth的一般流程

（1）收集数据：使用任意方法

（2）准备数据：由于存储的是集合，所以需要离散数据

（3）分析数据：使用任意方法

（4）训练算法：构建一个FP树，并对树进行挖掘

（5）测试算法：没有测试过程

（6）使用算法：可用于识别经常出现的元素项

# 2. 构建FP树

构建FP树，需要对原始数据集扫描两遍，第一遍统计出现频率，第二遍扫描中只考虑那些频繁元素。

## 2.1 创建FP树的数据结构


```python
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}
    def inc(self, numOccur):
        self.count += numOccur
    def disp(self, ind = 1):
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)

rootNode = treeNode('pyramid', 9, None)
rootNode.children['eye'] = treeNode('eye', 13, None)
rootNode.disp()
```

       pyramid   9
         eye   13
    


```python
rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
rootNode.disp()
```

       pyramid   9
         eye   13
         phoenix   3
    

## 2.2 构建FP树

第一次遍历数据集会获得每个元素项出现的频率，接下来，去掉不满足最小支持度的元素项，接着构建FP树。

在构建时，读入每个项集并将其添加到一条已存在的路径中，如果该路径不存在，则创建一条新路径。

在将集合添加到树之前，需要对每个集合进行排序和过滤，排序基于元素项的绝对出现频率来进行，过滤掉不满足最小支持项的元素项。

排序和过滤结束以后，添加到树中，如果树中已存在现有元素，则增加现有元素的值，如果元素不存在，则向树添加一个分枝。


```python
def updateHeader(nodeToTest, targetNode):
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)  #对剩下的元素项迭代调用updateTree函数

def createTree(dataSet, minSup = 1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    headerTable = {k : v for k, v in headerTable.items() if v >= minSup}  #移除不满足最小支持度的元素项
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None         #如果没有元素项满足要求，则退出
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localD = {}
        for item in tranSet:                        #根据全局频率对每个事务中的元素进行排序
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda x:x[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)  #使用排序后的频率项集对树进行填充
    return retTree, headerTable

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

simpDat = loadSimpDat()
simpDat
```




    [['r', 'z', 'h', 'j', 'p'],
     ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
     ['z'],
     ['r', 'x', 'n', 'o', 's'],
     ['y', 'r', 'x', 'z', 'q', 't', 'p'],
     ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]




```python
initSet = createInitSet(simpDat)
initSet
```




    {frozenset({'h', 'j', 'p', 'r', 'z'}): 1,
     frozenset({'s', 't', 'u', 'v', 'w', 'x', 'y', 'z'}): 1,
     frozenset({'z'}): 1,
     frozenset({'n', 'o', 'r', 's', 'x'}): 1,
     frozenset({'p', 'q', 'r', 't', 'x', 'y', 'z'}): 1,
     frozenset({'e', 'm', 'q', 's', 't', 'x', 'y', 'z'}): 1}




```python
myFPtree, myHeaderTab = createTree(initSet, 3)
myFPtree.disp()
```

       Null Set   1
         z   5
           r   1
           x   3
             s   1
               t   1
                 y   1
             r   1
               t   1
                 y   1
             t   1
               s   1
                 y   1
         x   1
           r   1
             s   1
    

# 3. 从一棵FP树中挖掘频繁项集

从FP树中抽取频繁项集的三个基本步骤如下：

（1）从FP树中获得条件模式基

（2）利用条件模式基，构建一个条件FP树

（3）迭代重复步骤（1）步骤（2），直到树包含一个元素项为止。

## 3.1 抽取条件模式基

条件模式基(conditional pattern base)是以所查找元素项为结尾的路径集合，每一条路径其实都是一条前缀路径(prefix path)。


```python
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:           #迭代上溯整棵树
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

findPrefixPath('x', myHeaderTab['x'][1])
```




    {frozenset({'z'}): 3}




```python
findPrefixPath('z', myHeaderTab['z'][1])
```




    {}




```python
findPrefixPath('r', myHeaderTab['r'][1])
```




    {frozenset({'z'}): 1, frozenset({'x'}): 1, frozenset({'x', 'z'}): 1}



## 3.2 创建条件FP树

对于每一个频繁项，都要创建一棵条件FP树，可以利用刚才发现的条件模式基作为输入数据。

例如，t的条件FP树创建过程如下：

最初树以空集作为根节点，接下来，原始的集合{y, x, s, z}中的集合{y, x, z}被添加进来，s因为不符合最小支持度要求被舍去。类似地，{y, x, z}也从原始集合{y, x, r, z}中添加进来。

可以注意到，单独看s和r，它们都是频繁项，但在t的条件树中，它们却是不频繁的，即{t, r}和{t, s}是不频繁的。

接下来，对集合{t, z}、{t, x}和{t, y}来挖掘对应的条件树，重复进行，直到条件树中没有元素为止。


```python
def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda x:x[1][0])]     #从头指针表的底端开始
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)          #从条件模式基来构建条件FP树
        if myHead != None:                                              #挖掘条件FP树
            print('conditional tree for: ', newFreqSet)
            myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)
            
freqItems = []
mineTree(myFPtree, myHeaderTab, 3, set(), freqItems)
```

    conditional tree for:  {'s'}
       Null Set   1
         x   3
    conditional tree for:  {'t'}
       Null Set   1
         z   3
           x   3
    conditional tree for:  {'t', 'x'}
       Null Set   1
         z   3
    conditional tree for:  {'x'}
       Null Set   1
         z   3
    


```python
freqItems
```




    [{'r'},
     {'s'},
     {'s', 'x'},
     {'t'},
     {'t', 'z'},
     {'t', 'x'},
     {'t', 'x', 'z'},
     {'y'},
     {'x'},
     {'x', 'z'},
     {'z'}]



# 4. 示例：从新闻网站点击流中挖掘

构建FP树，并从中寻找那些至少被10万人浏览过的新闻报道：


```python
parsedDat = []
with open('kosarak.dat') as f:
    parsedDat = [line.split() for line in f.readlines()]
initSet = createInitSet(parsedDat)
myFPtree, myHeaderTab = createTree(initSet, 100000)
myFreqList = []
mineTree(myFPtree, myHeaderTab, 100000, set(), myFreqList)
len(myFreqList)
```

    conditional tree for:  {'1'}
       Null Set   1
         6   107404
    conditional tree for:  {'3'}
       Null Set   1
         6   186289
           11   117401
         11   9718
    conditional tree for:  {'11', '3'}
       Null Set   1
         6   117401
    conditional tree for:  {'11'}
       Null Set   1
         6   261773
    




    9




```python
myFreqList
```




    [{'1'},
     {'1', '6'},
     {'3'},
     {'11', '3'},
     {'11', '3', '6'},
     {'3', '6'},
     {'11'},
     {'11', '6'},
     {'6'}]



当然可以使用其他设置来查看运行结果，比如降低置信度级别。
