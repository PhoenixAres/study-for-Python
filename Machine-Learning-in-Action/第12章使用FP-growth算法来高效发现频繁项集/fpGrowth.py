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

# rootNode = treeNode('pyramid', 9, None)
# rootNode.children['eye'] = treeNode('eye', 13, None)
# rootNode.disp()
# rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
# rootNode.disp()

# simpDat = loadSimpDat()
# print(simpDat)
# initSet = createInitSet(simpDat)
# print(initSet)
# myFPtree, myHeaderTab = createTree(initSet, 3)
# myFPtree.disp()
# print(findPrefixPath('x', myHeaderTab['x'][1]))
# print(findPrefixPath('z', myHeaderTab['z'][1]))
# print(findPrefixPath('r', myHeaderTab['r'][1]))
#
# freqItems = []
# mineTree(myFPtree, myHeaderTab, 3, set(), freqItems)
# print(freqItems)

# parsedDat = []
# with open('kosarak.dat') as f:
#     parsedDat = [line.split() for line in f.readlines()]
# initSet = createInitSet(parsedDat)
# myFPtree, myHeaderTab = createTree(initSet, 100000)
# myFreqList = []
# mineTree(myFPtree, myHeaderTab, 100000, set(), myFreqList)
# print(len(myFreqList))
# print(myFreqList)