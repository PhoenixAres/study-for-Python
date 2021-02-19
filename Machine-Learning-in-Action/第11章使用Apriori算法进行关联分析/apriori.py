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

# dataSet = loadDataSet()
# print(dataSet)
# C1 = createC1(dataSet)
# print(C1)
# D = list(map(set, dataSet))
# print(D)
# L1, suppData0 = scanD(D, C1, 0.5)
# print(L1)

# dataSet = loadDataSet()
# L, suppData = apriori(dataSet)
# print(L)
# print(aprioriGen(L[0], 2))
# L, suppData = apriori(dataSet, minSupport=0.7)
# print(L)

# dataSet = loadDataSet()
# L, suppData = apriori(dataSet, minSupport=0.5)
# rules = generateRules(L, suppData, minConf=0.7)
# print(rules)
# rules = generateRules(L, suppData, minConf=0.5)
# print(rules)

# mushDatSet = []
# with open('mushroom.dat') as f:
#     mushDatSet = [line.split() for line in f.readlines()]
# L, suppData = apriori(mushDatSet, minSupport=0.3)
# for item in L[1]:
#     if item.intersection('2'):
#         print(item)
#
# for item in L[3]:
#     if item.intersection('2'):
#         print(item)