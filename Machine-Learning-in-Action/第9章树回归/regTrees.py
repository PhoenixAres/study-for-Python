from numpy import *

def loadDataSet(filename):
    dataMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            curLine = line.strip().split('\t')
            fltLine = list(map(float, curLine))     #将每行映射成浮点数
            dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1

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
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree

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

# testMat = eye(4)
# print(testMat)
# mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
# print(mat0)
# print(mat1)

# myDat = loadDataSet('ex00.txt')
# myMat = array(myDat)
# print(createTree(myMat))

# myDat1 = loadDataSet('ex0.txt')
# myMat1 = array(myDat1)
# print(createTree(myMat1))

# myDat2 = loadDataSet('ex2.txt')
# myMat2 = array(myDat2)
# print(createTree(myMat2))
# print(createTree(myMat2, ops=(10000, 4)))

# myDat2 = loadDataSet('ex2.txt')
# myMat2 = array(myDat2)
# myTree = createTree(myMat2, ops=(0, 1))
# myDatTest = loadDataSet('ex2test.txt')
# myMat2Test = array(myDatTest)
# print(prune(myTree, myMat2Test))

# myMat2 = array(loadDataSet('exp2.txt'))
# print(createTree(myMat2, modelLeaf, modelErr, (1, 10)))

# trainMat = array(loadDataSet('bikeSpeedVsIq_train.txt'))
# testMat = array(loadDataSet('bikeSpeedVsIq_test.txt'))
# myTree = createTree(trainMat, ops=(1, 20))
# yHat = createForeCast(myTree, testMat[:, 0])
# print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
# myTree = createTree(trainMat, modelLeaf, modelErr, (1, 20))
# yHat = createForeCast(myTree, testMat[:, 0], modelTreeEval)
# print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
# ws, X, Y = linearSolve(trainMat)
# print(ws)
# for i in range(shape(testMat)[0]):
#     yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
# print(corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])