from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(filename):
    dataMat = []
    labelMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            numFeat = len(line.split('\t')) - 1
            lineArr = []
            curline = line.strip().split('\t')
            for i in range(numFeat):
                lineArr.append(float(curline[i]))
            dataMat.append(lineArr)
            labelMat.append([float(curline[-1])])
    return dataMat, labelMat

def standRegres(xArr, yArr):
    xMat = array(xArr)
    yMat = array(yArr)
    xTx = dot(xMat.T, xMat)
    ws = dot(linalg.pinv(xTx), dot(xMat.T, yMat))
    return ws

def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = array(xArr)
    yMat = array(yArr)
    m = shape(xMat)[0]
    weights = eye(m)             #创建对角矩阵
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(dot(diffMat, diffMat) / (-2.0 * k**2))  #权重值大小以指数级衰减
    xTx = dot(xMat.T, dot(weights, xMat))
    ws = dot(linalg.pinv(xTx), dot(xMat.T, dot(weights, yMat)))
    return dot(testPoint, ws)

def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()

def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = dot(xMat.T, xMat)
    denom = xTx + eye(shape(xMat)[1]) * lam
    ws = dot(linalg.pinv(denom), dot(xMat.T, yMat))
    return ws

def ridgeTest(xArr, yArr):
    xMat = array(xArr)
    yMat = array(yArr)
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar       #数据标准化
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i-10))
        wMat[i, :] = ws.T
    return wMat

def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    xMat = array(xArr)
    yMat = array(yArr)
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1))
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = dot(xMat, wsTest)
                rssE = rssError(yMat, yTest)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat




# xArr, yArr = loadDataSet('ex0.txt')
# print(xArr[0:2])
# ws = standRegres(xArr, yArr)
# print(ws)
# xMat = array(xArr)
# yMat = array(yArr)
# yHat = dot(xMat, ws)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xMat[:, 1].flatten(), yMat[:, 0].flatten())
# xCopy = xMat.copy()
# xCopy.sort(0)
# yHat = dot(xCopy, ws)
# ax.plot(xCopy[:, 1], yHat)
# plt.show()
# yHat = dot(xMat, ws)
# print(corrcoef(yHat.T, yMat.T))

# xArr, yArr = loadDataSet('ex0.txt')
# print(yArr[0])
# print(lwlr(xArr[0], xArr, yArr, 1.0))
# print(lwlr(xArr[0], xArr, yArr, 0.001))
# yHat = lwlrTest(xArr, xArr, yArr, 0.003)
# xMat = array(xArr)
# srtInd = xMat[:, 1].argsort(0)
# xSort = xMat[srtInd]
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(xSort[:, 1], yHat[srtInd])
# ax.scatter(xMat[:, 1].flatten(), array(yArr).flatten(), s=2, c='red')
# plt.show()

# abX, abY = loadDataSet('abalone.txt')
# yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
# yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
# yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
# print(rssError(abY[0:99], array([yHat01]).T))
# print(rssError(abY[0:99], array([yHat1]).T))
# print(rssError(abY[0:99], array([yHat10]).T))
#
# yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
# print(rssError(abY[100:199], array([yHat01]).T))
#
# yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
# print(rssError(abY[100:199], array([yHat1]).T))
#
# yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
# print(rssError(abY[100:199], array([yHat10]).T))
#
# ws = standRegres(abX[0:99], abY[0:99])
# yHat = dot(array(abX[100:199]), ws)
# print(rssError(abY[100:199], array([yHat]).T))

# abX, abY = loadDataSet('abalone.txt')
# ridgeWeights = ridgeTest(abX, abY)
# print(ridgeWeights)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(ridgeWeights)
# plt.show()

# xArr, yArr = loadDataSet('abalone.txt')
# stageWise(xArr, yArr, 0.01, 200)
# stageWise(xArr, yArr, 0.001, 5000)
# xMat = array(xArr)
# yMat = array(yArr)
# yMean = mean(yMat, 0)
# yMat = yMat - yMean
# xMeans = mean(xMat, 0)
# xVar = var(xMat, 0)
# xMat = (xMat - xMeans) / xVar
# weights = standRegres(xMat, yMat)
# print(weights.T)
