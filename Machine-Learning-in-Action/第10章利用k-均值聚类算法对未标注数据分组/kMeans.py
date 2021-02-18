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
    for j in range(n):              #构建簇质心
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k)
    return centroids

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

# datMat = array(loadDataSet('testSet.txt'))
# print(min(datMat[:, 0]))
# print(min(datMat[:, 1]))
# print(max(datMat[:, 1]))
# print(max(datMat[:, 0]))
# print(randCent(datMat, 2))
# print(distEclud(datMat[0], datMat[1]))

# datMat = array(loadDataSet('testSet.txt'))
# myCentroids, clustAssing = kMeans(datMat, 4)

# datMat3 = array(loadDataSet('testSet2.txt'))
# centList, myNewAssments = biKmeans(datMat3, 3)
# print(centList)
