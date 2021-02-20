from numpy import *
import matplotlib.pyplot as plt

def loadDataSet(fileName, delim = '\t'):
    datArr = []
    with open(fileName) as fr:
        stringArr = [line.strip().split(delim) for line in fr.readlines()]
        datArr = [list(map(float, line)) for line in stringArr]
    return array(datArr)

def pca(dataMat, topNfeat = 9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals                #去平均值
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(array(covMat))
    eigValInd = argsort(eigVals)                   #从小到大对N个值排序
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    redEigVects = eigVects[:, eigValInd]
    lowDDataMat = dot(meanRemoved, redEigVects)    #将数据转换到新空间
    reconMat = dot(lowDDataMat, redEigVects.T) + meanVals
    return lowDDataMat, reconMat

def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i]))[0], i])   #计算所有非NaN的平均值
        datMat[nonzero(isnan(datMat[:, i]))[0], i] = meanVal          #将所有NaN置为平均值
    return datMat

# dataMat = loadDataSet('testSet.txt')
# lowDMat, reconMat = pca(dataMat, 1)
# print(shape(lowDMat))
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(dataMat[:, 0].flatten(), dataMat[:, 1].flatten(), marker='^', s=90)
# ax.scatter(reconMat[:, 0].flatten(), reconMat[:, 1].flatten(), marker='o', s=50, c='red')
# plt.show()

# dataMat = replaceNanWithMean()
# meanVals = mean(dataMat, axis=0)
# meanRemoved = dataMat - meanVals
# covMat = cov(meanRemoved, rowvar=0)
# eigVals, eigVects = linalg.eig(array(covMat))
# print(eigVals)