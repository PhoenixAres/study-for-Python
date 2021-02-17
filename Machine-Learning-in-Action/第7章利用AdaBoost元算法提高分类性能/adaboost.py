from numpy import *
import matplotlib.pyplot as plt

def loadSimpData():
    datmat = array([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    classLabels = [[1.0], [1.0], [-1.0], [-1.0], [1.0]]
    return datmat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    dataMatrix = array(dataArr)
    labelMat = array(classLabels)
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = zeros((m, 1))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + float(j) * stepSize
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = ones((m, 1))
                errArr[predictedVals == labelMat] = 0
                weightedError = dot(D.T, errArr)        #计算加权错误率
                print('split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f' \
                       % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt = 40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = ones((m, 1)) / m
    aggClassEst = zeros((m, 1))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print('D:', D.T)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEst:', classEst.T)
        expon = classEst * (-1 * alpha * array(classLabels))
        D = D * exp(expon)
        D = D / D.sum()                                        #为下一次迭代计算D
        aggClassEst += alpha * classEst
        print('aggClassEst:', aggClassEst.T)
        aggErrors = ones((m, 1)) * (sign(aggClassEst) != array(classLabels))
        errorRate = aggErrors.sum() / m                        #错误率累加计算
        print('total error:', errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst

def adaClassify(datToClass, classifierArr):
    dataMatrix = array(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = zeros((m, 1))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return sign(aggClassEst)

def loadDataSet(filename):
    dataMat = []
    labelMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            numFeat = len(line.split('\t'))
            lineArr = []
            curline = line.strip().split('\t')
            for i in range(numFeat - 1):
                lineArr.append(float(curline[i]))
            dataMat.append(lineArr)
            labelMat.append([float(curline[-1])])
    return dataMat, labelMat

def plotROC(predStrengths, classLabels):
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()           #获取排好序的索引
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index][0] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print('the Area Under the Curve is: ', ySum * xStep)

#datMat, classLabels = loadSimpData()
#D = ones((5, 1)) / 5
#print(buildStump(datMat, classLabels, D))

#classifierArray = adaBoostTrainDS(datMat, classLabels, 9)
#print(classifierArray)

#datArr, labelArr = loadSimpData()
#classifierArr = adaBoostTrainDS(datArr, labelArr, 30)
#print(adaClassify([[0, 0]], classifierArr))
#print(adaClassify([[5, 5], [0, 0]], classifierArr))

#dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
#classifierArray = adaBoostTrainDS(dataArr, labelArr, 10)
#testArr, testLabelArr = loadDataSet('horseColicTest2.txt')
#prediction10 = adaClassify(testArr, classifierArray)
#errArr = ones((67, 1))
#print(errArr[prediction10 != array(testLabelArr)].sum())

#dataArr, labelArr = loadDataSet('horseColicTraining2.txt')
#classifierArray, aggClassEst = adaBoostTrainDS(dataArr, labelArr, 10)
#plotROC(aggClassEst.T, labelArr)
