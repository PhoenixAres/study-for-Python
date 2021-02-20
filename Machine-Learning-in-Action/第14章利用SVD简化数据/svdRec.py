from numpy import *
from numpy import linalg as la

def loadExData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]

def loadExData1():
    return [[0, 0, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0]]


def eulidSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))

def pearsSim(inA, inB):
    return 1.0 if len(inA) < 3 else 0.5 + 0.5 * corrcoef(inA, inB, rowvar=0)[0][1]

def cosSim(inA, inB):
    num = float(dot(inA.T, inB))
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)

def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overLap = nonzero(logical_and(dataMat[:, item] > 0, dataMat[:, j] > 0))[0]  #寻找两个用户都评级的物品
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    return ratSimTotal / simTotal

def recommend(dataMat, user, N = 3, simMeas = cosSim, estMethod = standEst):
    unratedItems = nonzero(array([dataMat[user, :]]) == 0)[1]         #寻找未评级的物品
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda x:x[1], reverse=True)[:N]  #寻找前N个未评级的物品

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simToTal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    Sig4 = eye(4) * Sigma[:4]          #建立对角矩阵
    xformedItems = dot(dataMat.T, dot(U[:, :4], la.pinv(Sig4)))   #构建转换后的物品
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simToTal += similarity
        ratSimTotal += similarity * userRating
    if simToTal == 0:
        return 0
    return ratSimTotal / simToTal

def printMat(inMat, thresh = 0.8):
    for i in range(32):
        for k in range(32):
            print(1 if float(inMat[i, k]) > thresh else 0, end='')
        print()

def imgCompress(numSV = 3, thresh = 0.8):
    my1 = []
    with open('0_5.txt') as f:
        for line in f.readlines():
            newRow = []
            for i in range(32):
                newRow.append(int(line[i]))
            my1.append(newRow)
    myMat = array(my1)
    print('****original matrix******')
    printMat(myMat, thresh)
    U, Sigma, VT = la.svd(myMat)
    SigRecon = zeros((numSV, numSV))
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    reconMat = dot(U[:, :numSV], dot(SigRecon, VT[:numSV, :]))
    print('****reconstructed matrix using %d singular values******' % numSV)
    printMat(reconMat, thresh)

# U, Sigma, VT = la.svd([[1, 1], [7, 7]])
# print(U)
# print(Sigma)
# print(VT)

# Data = loadExData()
# U, Sigma, VT = la.svd(Data)
# print(Sigma)
# Sig3 = array([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
# print(dot(U[:, :3], dot(Sig3, VT[:3, :])))

# myMat = array(loadExData1())
# print(eulidSim(myMat[:, 0], myMat[:, 4]))
# print(eulidSim(myMat[:, 0], myMat[:, 0]))
# print(cosSim(myMat[:, 0], myMat[:, 4]))
# print(cosSim(myMat[:, 0], myMat[:, 0]))
# print(pearsSim(myMat[:, 0], myMat[:, 4]))
# print(pearsSim(myMat[:, 0], myMat[:, 0]))

# myMat = array(loadExData1())
# myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
# myMat[3, 3] = 2
# print(myMat)
# print(recommend(myMat, 2))
# print(recommend(myMat, 2, simMeas=eulidSim))
# print(recommend(myMat, 2, simMeas=pearsSim))

# U, Sigma, VT = la.svd(array(loadExData2()))
# print(Sigma)
# Sig2 = Sigma**2
# print(sum(Sig2))
# print(sum(Sig2) * 0.9)
# print(sum(Sig2[:2]))
# print(sum(Sig2[:3]))

# myMat = array(loadExData2())
# print(recommend(myMat, 1, estMethod=svdEst))
# print(recommend(myMat, 1, estMethod=svdEst, simMeas=pearsSim))

# imgCompress(2)
