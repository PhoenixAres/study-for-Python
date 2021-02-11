from numpy import *

def loadDataSet(filename):
    dataMat = []
    labelMat = []
    with open(filename) as fr:
        for line in fr.readlines():
            lineArr = line.strip().split('\t')
            dataMat.append([float(lineArr[0]), float(lineArr[1])])
            labelMat.append([float(lineArr[2])])
    return dataMat, labelMat

def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j

def clipAlpha(aj, H, L):
    aj = min(aj, H)
    aj = max(aj, L)
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    dataMatrix = array(dataMatIn)
    labelMat = array(classLabels)
    b = 0
    m, n = shape(dataMatrix)
    alphas = zeros((m, 1))
    iter = 0
    while iter < maxIter:
        alphaPairsChanged = 0
        for i in range(m):
            fXi = dot((alphas * labelMat).T, dot(dataMatrix, dataMatrix[i, :])) + b
            Ei = fXi - labelMat[i]
            if (labelMat[i] * Ei < -toler and alphas[i] < C) or (labelMat[i] * Ei > toler and alphas[i] > 0):
                                                                 #如果alpha可以更改进入优化过程
                j = selectJrand(i, m)                            #随机选择第二个alpha
                fXj = dot((alphas * labelMat).T, dot(dataMatrix, dataMatrix[j, :])) + b
                Ej = fXj - labelMat[j]
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if labelMat[i] != labelMat[j]:                   #保证alpha在0与C之间
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print('L == H')
                    continue
                eta = 2.0 * dot(dataMatrix[i, :], dataMatrix[j, :]) - \
                            dot(dataMatrix[i, :], dataMatrix[i, :]) - \
                            dot(dataMatrix[j, :], dataMatrix[j, :])
                if eta >= 0:
                    print('eta >= 0')
                    continue
                alphas[j] -= labelMat[j] * (Ei - Ej) / eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if abs(alphas[j] - alphaJold) < 0.00001:
                    print('j not moving enough')
                    continue
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])   #对i进行修改，修改量与j相同，但方向相反
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dot(dataMatrix[i, :], dataMatrix[i, :]) - \
                              labelMat[j] * (alphas[j] - alphaJold) * dot(dataMatrix[i, :], dataMatrix[j, :])
                b2 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dot(dataMatrix[i, :], dataMatrix[j, :]) - \
                              labelMat[j] * (alphas[j] - alphaJold) * dot(dataMatrix[j, :], dataMatrix[j, :])
                if 0 < alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] < C:               #设置常数项
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alphaPairsChanged += 1
                print('iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
        if alphaPairsChanged == 0:
            iter += 1
        else:
            iter = 0
        print('iteration number: %d' % iter)
    return b, alphas

class optStruct:
    def __init__(self, dataMatIn, classLabels, C, toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = zeros((self.m, 1))
        self.b = 0
        self.eCache = zeros((self.m, 2))        #误差缓存

def calcEk(oS, k):
    fXk = dot((oS.alphas * oS.labelMat).T, dot(oS.X, oS.X[k, :])) + oS.b
    Ek = fXk - oS.labelMat[k]
    return Ek

def selectJ(i, oS, Ei):           #内循环中的启发式方法
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]
    validEcacheList = nonzero(oS.eCache[:, 0])[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:
            if k == i:
                continue
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:          #选择具有最大步长的j
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1, Ek]

def innerL(i, oS):
    Ei = calcEk(oS, i)
    if (oS.labelMat[i] * Ei < -oS.tol and oS.alphas[i] < oS.C) or (oS.labelMat[i] * Ei > oS.tol and oS.alphas[i] > 0):
        j, Ej = selectJ(i, oS, Ei)            # 第二个alpha选择中的启发式方法
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L == H:
            print('L == H')
            return 0
        eta = 2.0 * dot(oS.X[i, :], oS.X[j, :]) - \
                    dot(oS.X[i, :], oS.X[i, :]) - \
                    dot(oS.X[j, :], oS.X[j, :])
        if eta >= 0:
            print('eta >= 0')
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clipAlpha(oS.alphas[j], H, L)
        updateEk(oS, j)                           #更新误差缓存
        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print('j not moving enough')
            return 0
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        updateEk(oS, i)                           #更新误差缓存
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * dot(oS.X[i, :], oS.X[i, :]) - \
                         oS.labelMat[j] * (oS.alphas[j] - alphaJold) * dot(oS.X[i, :], oS.X[j, :])
        b2 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * dot(oS.X[i, :], oS.X[j, :]) - \
                         oS.labelMat[j] * (oS.alphas[j] - alphaJold) * dot(oS.X[j, :], oS.X[j, :])
        if 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j] < oS.C:  # 设置常数项
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter):
    oS = optStruct(array(dataMatIn), array(classLabels), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxIter) and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        if entireSet:                 #遍历所有值
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print('fullSet, iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1
        else:                         #遍历非边界值
            nonBoundIs = nonzero((oS.alphas > 0) * (oS.alphas < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print('non-bound, iter: %d i: %d, pairs changed %d' % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
        print('iteration number: %d' % iter)
    return oS.b, oS.alphas

def calcWs(alphas, dataArr, classLabels):
    X = array(dataArr)
    labelMat = array(classLabels)
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += alphas[i] * labelMat[i] * array([X[i, :]]).T
    return w
