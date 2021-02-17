from numpy import *
import re

def loadDataSet():
    postingList = [ ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'] ]
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set()
    for document in dataSet:
        vocabSet |= set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p0Vect = log(p0Num / p0Denom)
    p1Vect = log(p1Num / p1Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p0 = dot(vec2Classify, p0Vec) + log(1.0 - pClass1)
    p1 = dot(vec2Classify, p1Vec) + log(pClass1)
    return 1 if p1 > p0 else 0

def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        with open('email/spam/%d.txt' % i) as fp:
            wordList = textParse(fp.read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1)
        with open('email/ham/%d.txt' % i) as fp:
            wordList = textParse(fp.read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(0)
    vocabList = list(set(fullText))
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print('classification error ', docList[docIndex])
    print('the error rate is: ', float(errorCount) / len(testSet))

#listOPosts, listClasses = loadDataSet()
#myVocabList = createVocabList(listOPosts)
#print(myVocabList)
#print(setOfWords2Vec(myVocabList, listOPosts[0]))
#print(setOfWords2Vec(myVocabList, listOPosts[3]))

#listOPosts, listClasses = loadDataSet()
#myVocabList = createVocabList(listOPosts)
#trainMat = []
#for postinDoc in listOPosts:
#    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
#p0V, p1V, pAb = trainNB0(trainMat, listClasses)
#print(pAb)
#print(p0V)
#print(p1V)

#testingNB()

#mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
#print(mySent.split())
#regEx = re.compile('\\W+')
#listOfTokens = regEx.split(mySent)
#print(listOfTokens)
#print([tok.lower() for tok in listOfTokens if len(tok) > 0])
#with open('email/ham/6.txt') as fp:
#    emailText = fp.read()
#    listOfTokens = regEx.split(emailText)
#    print(listOfTokens)

#spamTest()
