# 1. 基于贝叶斯决策理论的分类方法

朴素贝叶斯：

优点：在数据较少的情况下仍然有效，可以处理多类别问题

缺点：对于输入数据的准备方式较为敏感

适用数据类型：标称型数据

# 2. 使用条件概率来分类

应用贝叶斯准则，可以得到：

$$ p(c_i|x,y) = \frac {p(x,y|c_i)p(c_i)}{p(x,y)} $$

于是可以定义贝叶斯分类准则为：

（1）如果$ P(c_1|x,y) > P(c_2|x,y)$，那么属于类别$c_1$

（2）如果$ P(c_1|x,y) < P(c_2|x,y)$，那么属于类别$c_2$

# 3. 使用朴素贝叶斯进行文档分类

朴素贝叶斯的一般过程：

（1）收集数据：可以使用任何方法

（2）准备数据：需要数值型或布尔型数据

（3）分析数据：有大量特征时，绘制特征作用不大，此时使用直方图效果更好

（4）训练算法：计算不同的独立特征的条件概率

（5）测试算法：计算错误率

（6）使用算法：可以在任意的分类场景中使用朴素贝叶斯分类器

朴素贝叶斯的两大假设：特征独立且同等重要。

# 4. 使用Python进行文本分类

## 4.1 准备数据：从文本中构建词向量


```python
def loadDataSet():
    postingList = [ ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid'] ]
    classVec = [0, 1, 0, 1, 0, 1]  #1代表侮辱性文字，0代表正常言论
    return postingList, classVec

def createVocabList(dataSet):
    vocabSet = set()                #创建一个空集
    for document in dataSet:
        vocabSet |= set(document)  #创建两个集合的并集
    return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)     #创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
print(myVocabList)
```

    ['park', 'how', 'take', 'quit', 'please', 'buying', 'flea', 'has', 'not', 'posting', 'dog', 'I', 'problems', 'maybe', 'food', 'dalmation', 'is', 'stop', 'worthless', 'licks', 'him', 'stupid', 'so', 'steak', 'mr', 'love', 'my', 'ate', 'cute', 'to', 'garbage', 'help']
    


```python
print(setOfWords2Vec(myVocabList, listOPosts[0]))
```

    [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    


```python
print(setOfWords2Vec(myVocabList, listOPosts[3]))
```

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    

## 4.2 训练算法：从词向量计算概率

根据条件独立性假设，函数的伪代码如下：

计算每个类别中的文档数目
    
    对每篇训练文档：
        对每个类别：
            如果词条出现在文档中->增加该词条的计数值
            增加所有词条的计数值
        对每个类别：
            对每个词条：
                将该词条的数目除以总词条数目得到条件概率
        返回每个类别的条件概率


```python
from numpy import *

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = zeros(numWords)                          #初始化概率
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:                  #向量相加
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p0Vect = p0Num / p0Denom                  #对每个元素做除法
    p1Vect = p1Num / p1Denom
    return p0Vect, p1Vect, pAbusive

listOPosts, listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
p0V, p1V, pAb = trainNB0(trainMat, listClasses)
pAb
```




    0.5




```python
p0V
```




    array([0.        , 0.04166667, 0.        , 0.        , 0.04166667,
           0.        , 0.04166667, 0.04166667, 0.        , 0.        ,
           0.04166667, 0.04166667, 0.04166667, 0.        , 0.        ,
           0.04166667, 0.04166667, 0.04166667, 0.        , 0.04166667,
           0.08333333, 0.        , 0.04166667, 0.04166667, 0.04166667,
           0.04166667, 0.125     , 0.04166667, 0.04166667, 0.04166667,
           0.        , 0.04166667])




```python
p1V
```




    array([0.05263158, 0.        , 0.05263158, 0.05263158, 0.        ,
           0.05263158, 0.        , 0.        , 0.05263158, 0.05263158,
           0.10526316, 0.        , 0.        , 0.05263158, 0.05263158,
           0.        , 0.        , 0.05263158, 0.10526316, 0.        ,
           0.05263158, 0.15789474, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.05263158,
           0.05263158, 0.        ])



## 4.3 测试算法：根据现实情况修改分类器

（1）要计算多个概率的乘积以获得文档属于某个类别的概率，如果其中一个概率值为0，那么最后概率值为0

为了降低这种影响，可以将所有词的出现数初始化为1，并将分母初始化为2

（2）另一个遇到的问题是下溢出，这是由于太多很小的数相乘造成的，于是可以取对数计算


```python
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
    p0 = vec2Classify@p0Vec + log(1.0 - pClass1)        #元素相乘
    p1 = vec2Classify@p1Vec + log(pClass1)
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
    
testingNB()
```

    ['love', 'my', 'dalmation'] classified as:  0
    ['stupid', 'garbage'] classified as:  1
    

## 4.4 准备数据：文档词袋模型

词集模型：每个词的出现与否作为一个特征

词袋模型：对每个词统计出现次数


```python
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
```

# 5. 示例：使用朴素贝叶斯过滤垃圾邮件

使用朴素贝叶斯对电子邮件进行分类：

（1）收集数据：提供文本文件

（2）准备数据：将文本文件解析成词条向量

（3）分析数据：检查词条确保解析的正确性

（4）训练算法：使用之前的trainNB0()函数

（5）测试算法：使用classifyNB()，并且构建一个新的测试函数来计算文档集的错误率

（6）使用算法：构建一个完整的程序对一组文档进行分类，将错分的文档输出到屏幕上

## 5.1 准备数据：切分文本


```python
mySent = 'This book is the best book on Python or M.L. I have ever laid eyes upon.'
print(mySent.split())
```

    ['This', 'book', 'is', 'the', 'best', 'book', 'on', 'Python', 'or', 'M.L.', 'I', 'have', 'ever', 'laid', 'eyes', 'upon.']
    


```python
import re

regEx = re.compile('\\W+')
listOfTokens = regEx.split(mySent)
print(listOfTokens)
```

    ['This', 'book', 'is', 'the', 'best', 'book', 'on', 'Python', 'or', 'M', 'L', 'I', 'have', 'ever', 'laid', 'eyes', 'upon', '']
    


```python
print([tok.lower() for tok in listOfTokens if len(tok) > 0])
```

    ['this', 'book', 'is', 'the', 'best', 'book', 'on', 'python', 'or', 'm', 'l', 'i', 'have', 'ever', 'laid', 'eyes', 'upon']
    


```python
with open('email/ham/6.txt') as fp:
    emailText = fp.read()
    listOfTokens = regEx.split(emailText)
    print(listOfTokens)
```

    ['Hello', 'Since', 'you', 'are', 'an', 'owner', 'of', 'at', 'least', 'one', 'Google', 'Groups', 'group', 'that', 'uses', 'the', 'customized', 'welcome', 'message', 'pages', 'or', 'files', 'we', 'are', 'writing', 'to', 'inform', 'you', 'that', 'we', 'will', 'no', 'longer', 'be', 'supporting', 'these', 'features', 'starting', 'February', '2011', 'We', 'made', 'this', 'decision', 'so', 'that', 'we', 'can', 'focus', 'on', 'improving', 'the', 'core', 'functionalities', 'of', 'Google', 'Groups', 'mailing', 'lists', 'and', 'forum', 'discussions', 'Instead', 'of', 'these', 'features', 'we', 'encourage', 'you', 'to', 'use', 'products', 'that', 'are', 'designed', 'specifically', 'for', 'file', 'storage', 'and', 'page', 'creation', 'such', 'as', 'Google', 'Docs', 'and', 'Google', 'Sites', 'For', 'example', 'you', 'can', 'easily', 'create', 'your', 'pages', 'on', 'Google', 'Sites', 'and', 'share', 'the', 'site', 'http', 'www', 'google', 'com', 'support', 'sites', 'bin', 'answer', 'py', 'hl', 'en', 'answer', '174623', 'with', 'the', 'members', 'of', 'your', 'group', 'You', 'can', 'also', 'store', 'your', 'files', 'on', 'the', 'site', 'by', 'attaching', 'files', 'to', 'pages', 'http', 'www', 'google', 'com', 'support', 'sites', 'bin', 'answer', 'py', 'hl', 'en', 'answer', '90563', 'on', 'the', 'site', 'If', 'you抮e', 'just', 'looking', 'for', 'a', 'place', 'to', 'upload', 'your', 'files', 'so', 'that', 'your', 'group', 'members', 'can', 'download', 'them', 'we', 'suggest', 'you', 'try', 'Google', 'Docs', 'You', 'can', 'upload', 'files', 'http', 'docs', 'google', 'com', 'support', 'bin', 'answer', 'py', 'hl', 'en', 'answer', '50092', 'and', 'share', 'access', 'with', 'either', 'a', 'group', 'http', 'docs', 'google', 'com', 'support', 'bin', 'answer', 'py', 'hl', 'en', 'answer', '66343', 'or', 'an', 'individual', 'http', 'docs', 'google', 'com', 'support', 'bin', 'answer', 'py', 'hl', 'en', 'answer', '86152', 'assigning', 'either', 'edit', 'or', 'download', 'only', 'access', 'to', 'the', 'files', 'you', 'have', 'received', 'this', 'mandatory', 'email', 'service', 'announcement', 'to', 'update', 'you', 'about', 'important', 'changes', 'to', 'Google', 'Groups', '']
    

## 5.2 测试算法：使用朴素贝叶斯进行交叉验证


```python
def textParse(bigString):
    listOfTokens = re.split(r'\W+', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):                           #导入并解析文本文件
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
    for i in range(10):                                         #随机构建训练集
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:                                        #对测试集分类
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
    
spamTest()
```

    classification error  ['yeah', 'ready', 'may', 'not', 'here', 'because', 'jar', 'jar', 'has', 'plane', 'tickets', 'germany', 'for']
    the error rate is:  0.1
    


```python
spamTest()
```

    classification error  ['scifinance', 'now', 'automatically', 'generates', 'gpu', 'enabled', 'pricing', 'risk', 'model', 'source', 'code', 'that', 'runs', '300x', 'faster', 'than', 'serial', 'code', 'using', 'new', 'nvidia', 'fermi', 'class', 'tesla', 'series', 'gpu', 'scifinance', 'derivatives', 'pricing', 'and', 'risk', 'model', 'development', 'tool', 'that', 'automatically', 'generates', 'and', 'gpu', 'enabled', 'source', 'code', 'from', 'concise', 'high', 'level', 'model', 'specifications', 'parallel', 'computing', 'cuda', 'programming', 'expertise', 'required', 'scifinance', 'automatic', 'gpu', 'enabled', 'monte', 'carlo', 'pricing', 'model', 'source', 'code', 'generation', 'capabilities', 'have', 'been', 'significantly', 'extended', 'the', 'latest', 'release', 'this', 'includes']
    the error rate is:  0.1
    


```python
spamTest()
```

    classification error  ['benoit', 'mandelbrot', '1924', '2010', 'benoit', 'mandelbrot', '1924', '2010', 'wilmott', 'team', 'benoit', 'mandelbrot', 'the', 'mathematician', 'the', 'father', 'fractal', 'mathematics', 'and', 'advocate', 'more', 'sophisticated', 'modelling', 'quantitative', 'finance', 'died', '14th', 'october', '2010', 'aged', 'wilmott', 'magazine', 'has', 'often', 'featured', 'mandelbrot', 'his', 'ideas', 'and', 'the', 'work', 'others', 'inspired', 'his', 'fundamental', 'insights', 'you', 'must', 'logged', 'view', 'these', 'articles', 'from', 'past', 'issues', 'wilmott', 'magazine']
    classification error  ['yay', 'you', 'both', 'doing', 'fine', 'working', 'mba', 'design', 'strategy', 'cca', 'top', 'art', 'school', 'new', 'program', 'focusing', 'more', 'right', 'brained', 'creative', 'and', 'strategic', 'approach', 'management', 'the', 'way', 'done', 'today']
    the error rate is:  0.2
    
