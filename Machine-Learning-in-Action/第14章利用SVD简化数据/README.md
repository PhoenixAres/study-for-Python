# 1. SVD的应用

奇异值分解(Singular Value Decomposition，SVD)。通过使用SVD，我们能够用小得多的数据集来表示原始数据集，实际上是去除了噪声和冗余信息。

可以将SVD看作是从有噪声数据中抽取相关特征。

奇异值分解：

优点：简化数据，去除噪声，提高算法的结果

缺点：数据的转换可能难以理解

适用数据类型：数值型数据

## 1.1 隐性语义索引

最早的SVD应用之一就是信息检索，我们称利用SVD的方法为隐性语义索引(Latent Semantic Indexing，LSI)或隐性语义分析(Latent Semantic Analysis，LSA)。

在LSI中，一个矩阵是由文档和词语组成，当我们在该矩阵上应用SVD时，就会构建出多个奇异值，这些奇异值代表了文档中的概念或主题，这一特点可以用于更高效的文档搜索。

在词语拼写错误时，只基于词语存在与否的简单搜索方法会遇到问题，简单搜索的另一个问题是同义词所在文档并不会匹配上。

如果我们从上千篇相似的文档中抽取出概念，那么同义词就会映射为同一概念。

## 1.2 推荐系统

SVD的另一个应用就是推荐系统，简单版本的推荐系统能够计算项或者人之间的相似度，更先进的方法则是利用SVD从数据中构建一个主题空间，然后在该空间下计算其相似度，考虑如下矩阵：

|    | 鳗鱼饭  | 日式炸鸡排 | 寿司饭 | 烤牛肉 |手撕猪肉 |
|  :----:  | :----:  | :----: | :----: | :----: | :----: |
| Ed  | 0 | 0 | 0 | 2 | 2 |
| Peter  | 0 | 0 | 0 | 3 | 3 |
| Tracy  | 0 | 0 | 0 | 1 | 1 |
| Fan  | 1 | 1 | 1 | 0 | 0 |
| Ming  | 2 | 2 | 2 | 0 | 0 |
| Pachi  | 5 | 5 | 5 | 0 | 0 |
| Jocelyn  | 1 | 1 | 1 | 0 | 0 |

上述矩阵由餐馆的菜和品菜师对这些菜的意见构成，品菜师可以采用1到5之间的任一个整数来对菜评级，如果品菜师没有尝过这道菜，则评级为0。

对上述矩阵进行SVD处理，会得到两个奇异值，因此，就会仿佛有两个概念或主题与此数据集相关联。

通过观察该矩阵，我们发现可以用二维的信息来表示，基于每个组的共同特征，我们可以将这二维命名为美式BBQ和日式食品。

而使用SVD会得到$U$和$V^T$两个矩阵，$V^T$矩阵会将用户映射到BBQ/日式食品空间去，而$U$矩阵会将餐馆的菜映射到BBQ/日式食品空间去。

推荐引擎中可能也会有噪声数据，比如某个人对这些菜的评级就可能存在噪声，并且推荐系统也可以将数据抽取为这些基本主题，基于这些主题，推荐系统就能取得比原始数据更好的推荐效果。

# 2. 矩阵分解

SVD将原始数据集矩阵$Data$分解成三个矩阵$U$、$\Sigma$和$V^T$，原始矩阵m行n列，那么$U$、$\Sigma$和$V^T$就分别是m行m列、m行n列和n行n列：

$$ Data_{m \times n} = U_{m \times m} \Sigma_{m \times n}V^T_{n \times n}$$

上述分解中的$\Sigma$矩阵，该矩阵只有对角元素，其他元素均为0，另一个惯例是，$\Sigma$的对角元素是从大到小排列的。

这些对角元素就是奇异值(Singular Value)，它们对应了原始数据集矩阵$Data$的奇异值。

注：奇异值和特征值是有关系的，这里的奇异值就是矩阵$Data * Data^T$特征值的平方根。

在科学和工程中，一直存在着这样一个普遍事实：在某个奇异值的数目（r个）之后，其他的奇异值都置为0（即去除噪声）。

这就意味着数据集中仅有r个重要特征，而其余特征则都是噪声或冗余特征。

# 3. 利用Python实现SVD


```python
from numpy import *
from numpy import linalg as la

U, Sigma, VT = la.svd([[1, 1], [7, 7]])
U
```




    array([[-0.14142136, -0.98994949],
           [-0.98994949,  0.14142136]])




```python
Sigma
```




    array([10.,  0.])




```python
VT
```




    array([[-0.70710678, -0.70710678],
           [-0.70710678,  0.70710678]])



值得一提的是，矩阵Sigma返回的是行向量，这是为了节省空间，实际上它代表着一个对角矩阵，它存储的是主对角线上的元素。


```python
def loadExData():
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]

Data = loadExData()
U, Sigma, VT = la.svd(Data)
Sigma
```




    array([9.72140007e+00, 5.29397912e+00, 6.84226362e-01, 1.50962387e-15,
           1.15387192e-31])



可以看出，前3个数值比其他值大了很多，而其他值可以看成0，因此可以将最后两个值去掉。

于是原始数据集可以用如下结果近似代替：

$$ Data_{m \times n} \approx U_{m \times 3} \Sigma_{3 \times 3} V^T_{3 \times n} $$


```python
Sig3 = array([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
dot(U[:, :3], dot(Sig3, VT[:3, :]))
```




    array([[ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
            -2.87390037e-16, -3.02026766e-16],
           [ 2.00000000e+00,  2.00000000e+00,  2.00000000e+00,
             4.51836922e-16,  4.22563463e-16],
           [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00,
             3.11293483e-16,  2.96765174e-16],
           [ 5.00000000e+00,  5.00000000e+00,  5.00000000e+00,
            -1.43231976e-16, -2.16090362e-16],
           [ 1.00000000e+00,  1.00000000e+00, -7.28747323e-16,
             2.00000000e+00,  2.00000000e+00],
           [-6.81513666e-17,  9.90189482e-16, -1.27298132e-15,
             3.00000000e+00,  3.00000000e+00],
           [-8.87643769e-17,  2.70954800e-16, -2.90278033e-16,
             1.00000000e+00,  1.00000000e+00]])



我们是如何知道仅需保留前3个奇异值的呢？确定要保留的奇异值数目有很多启发式的策略。

其中一个典型的做法是保留矩阵中90%的能量信息，为了计算总能量信息，可以将奇异值的平方和累加到总值的90%为止。

另一个启发式策略是，当矩阵有上万的奇异值时，那么就保留前面的2000或3000个。

# 4. 基于协同过滤的推荐引擎

## 4.1 相似度计算

协同过滤中的方法：利用用户对它们的意见来计算相似度，而不是利用专家所给出的重要属性。例如，下图给出了由一些用户给出的对菜品的评级信息：

|    | 鳗鱼饭  | 日式炸鸡排 | 寿司饭 | 烤牛肉 |手撕猪肉 |
|  :----:  | :----:  | :----: | :----: | :----: | :----: |
| Jim  | 2 | 0 | 0 | 4 | 4 |
| John  | 5 | 5 | 5 | 3 | 3 |
| Sally  | 2 | 4 | 2 | 1 | 2 |

我们计算一下手撕猪肉和烤牛肉之间的相似度，使用欧式距离：

$$ \sqrt {(4-4)^2 + (3-3)^2 + (2-1)^2} = 1 $$

而手撕猪肉和鳗鱼饭的欧式距离为：

$$ \sqrt {(4-2)^2 + (3-5)^2 + (2-2)^2} = 2.83 $$

在该数据中，手撕猪肉和烤牛肉的距离小于手撕猪肉和鳗鱼饭的距离，因此手撕猪肉与烤牛肉比与鳗鱼饭更为相似。

我们希望相似度值在0到1之间变化，并且物品对越相似，它们的相似度值也就越大，于是可以用如下公式计算相似度：

$$ 相似度 = \frac {1} {1 + 距离} $$

第二种计算距离的方法是皮尔逊相关系数(Pearson correlation)，它度量的是两个向量之间的相似度。

该方法相对于欧式距离的优势在于：它对用户评级的量级并不敏感。比如某个狂躁者对所有物品的评分都是5分，而另一个忧郁者对所有物品的评分都是1分，皮尔逊相关系数会认为这两个向量是相等的。

在NumPy中，皮尔逊相关系数的计算由函数corrcoef()进行，其取值范围为-1到1，为将其归一化到0到1之间，使用如下公式：

$$ 0.5 + 0.5*corrcoef() $$

第三种常用的距离计算方法是余弦相似度(cosine similarity)，其计算的是两个向量夹角的余弦值，如果夹角为90度，则相似度为0，如果两个向量方向相同，则相似度为1.0。

同皮尔逊相关系数，余弦相似度的取值范围也在-1到1之间，将其归一化：

$$ cos\theta = \frac {A \cdot B} {||A|| \cdot ||B||} $$

其中，$||A||$和$||B||$表示$A$、$B$的2范数，你可以定义向量的任一范数，省略时，默认为2范数，例如，向量(4, 2, 2)的2范数为：

$$\sqrt {4^2 + 2^2 + 2^2} $$

NumPy中提供了计算范数的方法linalg.norm()。


```python
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

myMat = array(loadExData1())
print(eulidSim(myMat[:, 0], myMat[:, 4]))
print(eulidSim(myMat[:, 0], myMat[:, 0]))
```

    0.12973190755680383
    1.0
    


```python
print(cosSim(myMat[:, 0], myMat[:, 4]))
print(cosSim(myMat[:, 0], myMat[:, 0]))
```

    0.5
    1.0
    


```python
print(pearsSim(myMat[:, 0], myMat[:, 4]))
print(pearsSim(myMat[:, 0], myMat[:, 0]))
```

    0.20596538173840334
    1.0
    

## 4.2 基于物品的相似度还是基于用户的相似度？

两个餐馆菜肴之间的距离，称为基于物品(item-based)的相似度，计算用户距离的方法称为基于用户(user-based)的相似度。

基于物品的相似度计算的时间会随物品数量的增加而增加，基于用户的相似度计算的时间会随用户数量的增加而增加，选择哪一个要根据实际情况。

对于大部分产品导向的推荐引擎而言，用户数量往往大于物品的数量，即购买商品的用户会对于出售的商品种类。

## 4.3 推荐引擎的评价

可以通过交叉测试的方法，具体做法是：将某些已知的评分值去掉，然后对它们进行预测，最后计算预测值和真实值之间的差异。

通常用于推荐引擎评价的指标是最小均方根误差(Root Mean Squared Error，RMSE)，它首先计算均方误差的平均值然后取其平方根。

如果一个评级在1星到5星这个范围内，而我们的RMSE为1.0，则意味着我们的预测值和用户给出的真实评价相差了一个星级。

# 5. 示例：餐馆菜肴推荐引擎

## 5.1 推荐未尝过的菜肴

推荐系统的工作过程是：给定一个用户，系统会为此用户返回N个最好的推荐菜，为了实现这一点，我们需要做到：

（1）寻找用户没有评级的菜肴，即在用户-物品矩阵中的0值

（2）在用户没有评级的所有物品中，对每个物品预计一个可能的评级分数。

（3）对这些物品的评分从高到低进行排序，返回前N个物品


```python
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

myMat = array(loadExData1())
myMat[0, 1] = myMat[0, 0] = myMat[1, 0] = myMat[2, 0] = 4
myMat[3, 3] = 2
myMat
```




    array([[4, 4, 0, 2, 2],
           [4, 0, 0, 3, 3],
           [4, 0, 0, 1, 1],
           [1, 1, 1, 2, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0]])




```python
recommend(myMat, 2)
```

    the 1 and 0 similarity is: 1.000000
    the 1 and 3 similarity is: 0.928746
    the 1 and 4 similarity is: 1.000000
    the 2 and 0 similarity is: 1.000000
    the 2 and 3 similarity is: 1.000000
    the 2 and 4 similarity is: 0.000000
    




    [(2, 2.5), (1, 2.0243290220056256)]




```python
recommend(myMat, 2, simMeas=eulidSim)
```

    the 1 and 0 similarity is: 1.000000
    the 1 and 3 similarity is: 0.309017
    the 1 and 4 similarity is: 0.333333
    the 2 and 0 similarity is: 1.000000
    the 2 and 3 similarity is: 0.500000
    the 2 and 4 similarity is: 0.000000
    




    [(2, 3.0), (1, 2.8266504712098603)]




```python
recommend(myMat, 2, simMeas=pearsSim)
```

    the 1 and 0 similarity is: 1.000000
    the 1 and 3 similarity is: 1.000000
    the 1 and 4 similarity is: 1.000000
    the 2 and 0 similarity is: 1.000000
    the 2 and 3 similarity is: 1.000000
    the 2 and 4 similarity is: 0.000000
    




    [(2, 2.5), (1, 2.0)]



## 5.2 利用SVD提高推荐效果


```python
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

U, Sigma, VT = la.svd(array(loadExData2()))
Sigma
```




    array([15.77075346, 11.40670395, 11.03044558,  4.84639758,  3.09292055,
            2.58097379,  1.00413543,  0.72817072,  0.43800353,  0.22082113,
            0.07367823])



计算总能量：


```python
Sig2 = Sigma**2
sum(Sig2)
```




    541.9999999999995



计算总能量的90%：


```python
sum(Sig2) * 0.9
```




    487.7999999999996



计算前两个元素和前三个元素包含的能量：


```python
print(sum(Sig2[:2]))
print(sum(Sig2[:3]))
```

    378.8295595113579
    500.50028912757926
    

可以发现，前三个元素包含的能量高于总能量的90%，于是，我们可以将一个11维的矩阵转换成一个3维的矩阵。


```python
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

myMat = array(loadExData2())
recommend(myMat, 1, estMethod=svdEst)
```

    the 0 and 3 similarity is: 0.490950
    the 0 and 5 similarity is: 0.484274
    the 0 and 10 similarity is: 0.512755
    the 1 and 3 similarity is: 0.491294
    the 1 and 5 similarity is: 0.481516
    the 1 and 10 similarity is: 0.509709
    the 2 and 3 similarity is: 0.491573
    the 2 and 5 similarity is: 0.482346
    the 2 and 10 similarity is: 0.510584
    the 4 and 3 similarity is: 0.450495
    the 4 and 5 similarity is: 0.506795
    the 4 and 10 similarity is: 0.512896
    the 6 and 3 similarity is: 0.743699
    the 6 and 5 similarity is: 0.468366
    the 6 and 10 similarity is: 0.439465
    the 7 and 3 similarity is: 0.482175
    the 7 and 5 similarity is: 0.494716
    the 7 and 10 similarity is: 0.524970
    the 8 and 3 similarity is: 0.491307
    the 8 and 5 similarity is: 0.491228
    the 8 and 10 similarity is: 0.520290
    the 9 and 3 similarity is: 0.522379
    the 9 and 5 similarity is: 0.496130
    the 9 and 10 similarity is: 0.493617
    




    [(4, 3.344714938469228), (7, 3.329402072452697), (9, 3.328100876390069)]



尝试另一种相似度计算方式：


```python
recommend(myMat, 1, estMethod=svdEst, simMeas=pearsSim)
```

    the 0 and 3 similarity is: 0.341942
    the 0 and 5 similarity is: 0.124132
    the 0 and 10 similarity is: 0.116698
    the 1 and 3 similarity is: 0.345560
    the 1 and 5 similarity is: 0.126456
    the 1 and 10 similarity is: 0.118892
    the 2 and 3 similarity is: 0.345149
    the 2 and 5 similarity is: 0.126190
    the 2 and 10 similarity is: 0.118640
    the 4 and 3 similarity is: 0.450126
    the 4 and 5 similarity is: 0.528504
    the 4 and 10 similarity is: 0.544647
    the 6 and 3 similarity is: 0.923822
    the 6 and 5 similarity is: 0.724840
    the 6 and 10 similarity is: 0.710896
    the 7 and 3 similarity is: 0.319482
    the 7 and 5 similarity is: 0.118324
    the 7 and 10 similarity is: 0.113370
    the 8 and 3 similarity is: 0.334910
    the 8 and 5 similarity is: 0.119673
    the 8 and 10 similarity is: 0.112497
    the 9 and 3 similarity is: 0.566918
    the 9 and 5 similarity is: 0.590049
    the 9 and 10 similarity is: 0.602380
    




    [(4, 3.346952186702173), (9, 3.3353796573274708), (6, 3.3071930278130375)]



## 5.3 构建推荐引擎面临的挑战

一是SVD运行效率问题，SVD分解可以在程序调入时运行一次，甚至离线运行，不必每次评分时都做SVD分解。

二是矩阵的表示方法，在实际系统中，将会有许多0，也许可以只存储非0元素来节省内存和计算开销？

三是资源浪费，我们计算多个物品的相似度得分，这些记录可以在多个用户之间重复使用，普遍的做法是离线计算并保存相似度得分。

四是如何在缺乏数据时给出好的推荐，这称为冷启动(cold-start)问题。

这个问题的另一说法是，用户不喜欢无效的物品，而用户喜欢的物品又无效，也就是说，在协同过滤的场景下，由于新物品到来时由于缺乏所有用户对其的喜好信息，因此无法判断每个用户对其的喜好，而无法判断某个用户对其的喜好，也就无法利用该商品。

冷启动的解决方案：将推荐看成搜索问题，我们需要使用物品的属性，或将属性作为相似度计算所需要的数据，这被称为基于内容(content-based)的推荐。

# 6. 示例：基于SVD的图像压缩

使用SVD对数据降维，从而实现图像的压缩：


```python
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
    
imgCompress(2)
```

    ****original matrix******
    00000000000000110000000000000000
    00000000000011111100000000000000
    00000000000111111110000000000000
    00000000001111111111000000000000
    00000000111111111111100000000000
    00000001111111111111110000000000
    00000000111111111111111000000000
    00000000111111100001111100000000
    00000001111111000001111100000000
    00000011111100000000111100000000
    00000011111100000000111110000000
    00000011111100000000011110000000
    00000011111100000000011110000000
    00000001111110000000001111000000
    00000011111110000000001111000000
    00000011111100000000001111000000
    00000001111100000000001111000000
    00000011111100000000001111000000
    00000001111100000000001111000000
    00000001111100000000011111000000
    00000000111110000000001111100000
    00000000111110000000001111100000
    00000000111110000000001111100000
    00000000111110000000011111000000
    00000000111110000000111111000000
    00000000111111000001111110000000
    00000000011111111111111110000000
    00000000001111111111111110000000
    00000000001111111111111110000000
    00000000000111111111111000000000
    00000000000011111111110000000000
    00000000000000111111000000000000
    ****reconstructed matrix using 2 singular values******
    00000000000000000000000000000000
    00000000000000000000000000000000
    00000000000001111100000000000000
    00000000000011111111000000000000
    00000000000111111111100000000000
    00000000001111111111110000000000
    00000000001111111111110000000000
    00000000011110000000001000000000
    00000000111100000000001100000000
    00000000111100000000001110000000
    00000000111100000000001110000000
    00000000111100000000001110000000
    00000000111100000000001110000000
    00000000111100000000001110000000
    00000000111100000000001110000000
    00000000111100000000001110000000
    00000000111100000000001110000000
    00000000111100000000001110000000
    00000000111100000000001110000000
    00000000111100000000001110000000
    00000000111100000000001110000000
    00000000111100000000001110000000
    00000000111100000000001110000000
    00000000111100000000001110000000
    00000000111100000000001110000000
    00000000111100000000001100000000
    00000000001111111111111000000000
    00000000001111111111110000000000
    00000000001111111111110000000000
    00000000000011111111100000000000
    00000000000011111111000000000000
    00000000000000000000000000000000
    

可以看到，只需要两个奇异值就能相当精确地对图像实现重构。

$U$和$V^T$都是$32 \times 2$的矩阵，有两个奇异值，因此总数字的数目是64+64+2=130，和原数目1024相比，获得了几乎10倍的压缩比。
