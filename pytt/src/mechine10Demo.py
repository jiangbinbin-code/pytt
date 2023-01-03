from numpy import *

'''
K-均值是发现给定数据集的k个簇的算法。簇个数k是用户给定的，每一个簇通过其质心（centroid），即簇中所有点的中心来描述。
K-均值算法的工作流程是这样的。首先，随机确定k个初始点作为质心。然后将数据集中的每个点分配到一个簇中，
具体来讲，为每个点找距其最近的质心，并将其分配给该质心所对应的簇。这一步完成后，每个簇的质心更新为该簇所有点的平均值。


'''


def loadDataSet(fileName):  # general function to parse tab -delimited floats
    dataMat = []  # assume last column is target value
    fr = open(fileName)  # 打开文件
    for line in fr.readlines():  # 读取每一行数据
        curLine = line.strip().split('\t')  # 去掉首位的空格，并且以‘\t’分割数据
        fltLine = list(map(float, curLine))  # map all elements to float()使用map把函数转化为float类型
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):  # 计算向量A和向量B之间的距离
    return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB)


# 随机生成中心
def randCent(dataSet, k):
    n = shape(dataSet)[1]  # 得到数据列的数量，即数据的维度
    centroids = mat(zeros((k, n)))  # create centroid mat创建一个由k个质心组成的零矩阵
    for j in range(n):  # create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:, j])  # 得到第j个维度的最小值
        rangeJ = float(max(dataSet[:, j]) - minJ)  # 得到第j个维度的取值范围
        centroids[:, j] = mat(minJ + rangeJ * random.rand(k, 1))  # 生成k*1的随机数（在数据该维度的取值范围内）
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):  # 输入变量有4个，数据集，聚类中心的个数，计算距离函数和随机生成聚类中心函数
    m = shape(dataSet)[0]  # 得到数据的个数
    clusterAssment = mat(zeros((m, 2)))  # create mat to assign data points生成m*2的零矩阵
    # to a centroid, also holds SE of each point
    centroids = createCent(dataSet, k)  # 随机创建k个中心
    clusterChanged = True  # 中心是否改变的标志
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # for each data point assign it to the closest centroid循环每个数据点
            minDist = inf
            minIndex = -1
            for j in range(k):  # 循环每个聚类中心
                distJI = distMeas(centroids[j, :], dataSet[i, :])  # 计算每个数据点到聚类中心的距离
                if distJI < minDist:  # 选择距离最小的聚类中心
                    minDist = distJI  # 赋值
                    minIndex = j  # 最小的聚类中心的标志
            if clusterAssment[i, 0] != minIndex: clusterChanged = True  # 该数据的最近的聚类中心不会发生变化，当所有的数据最近的聚类中心不会变化时，停止迭代
            clusterAssment[i, :] = minIndex, minDist ** 2  # 给clusterAssment每行赋值，第一个值是那个聚类中心距离该数据点距离最小，第二个值是最小距离的平方是多少
        print(centroids)  # 输出聚类中心
        for cent in range(k):  # recalculate centroids重新计算聚类中心
            ptsInClust = dataSet[
                nonzero(clusterAssment[:, 0].A == cent)[0]]  # get all the point in this cluster得到属于该聚类中心的所有点
            centroids[cent, :] = mean(ptsInClust, axis=0)  # assign centroid to mean计算平均值
    return centroids, clusterAssment  # 返回聚类中心和数据属于哪个聚类中心的矩阵


from numpy import *
import matplotlib.pyplot as plt

dataMat = mat(loadDataSet('testSet.txt'))  # 载入数据
myCentroids, clustAssing = kMeans(dataMat, 4)  # K-均值算法
# 进行绘图 数据可视化
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataMat[:, 0].tolist(), dataMat[:, 1].tolist(), 20, 15.0 * clustAssing[:, 0].reshape(1, 80).A[0])
ax.scatter(myCentroids[:, 0].tolist(), myCentroids[:, 1].tolist(), marker='x', color='r')
plt.show()
