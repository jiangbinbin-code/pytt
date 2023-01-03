from numpy import *
import matplotlib.pyplot as plt

'''
机器学习实战1  
监督学习
回归算法：两个集合 计算回归系统ws，然后用回归系统ws计算预测值
'''
"""
回归,两个集合 x和y  y为真实值。
y=wx 
"""


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    xTx = xMat.T * xMat
    if linalg.det(xTx) == mat(yArr).T:
        print("this matrix is singular, cannot inverse")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws


def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1  # 得到每行的列数
    xArr = [];
    yArr = []  # 为可能的数据和标签创建空列表
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr


def standRegres(xArr, yArr):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    xTx = xMat.T * xMat  # 根据文中推导的公示计算回归系数
    if linalg.det(xTx) == 0.0:  # 判断行列式是否为0，若为0则不存在逆矩阵
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T * yMat)  # 求出w，.I是对矩阵求逆矩阵
    return ws


def plotRegression():
    xArr, yArr = loadDataSet('D:\\learning\\ex0.txt')  # 加载数据集
    ws = standRegres(xArr, yArr)  # 计算回归系数
    xMat = mat(xArr)  # 创建xMat矩阵
    yMat = mat(yArr)  # 创建yMat矩阵
    xCopy = xMat.copy()  # 深拷贝xMat矩阵
    xCopy.sort(0)  # 排序
    yHat = xCopy * ws  # 计算对应的y值
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.plot(xCopy[:, 1], yHat, c='red')  # 绘制回归曲线
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.flatten().A[0], s=20, c='blue', alpha=.5)  # 绘制样本点
    plt.title('DataSet')  # 绘制title
    plt.xlabel('X')
    plt.show()


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr);
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))  # 创建权重对角矩阵，200行
    for j in range(m):  # 遍历数据集计算每个样本的权重
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))
    xTx = xMat.T * (weights * xMat)  # 按照公式进行计算
    if linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能求逆")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))  # 计算回归系数
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]  # 计算测试数据集大小,200行

    yHat = zeros(m)
    for i in range(m):  # 对每个样本点进行预测
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)  # 第一个参数testArr[i]表示训练集中的每一行，每次拿一个样本进行预测
    return yHat
