# 这个是我在网上寻找的模板，一边阅读一边查询各个函数的意义，发现与课堂所学的内容是相符合的，下面是我对knn算法以及这份代码自己的理解
# 1.也就是需要先将训练集都制作成一个多维向量的点集，也就是这个py文件的datatoarray、get_labels、train_data三个函数的功能
# 2.其次就是将要测试的图片转化成跟训练集一样的多维向量，然后通过计算距离并寻找在最近的K个点中出现最多的是哪一种点（假设为n）
# 3.选出这个最多的，就可以推测测试数据就是n，也就是这个py文件中的knn函数
# 4.最后通过建立循环来重复2、3功能，直到整个测试集都测试完成之后，对比准确率如何，来判断这个knn算法的实战性能，也就是主函数段所做的事情

import numpy as np
import operator
import os
# KNN算法
def knn(k, testdata, traindata, labels):  # (k,测试集,训练集,分类)
    traindatasize = traindata.shape[0]  # 行数
    # 测试样本（一维）和训练集样本数不一样，因此需要将测试集样本数扩展成和训练集一样多（因为需要求这个样本和所有训练样本之间的距离）
    # 从行方向扩展 tile(a,(size,1))
    dif = np.tile(testdata, (traindatasize, 1)) - traindata


    # 计算距离
    sqdif = dif ** 2
    sumsqdif = sqdif.sum(axis=1)#axis=1表示计算每一行的sum
    distance = sumsqdif ** 0.5

    sortdistance = distance.argsort()  # 从小到大排列，结果返回元素位置
    count = {}
    for i in range(k):
        vote = labels[sortdistance[i]]  #取出前k个距离最近的点
        # 统计每一类列样本的数量
        # get 表示如果字典count中已经存在了vote键，那么就返回其对应的值，如果不存在则返回0（相当于创建此时vote对应的键）
        count[vote] = count.get(vote, 0) + 1
    # 将count由字典转变为列表，key是告诉函数选取每个元组中的值（而非键）作为排序依据，也就是采用相似样例的数量做依据，reverse=True 说明是降序排序
    sortcount = sorted(count.items(), key=operator.itemgetter(1), reverse=True)
    # 取包含样本数量最多的那一类别
    return sortcount[0][0]


# 加载数据,将文件转化为数组形式
def datatoarray(filename):
    arr = []
    fh = open(filename)
    for i in range(32):
        thisline = fh.readline()
        for j in range(32):
            arr.append(int(thisline[j]))
    return arr


# 获取文件的lable
def get_labels(filename):
    label = int(filename.split('_')[0])
    return label


# 建立训练数据
def train_data():
    labels = []
    trainlist = os.listdir('traindata/')
    num = len(trainlist)
    # 长度1024（列），每一行存储一个文件
    # 用一个数组存储所有训练数据，行:文件总数，列：1024
    trainarr = np.zeros((num, 1024))
    for i in range(num):
        thisfile = trainlist[i]
        labels.append(get_labels(thisfile))
        trainarr[i, :] = datatoarray("traindata/" + thisfile)
    return trainarr, labels


# 用测试数据调用KNN算法进行测试
def datatest():
    a = []  # 准确结果
    b = []  # 预测结果
    traindata, labels = train_data()
    testlist = os.listdir('testdata/')
    fh = open('result_knn.csv', 'a')
    for test in testlist:
        testfile = 'testdata/' + test
        testdata = datatoarray(testfile)
        result = knn(3, testdata, traindata, labels)
        # 将预测结果存在文本中
        fh.write(test + '-----------' + str(result) + '\n')
        a.append(int(test.split('_')[0]))
        b.append(int(result))
    fh.close()
    return a, b


if __name__ == '__main__':
    a, b = datatest()
    num = 0
    for i in range(len(a)):
        if (a[i] == b[i]):
            num += 1
        else:
            print("预测失误：", a[i], "预测为", b[i])
    print("测试样本数为：", len(a))
    print("预测成功数为：", num)
    print("模型准确率为：", num / len(a))
