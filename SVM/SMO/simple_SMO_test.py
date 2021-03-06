# -*- coding:UTF-8 -*-
#实例测试SMO的简单版本
#SMO简单版本在一些细节处有所忽略，不过思路比较清晰，对结果没有太大影响，不过在迭代速度方面有优化空间


import svmMLiA
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


dataArr, labelArr= svmMLiA.loadDataSet('testSet.txt')
# smoSimple(dataMatIn, classLabels, C, toler, maxIter)
b, alphas=svmMLiA.smoSimple(dataArr, labelArr, 0.6, 0.001, 40)
print 'alphas:\n', alphas
print 'b:\n', b

#只有支持向量对应的alpha的值不是0
print alphas[alphas>0]

#找出哪些数据点是支持向量
for i in range(100):
    if alphas[i]>0.0:
        print dataArr[i], labelArr[i]

#绘制支持向量的情况-----------------
xcord0 = []
ycord0 = []
xcord1 = []
ycord1 = []
markers =[]
colors =[]
fr = open('testSet.txt')  #this file was generated by 2normalGen.py
for line in fr.readlines():
    lineSplit = line.strip().split('\t')
    xPt = float(lineSplit[0])
    yPt = float(lineSplit[1])
    label = int(lineSplit[2])
    if (label == -1):
        xcord0.append(xPt)
        ycord0.append(yPt)
    else:
        xcord1.append(xPt)
        ycord1.append(yPt)

fr.close()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xcord0,ycord0, marker='s', s=90)
ax.scatter(xcord1,ycord1, marker='o', s=50, c='red')
plt.title('Support Vectors Circled')
circle = Circle((4.6581910000000004, 3.507396), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)  #根据前面求出的支持向量来绘图
ax.add_patch(circle)
circle = Circle((3.4570959999999999, -0.082215999999999997), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
ax.add_patch(circle)
circle = Circle((6.0805730000000002, 0.41888599999999998), 0.5, facecolor='none', edgecolor=(0,0.8,0.8), linewidth=3, alpha=0.5)
ax.add_patch(circle)

#求分割平面
w=svmMLiA.calcWs(alphas, dataArr, labelArr)  #求w
print 'w: ', w
# w0=w[0], w1=w[1]
# x = arange(-2.0, 12.0, 0.1)
# y = (-w0*x - b)/w1  #超平面
# ax.plot(x,y)
# ax.axis([-2,12,-8,6])
# plt.show()
