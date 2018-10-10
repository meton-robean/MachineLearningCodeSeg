# -*- coding:UTF-8 -*-
'''
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
'''
from numpy import *

def loadSimpData():
    datMat = matrix([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) #get number of fields 
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#对data的指定属性（维度），根据阈值，进行分类 （决策树桩）
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):#just classify the data
    retArray = ones((shape(dataMatrix)[0],1))  #列向量
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    

def buildStump(dataArr, classLabels, D):  #D为样本概率分布（权重）
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T  #label为列向量
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf #init error sum, to +infinity
    for i in range(n):#loop over all dimensions
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps  #分numsteps步来遍历某一属性的所有属性值，找到最佳的分割阈值
        for j in range(-1,int(numSteps)+1):#loop over all range in current dimension     range从[-1, 0,1,...，numstep]
            for inequal in ['lt', 'gt']: #go over less than and greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)  #call stump classify with i, j, lessThan
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr  #calc total error multiplied by D   #计算基于threshVal阈值分类的错误率
                #print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst


#基于决策树桩的adaboost算法训练过程  函数名DS代表单层决策树
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)   #init D to all equal  一开始是平均分布
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)#build Stump
        print "D:",D.T
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))#calc alpha, throw in max(error,eps) to account for error=0
        bestStump['alpha'] = alpha  
        weakClassArr.append(bestStump)                  #store Stump Params in Array
        print "classEst: ",classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst) #exponent for D calc, getting messy
        #print 'expon: ', expon
        D = multiply(D,exp(expon))                              #Calc New D for next iteration
        D = D/D.sum()
        #calc training error of all classifiers, if this is 0 quit for loop early (use break)
        aggClassEst += alpha*classEst
        print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m,1)))
        errorRate = aggErrors.sum()/m
        print "total error: ",errorRate
        if errorRate == 0.0: break   #提前停止条件
    return weakClassArr, aggClassEst


#adaboost的分类器
def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)   #do stuff similar to last aggClassEst in adaBoostTrainDS
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])  #call stump classify
        aggClassEst += classifierArr[i]['alpha']*classEst
        #print aggClassEst
    return sign(aggClassEst)  #预测结果


def plotROC(predStrengths, classLabels):  #绘制ROC曲线 predStrengths，classLabels都是行向量
    import matplotlib.pyplot as plt
    cur = (1.0,1.0) #cursor
    ySum = 0.0 #variable to calculate AUC
    numPosClas = sum(array(classLabels)==1.0)   #正例数目
    yStep = 1/float(numPosClas); xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()#get sorted index, it's reverse
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    #loop through all the values, drawing a line segment at each point
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0; delY = yStep
        else:
            delX = xStep; delY = 0
            ySum += cur[1]
        #draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot( [ cur[0], cur[0]-delX ], [ cur[1], cur[1]-delY ],  c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0,1,0,1])
    plt.show()
    print "the Area Under the Curve is: ",ySum*xStep

##---TEST---##
dataMat, classLabels=loadSimpData()

#---test1---- 训练一个决策树桩（弱分类器）
# D=mat(ones((5,1))/5)
# bestStump,minError,bestClasEst=buildStump(dataMat, classLabels, D)
# print bestStump, minError, bestClasEst


#---test2---- #基于决策树桩的adaboost算法
# print '训练过程：\n'
# classifierArr, aggClassEst=adaBoostTrainDS(dataMat, classLabels, numIt=9)
# print '得到的分类器：\n'
# print classifierArr
#
# print '测试：\n'
# predict= adaClassify([[5,5],[0,0]], classifierArr)
# print predict

#----test3---- #病马数据集训练与测试
# datArr, labelArr=loadDataSet('horseColicTraining2.txt')
# classifierArray, aggClassEst2=adaBoostTrainDS(datArr, labelArr, numIt=10)
# print classifierArray


#---test4--- #
datArr, labelArr=loadDataSet('horseColicTraining2.txt')
classifierArray, aggClassEst3=adaBoostTrainDS(datArr, labelArr, numIt=10)
print classifierArray
print aggClassEst3.T
plotROC(aggClassEst3.T, labelArr)