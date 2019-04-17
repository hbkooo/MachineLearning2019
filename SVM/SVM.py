# -*- coding: utf-8 -*-
"""
pip install numpy
pip install matplotlib
pip install scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def get_data_set():
    class_count = 160
    np.random.seed(1111)
    # 生成两类随机点，并连接
    X = np.r_[np.random.randn(class_count,2)-[2,2],np.random.randn(class_count,2)+[2,2]]
    # 生成label
    Y = np.array([-1] * class_count+[1] * class_count)
    
    # 数据随机排序打乱
    choice = np.random.choice(class_count * 2, class_count * 2, replace=False)
    X = X[choice]
    Y = Y[choice]
    
    # 训练数据和测试数据
    x_train = X[0:class_count]
    y_train = Y[0:class_count]
    x_test = X[class_count::]
    y_test = Y[class_count::]
    return x_train,y_train,x_test,y_test

class svmStruct:
    def __init__(self, dataSet, labels, C, toler, kernelOption, max_iter):  
        self.train_x = dataSet                  # each row stands for a sample  
        self.train_y = labels                   # corresponding label  
        self.C = C                              # slack variable  
        self.toler = toler                      # termination condition for iteration  
        self.max_iter = max_iter                # max iter to train
        self.numSamples = dataSet.shape[0]      # number of samples   
        self.w = None   # trained weight
        self.a = 0                              # 斜率
        self.b = 0                              # 截距      
        self.kernelOpt = kernelOption           # 选项
        self.support_vectors = None             # 支持向量

def trainSVM(x_train, y_train, C=10, toler = 0.001, kernelOption = 'linear', max_iter=1000):
    
    SVM = svmStruct(x_train, y_train, C, toler, kernelOption, max_iter)
    
    clf = svm.SVC(C=SVM.C, kernel=SVM.kernelOpt, tol=SVM.toler,max_iter=SVM.max_iter)
    clf.fit(SVM.train_x, SVM.train_y)
    
    SVM.w = clf.coef_[0]                            # 权重系数
    SVM.a = -SVM.w[0]/SVM.w[1]                      # 计算斜率
    SVM.b = -(clf.intercept_[0])/SVM.w[1]           # 计算截距
    SVM.support_vectors = clf.support_vectors_      # 支持向量
    
    return SVM


    
def draw_plot(SVM):
    
    xx=np.linspace(-5,5)#产生-5到5的线性连续值，间隔为1
    yy = SVM.a *xx + SVM.b
    #clf.intercept_[0]是w3.即为公式a1*x1+a2*x2+w3中的w3。(clf.intercept_[0])/w[1]即为直线的截距
     
    #得出支持向量的方程
    b = SVM.support_vectors[0]
    yy_down = SVM.a * xx + (b[1] - SVM.a * b[0])#(b[1]-a*b[0])就是简单的算截距
    b = SVM.support_vectors[-1]
    yy_up = SVM.a * xx + (b[1] - SVM.a * b[0])
     
    #画图
    plt.plot(xx,yy,'g-')
    plt.plot(xx,yy_down,'k--')
    plt.plot(xx,yy_up,'k--')
     
    plt.scatter(SVM.support_vectors[:,0], SVM.support_vectors[:,0], s=80, facecolors='none')
    plt.scatter(SVM.train_x[:,0], SVM.train_x[:,1], c=SVM.train_y, cmap=plt.cm.Paired, marker="o")
     
    plt.title("C = {} ".format(SVM.C))
    plt.axis('tight')
    plt.show()
