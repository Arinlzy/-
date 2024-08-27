# -*- coding: utf-8 -*-
import csv
from ssl import CHANNEL_BINDING_TYPES
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

cancers = datasets.load_breast_cancer()
train_X = cancers['data'][:450]
train_Y = cancers['target'][:450]
test_X = cancers['data'][450:]
test_Y = cancers['target'][450:]

class LR:
    def  __init__(self, data, data_test):
        self.m = data.shape[0]
        self.cols = data.shape[1] + 1
        self.w = np.zeros(self.cols)
        self.b = np.ones(self.m).reshape(self.m, 1)
        self.lr = 0.001
        self.train_X = np.hstack([self.b,data])  # 引入偏置项

        self.m_test = data_test.shape[0]
        self.b_test = np.ones(self.m_test).reshape(self.m_test, 1)
        self.test_X = np.hstack([self.b_test,data_test])

    def sigmoid(self,x):
        res = 1/(1 + np.exp(-x))
        return np.clip(res, 1e-8, (1-(1e-8)))

    def stop_stratege(self,loss,loss_update,threshold):
        return  loss-loss_update < threshold

    def Logistic_Regression(self, X, Y, X_test, Y_test, epochs):  # Logistic 
        i = 0
        loss_record = []
        acc_record = []
        acc_test_record = []
        for i in range(epochs):

            ## predict
            # TODO XXX
            z = np.dot(X, self.w)
            y_pred = self.sigmoid(z)

            ## compute loss
            ### -sum((Y * ln(f(x))+(1-Y) * ln(1-f(x))))
            # loss = # TODO XXX
            loss = -np.sum((Y * np.log(y_pred) + (1 - Y)) * np.log(1 - y_pred))

            print(loss)
            loss_record.append(loss)
            
            ## compute gradiant
            # grad = # TODO XXX
            error = y_pred - Y
            grad = np.dot(X.T, error) / len(Y)

            ## update weight
            # self.w = # TODO XXX
            self.w -= self.lr * grad

            ## compute acc in training
            # acc = # TODO XXX
            y_pred_binary = np.where(y_pred >= 0.5, 1, 0)
            acc = np.mean(y_pred_binary == Y)

            acc_record.append(acc)

            ## print every 20 iteration 
            if i % 500 == 1 and i>500:
                self.visulization(i, X, Y)
            
            ## test each epoch
            acc_test = self.test(X_test, Y_test)
            acc_test_record.append(acc_test)

        return  loss_record, acc_record, acc_test_record
    

    def train(self, X, Y, X_test, Y_test, epochs):
        loss_record, acc_record, acc_test_record = self.Logistic_Regression(X, Y, X_test, Y_test, epochs)
        
        return loss_record, acc_record, acc_test_record

    def test(self, X, Y):
        z = np.dot(X, self.w)
        y_pred = self.sigmoid(z)
        # acc_test = # TODO XXX
        y_pred_binary = np.where(y_pred >= 0.5, 1, 0)
        acc_test = np.mean(y_pred_binary == Y)

        return acc_test

    def visulization(self,which_step,X_,Y_):
        x = np.linspace(-10,10)
        y = -(self.w[0] +self.w[1]*x)/self.w[2]
        plt.plot(x,y)

        index_0 = np.where(Y_==0)[0]
        index_1 = np.where(Y_==1)[0]

        plt.plot(X_[index_0, 1], X_[index_0, 2], 'bo', color='blue', label='0')
        plt.plot(X_[index_1, 1], X_[index_1, 2], 'bo', color='orange', label='0')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('The '+str(which_step)+' update figure')
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        plt.show()

def AllNorm(X):

    minVals=np.min(X)
    maxVals=np.max(X)

    ranges=maxVals-minVals
    normDataSet=np.zeros(np.shape(X))
    m=X.shape[0]
    normDataSet=X-np.tile(minVals,(m,1)) # 在行方向重复minVals m次和列方向上重复minVals 1次
    normDataSet=normDataSet/np.tile(ranges,(m,1))
    return normDataSet

def ChannalNorm(X):
    # TODO XXX
    # return normDataSet 
    pass

def plot_history(loss_record, acc_record, acc_test_record, epochs):
        plt.figure(figsize=(10,5))
        plt.subplot(2,2,1)
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.plot(range(epochs), loss_record, label='loss')
        plt.legend()
        plt.subplot(2,2,2)
        plt.xlabel('Epoch')
        plt.ylabel('acc_train')
        plt.plot(range(epochs), acc_record, label='acc_train')
        plt.subplot(2,1,2)
        plt.xlabel('Epoch')
        plt.ylabel('acc_test')
        plt.plot(range(epochs), acc_test_record, label='acc_test', color = 'red')

        #plt.plot(hist['epoch'], hist['acc'], label = 'acc',color = 'red')
        plt.legend()
        plt.show()

if __name__ == "__main__":

    '''
    data = [[-1,-1],[2,-1],[5,3],[-1,6],[-4,-1],[1,4]]
    label = [1,1,1,0,0,0]
    train_X = np.array(data)
    train_Y = np.array(label)
    '''
    # train_X = AllNorm(train_X)
    # test_X = AllNorm(test_X)

    # train_X = ChannalNorm(train_X)
    # test_X = ChannalNorm(test_X)

    Logist = LR(train_X, test_X)   # 实例化

    epochs = 500
    
    loss_record, acc_record, acc_test_record = Logist.train(Logist.train_X, train_Y, Logist.test_X, test_Y, epochs)

plot_history(loss_record, acc_record, acc_test_record, epochs)
