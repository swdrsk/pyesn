#coding:utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as linalg
import argparse
from generate_data import MemoryCapacity

import pdb

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f",
                        "--filename",
                        default="MackeyGlass_t17.txt",
                        help = "input-output filename")
    parser.add_argument("-d",
                        "--display",
                        default=True,
                        help="display graphs or not")
    args = parser.parse_args()
    mse = ESN(args.filename,args.display,sparsity=0.1)
    print(mse)

def ESN(filename,display=True,sparsity=0,variant=1):
    trainLen = 2000
    testLen = 2000
    initLen = 100
    #generate thte ESN reservoir
    inSize = 1
    outSize = 1
    resSize = 1000
    #sparsity = 0.2

    emodels = ESN_models(resSize)
    def active_function(Win,W,u,vm,vf):
        return emodels.leaky_esn(Win,W,u,vm,vf)
        #return emodels.kuramoto(Win,W,u,vm,vf)
    
    datadir = "./data/"
    filename = datadir+filename
    data = np.array(pd.read_csv(filename)["output"])

    if display:
        plt.figure(10).clear
        plt.plot(data[0:1000])
        plt.title('a simple of data')

    #np.random.seed(42)
    Win = (np.random.rand(resSize,1+inSize)-0.5) * 1
    W = np.random.rand(resSize,resSize)-0.5
    #W = np.random.normal(0.001,variant,[resSize,resSize])
    #W = np.ones([resSize,resSize])
    #W[0][0] = -1

    if not sparsity==0: W[np.random.random(W.shape)<sparsity]=0

    if sparsity==1:
        W = np.zeros([resSize,resSize])
    else:
        #W *= 0.135
        #if use real rudias value
        #print("computing spectral radius")
        rhoW = max(abs(linalg.eig(W)[0]))
        W *= 1.25/rhoW
    
    X = np.zeros((1+inSize+resSize,trainLen-initLen))
    Yt = data[None,initLen+1:trainLen+1]

    vf = np.zeros((resSize,1)) #firing
    vm = np.zeros((resSize,1)) #Membrain potential
    for t in range(trainLen):
        u = data[t]
        vm,vf = active_function(Win,W,u,vm,vf)
        if t >= initLen:
            X[:,t-initLen] = np.vstack((1,u,vf))[:,0]

    X_T=X.T
    Wout = np.dot(np.dot(Yt,X_T),linalg.pinv(np.dot(X,X_T))) #pseudo inverse

    #Wout = [map(lambda x: x if abs(x)<1000 else 0, Wout[0])]
    #pdb.set_trace()

    Y = np.zeros((outSize,testLen))
    u = data[trainLen]
    for t in range(testLen):
        vm,vf = active_function(Win,W,u,vm,vf)
        y = np.dot(Wout,np.vstack((1,u,vf)))
        Y[:,t] = y
        #generate mode:
        u = y
        #predictive mode:
        #u = data[trainLen+t+1]

    # compute MSE for the first errorLen time steps
    errorLen = 500
    mse = sum(np.square(data[trainLen+1:trainLen+errorLen+1] - Y[0,0:errorLen]))/errorLen
        
    if display:
        #plot Weight distribution
        #plt.figure()
        #plt.hist(Wout.T,bins = 100)
        #plt.hist(W.reshape(resSize*resSize,1),bins=100)
    
        #plot signals
        try:
            plt.figure(1).clear()
            plt.title("Target signal and predicted signal")
            plt.plot(data[trainLen+1:trainLen+testLen+1],'g')
            plt.plot(Y.T,'b')
            plt.legend(['Target signal','Free-running predited signal'])

        
            plt.figure(2).clear()
            plt.title("firing pattern")
            plt.plot(X[0:20,0:400].T)

            #plt.figure(3).clear()
            #plt.title("attractor plot")
            #plt.plot(X[])
        except Exception as e:
            print("Error as: "+str(e))

        print(MemoryCapacity(data[trainLen+1:trainLen+testLen+1],Y.T))
        print(mse)
        plt.show()

    return mse
        
class ESN_models():
    def __init__(self,n):
        self.kuramoto_freq = np.ones((n,1))*8 + np.random.random((n,1))*5

    '''
    activation function:
    return (membrain potential), (firing value)
    '''
    def leaky_esn(self,Win,W,u,vm,vf):
        a = 0.3 #leaky rate
        rst = (1-a)*vm + a*np.tanh(np.dot(Win, np.vstack((1,u))) + np.dot(W,vm))
        return rst,rst

    def kuramoto(self,Win,W,u,vm,vf):
        freq = self.kuramoto_freq
        n = np.size(freq)
        alpha = 0.0001
        for i in range(n):
            freq[i] += np.dot(W[i],np.sin(vm-vm[i]))/n# + np.dot(Win[i],np.sin(vm-u)))/n
        vm = vm + alpha*freq
        #vm = map(lambda x: x-2*np.pi if n>=2*np.pi else x, vm)
        vm[np.where(vm>=2*np.pi)] -= 2*np.pi
        vf = np.sin(vm)
        return vm,vf     
    
    

if __name__=="__main__":
    #generate_data("linear.txt")
    run()
