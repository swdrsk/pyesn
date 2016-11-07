#coding:utf-8
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from time_correlation import tec,mic

class Network(object):
    def __init__(self,
                 n=200,
                 leaky_rate=0.8,
                 sparsity=0.2,
                 radius=0.9,
                 random_state=1):
        self.N = n
        self.random_state_ = np.random.RandomState(seed=random_state)
        self.a = leaky_rate
        self.sparsity = sparsity
        self.W = self.init_weight(spectral_radius=radius)
        self.Win = self.init_input_weight()
        self.u = self.init_reservoir()
        self.reservoir = self.u
        self.inputsignal = None
        
    def init_weight(self,spectral_radius):
        W = self.random_state_.rand(self.N, self.N) - 0.5
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        W = W*(spectral_radius/radius)
        return W

    def init_input_weight(self):
        Win = self.random_state_.rand(self.N)*2 - 1
        Win[self.random_state_.rand(*Win.shape) < 0.1] = 0
        return Win

    def init_reservoir(self):
        return np.ones(self.N)*0.1
        # return np.zeros(self.N)
    
    def update(self, inp):
        u = self.a * self.u + (1-self.a) * np.tanh(self.Win*inp + np.dot(self.W,self.u))
        return u

    def run(self, inp):
        T = 1000
        if inp is None:
            self.reservoir = np.zeros([T, self.N])
            for t in range(T):
                self.u = self.update(0)
                self.reservoir[t,:] = self.u

        else:
            self.inputsignal = inp
            self.reservoir = np.zeros([len(inp), self.N])
            for n,i in enumerate(inp):
                self.u = self.update(i)
                self.reservoir[n, :] = self.u

    def plot_reservoir(self, cutoff=None, nums=10, figname=""):
        inp = self.inputsignal
        if cutoff is None:
            cutoff = [0, self.reservoir.shape[0]]
        if inp is not None:
            plt.plot(range(cutoff[1]-cutoff[0]), inp[cutoff[0]:cutoff[1]])
        reservoir = zip(*self.reservoir)
        for num in range(nums):
            plt.plot(range(cutoff[1]-cutoff[0]), reservoir[num][cutoff[0]:cutoff[1]])
        plt.ylabel("reservoir state")
        plt.xlabel("time[step]")
        if not figname=="":
            plt.savefig(figname)

    def network_analysis(self):
        connectivity = np.zeros([self.N,self.N])
        for i in range(self.N):
            for j in range(i):
                mutual_info = mic(list(self.reservoir[:,i]),
                                  list(self.reservoir[:.j]))
                connectivity[i,j] = mutual_info
                connectivity[j,i] = mutual_info

    def draw_reservoir(self, filename=None):
        if not filename:
            plt.savefig(filename)
        pass

    def get_reservoir_state(self):
        return self.reservoir
#
# class SmallWorldReservoir(Network):
#     def __init__(self):
#         Network.__init__()


class SigmoidNetwork(Network):
    def __init__(self,*args,**kargs):
        Network.__init__(self,*args,**kargs)

    def update(self,inp):
        def sigmoid(x):
            return 1/(1+np.exp(-5*x))
        u = self.a * self.u + (1-self.a) * sigmoid(self.Win*inp + np.dot(self.W, self.u))
        return u


class NoisyNetwok(Network):
    def __init__(self, noisy=0.001, **kargs):
        Network.__init__(self, **kargs)
        self.noisy = noisy

    def update(self,inp):
        u = self.a * self.u + (1-self.a) * np.tanh(self.Win*inp + np.dot(self.W, self.u)) +\
            self.random_state_.rand(self.N) * self.noisy
        return u


class ReLUNetwork(Network):
    def __init__(self,*args,**kargs):
        Network.__init__(self,*args,**kargs)

    def update(self,inp):
        u = self.a * self.u + (1-self.a) * self.relu(self.Win*inp + np.dot(self.W, self.u))
        return u

    def relu(self,inp):
        return np.maximum(inp, 0)


class NoLUNetwork(Network):
    def __init__(self,*args,**kargs):
        Network.__init__(self,*args,**kargs)

    def update(self,inp):
        u = self.a * self.u + (1-self.a) * (self.Win*inp + np.dot(self.W, self.u))
        return u


class HermansNetwork(NoLUNetwork):
    def __init__(self,*args,**kargs):
        Network.__init__(self,*args,**kargs)

    def init_weight(self,spectral_radius):
        W = self.random_state_.rand(self.N, self.N) - 0.5
        eigenvalues = np.linalg.eigvals(W)
        meaneigenvalue = np.mean(eigenvalues)
        maxeigenvalues = np.max(np.abs(eigenvalues-meaneigenvalue))
        self.W = (spectral_radius*(W-np.abs(meaneigenvalue)*np.identity(self.N))/maxeigenvalues-np.identity(self.N))
        print self.W
        print np.max(np.abs(np.linalg.eigvals(self.W)))


def convergence_time(states, start=0, threshold=1e-4):
    time = states.shape[0]
    res_n = states.shape[1]
    mses = []
    for t in range(start, time):
        mse = np.linalg.norm(states[t, :])/res_n
        mses.append(mse)
        if mse < threshold:
            rst = t
            break
    else:
        rst = time
    return rst-start


def sparsity_experiment():
    ctrl_sps = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    radiuss = [0.8, 0.9, 0.95]
    filename = "../data/inpulse.txt"
    inputsignal = pd.read_csv(filename)["input"]
    count = 20
    for radius in radiuss:
        cts = []
        err = []
        for ctrl_sp in ctrl_sps:
            ct = np.zeros(count)
            for i in range(count):
                nw = NoisyNetwok(n=300, leaky_rate=0.8, radius=radius, sparsity=ctrl_sp, random_state=None)
                nw.run(inputsignal)
                ct[i] = convergence_time(nw.get_reservoir_state(), start=3100)
            cts.append(np.average(ct))
            err.append(np.var(ct))
        plt.plot(ctrl_sps, cts, label="radius:%f"%radius)
        plt.scatter(ctrl_sps, cts, color="r")

    # plt.errorbar(ctrl_sps, cts, yerr=err)
    plt.legend()
    plt.ylim(0,max(cts)+50)
    plt.xlim(0,1)
    plt.title("convergence time")
    plt.ylabel("convergence time [step]")
    plt.xlabel("sparsity")
    plt.savefig('../result/sparsity_experiment.png')
    plt.show()


def variety_radius():
    radiuses = [0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7]
    plt.figure(figsize=[20,20])
    filename = "../data/inpulse.txt"
    inputsignal = pd.read_csv(filename)["input"]
    for i,radius in enumerate(radiuses):
        nw = NoLUNetwork(n=300, leaky_rate=0.9, radius=radius)
        nw.run(inputsignal)
        plt.subplot(330+i+1)
        nw.plot_reservoir(cutoff=[2900, 6000], nums=8)
        plt.title("radius: %f"%radius)
    plt.savefig("../result/inpulse_norelu.png")
    plt.show()


def demo():
    #nw = Network(n=300, leaky_rate=0, radius=1.6, sparsity=0.9)
    nw = NoisyNetwok(n=300, leaky_rate=0.8, radius=0.8, sparsity=0.9)
    # nw = ReLUNetwork(n = 300, leaky_rate=0.9, radius=1.0, random_state=None)
    # nw = SigmoidNetwork(n = 300, leaky_rate=0.9, radius=13, random_state=None)
    # nw = HermansNetwork(n=300, leaky_rate=0.9, radius=0.9)
    filename = "../data/inpulse.txt"
    # filename = "../data/continuous.txt"
    inputsignal = pd.read_csv(filename)["input"]
    inputsignal = pd.read_csv('../data/sinwave.txt')['output']
    inputsignal = pd.read_csv('../data/macky_glass.csv')['output']
    nw.run(inputsignal)
    # print "\rconvergence_time: [%d]"%convergence_time(nw.get_reservoir_state(),start=3100)
    nw.plot_reservoir(cutoff=[2900, 4000],nums=10)
    plt.show()

if __name__=="__main__":
    demo()

