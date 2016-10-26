#coding:utf-8
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

class Network:
    def __init__(self,
                 n=200,
                 leaky_rate=0.8,
                 sparsity=0.2,
                 radius=0.8,
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
    
    def __update(self, inp):
        u = self.a * self.u + (1-self.a) * np.tanh(self.Win*inp + np.dot(self.W,self.u))
        #sys.stdout.write(str(u)+"\r")
        #sys.stdout.flush()
        return u

    def run(self, inp):
        T = 1000
        if inp is None:
            self.reservoir = np.zeros([T, self.N])
            for t in range(T):
                self.u = self.__update(0)
                self.reservoir[t,:] = self.u

        else:
            self.inputsignal = inp
            self.reservoir = np.zeros([len(inp), self.N])
            for n,i in enumerate(inp):
                self.u = self.__update(i)
                self.reservoir[n, :] = self.u
                sys.stdout.write("%d \r"%n)
                sys.stdout.flush()

    def plot_reservoir(self, cutoff=None, nums=10):
        inp = self.inputsignal
        if cutoff is None:
            cutoff = [0, self.reservoir.shape[0]]
        plt.figure()
        if inp is not None:
            plt.plot(range(cutoff[1]-cutoff[0]), inp[cutoff[0]:cutoff[1]])
        reservoir = zip(*self.reservoir)
        for num in range(nums):
            plt.plot(range(cutoff[1]-cutoff[0]), reservoir[num][cutoff[0]:cutoff[1]])
        plt.show()

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


class ElmanRNN(Network):
    def __init__(self,
                 n=200,
                 leaky_rate=0.8,
                 sparsity=0.2,
                 radius=0.8,
                 random_state=1):
        Network.__init__(n=n,
                         leaky_rate=leaky_rate,
                         sparsity=sparsity,
                         radius=radius,
                         random_state=random_state)
        self.Win = self.init_input_weight()

    def init_input_weight(self):
        n = int(self.N/2)


class NoisyNetwok(Network):
    def __init__(self,
                 noisy=0.001,
                 n=200,
                 leaky_rate=0.8,
                 sparsity=0.2,
                 radius=0.8,
                 random_state=1):
        Network.__init__(self,
                         n=n,
                         leaky_rate=leaky_rate,
                         sparsity=sparsity,
                         radius=radius,
                         random_state=random_state)
        self.noisy = noisy

    def __update(self,inp):
        u = self.a * self.u + (1-self.a) * np.tanh(self.Win*inp + np.dot(self.W, self.u)) +\
            self.random_state_.rand(self.N) * self.noisy
        return u


if __name__ == "__main__":
    #nw = Network(n=300, leaky_rate=0, radius=1.6, sparsity=0.9)
    nw = NoisyNetwok(n=1000, leaky_rate=0., radius=0.99, sparsity=0.9)
    #filename = "../data/inpulse.txt"
    filename = "../data/continuous.txt"
    inputsignal = pd.read_csv(filename)["input"]
    #inputsignal = pd.read_csv('../data/sinwave.txt')['output']
    nw.run(inputsignal)
    nw.plot_reservoir(cutoff=[2900, 4000],nums=10)

