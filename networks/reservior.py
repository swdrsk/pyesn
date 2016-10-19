#coding:utf-8
import networkx as nx
import numpy as np


class Network:
    def __init__(self,n=200,
                 leaky_rate=0.2):
        self.N = n
        self.W = self.make_weight()
        self.Win = self.make_input_weight()
        self.u = self.init_reservoir()
        self.a = leaky_rate
        self.reservoir = self.u
        
    def make_weight(self,radius=0.95):
        W = np.random.random(self.N,self.N)
        return W

    def make_input_weight(self):
        W = np.random.random(self.N)
        return W

    def init_reservoir(self):
        return np.zeros(self.N)
    
    def __update(self,inp):
        self.u = -(1-self.a) * self.u \
                 self.a * np.tanh(self.Win*inp+self.W*self.u)
        np.vstack(self.reservoir,self.u)

    def run(self):
        pass
    
    def draw_graph(self):
        pass

    def get_reservoir_state(self):
        return 
