#coding:utf-8

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy

import pdb


class networks:
    def network_generate(self):
        n = 30
        k = 5
        p = 0.4
        ws = nx.watts_strogatz_graph(n,k,p)
        return ws

    def complete_graph(self):
        n = 30
        cg = nx.complete_graph(n)
        return cg
        
    def draw_network(self,graph):
        plt.figure()
        nx.draw(graph)
        plt.show()

class neuron_model:
    def __init__(self,graph):
        N = len(graph.nodes())
        network = np.zeros([N,N]) #[[0 for i in range(N)] for j in range(N)]
        for edge in graph.edges():
            network[edge[0]][edge[1]] = 1
            network[edge[1]][edge[0]] = 1
        self.node = N
        self.edge = network
                        
    def kuramoto_model(self,time):
        freq = np.ones(self.node)*10
        #2pi = 2*np.pi
        neurons = np.random.rand(self.node)*2*np.pi #[random.random()*2*np.pi for i in range(self.node)]
        pre_neurons = deepcopy(neurons)
        arrival = []
        step = 0.001 #[1ms]
        t = 0
        while t<time:
            delta = freq + np.sin(np.dot(self.edge,pre_neurons) - neurons)
            neurons = neurons + step*delta
            for idx in range(self.node):
                if neurons[idx]>=2*np.pi:
                    arrival.append([idx,t])
                    neurons[idx] = neurons[idx]-2*np.pi
            pre_neurons = neurons
            t += step            
        return arrival
        

if __name__=="__main__":
    nw = networks()
    #ws = nw.network_generate()
    ws = nw.complete_graph()
    #nw.draw_network(ws)
    nm = neuron_model(ws)
    print nm.kuramoto_model(1)
    #print nm.edge
