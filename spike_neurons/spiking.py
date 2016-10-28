# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import random

class SpikingNeurons:
    def __init__(self,n,eibalance):
        threshold = -50
        lekylate = -60

        self.n = n
        self.eib = eibalance
        self.spike = np.zeros(n)
        self.potential = np.zeros(n)
        self.spikes = None
        self.threshold = np.ones(n)
        self.leakyrate = np.ones(n)
        self.weight = np.ones([n,n])

    def set_threshold(self,thresholds):
        self.threshold = thresholds
        
    def set_random_threshold(self,mu,var):
        self.threshold = np.random.randn(self.n)*var + mu

    def set_weight(self):
        self.weight = np.array([[random.lognormvariate(0,1.0) for i in range(self.n)] for j in range(self.n)])

    def update(self):
        p_new =
        return self.spike

    def run(self,time):
        self.spikes = np.zeros([self.n, time])
        for t in range(time):
            self.spikes[:,t] = self.update()

    def draw(self):
        def raster(event_times_list, color='k'):
            # cited_from << http://shirokuma88.hatenablog.com/entry/2015/05/02/080410
            plt.gca()
            for ith, trial in enumerate(event_times_list):
                plt.vlines(trial, ith + .5, ith + 1.5, color=color)
            plt.ylim(.5, len(event_times_list) + .5)

        if self.spikes is None:
            print("not recorded spikes, let run(time)")
        else:
            fig = plt.figure()
            raster(self.spikes)
            plt.xlim(0, 1100)
            plt.xlabel('(Sec)')
            plt.ylabel('Cell No.')
            fig.show()