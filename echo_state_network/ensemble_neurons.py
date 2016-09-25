import brian
import matplotlib.pyplot as plt
import numpy as np

from brian import *
from brian.hears import *
from numpy.random import randn
from numpy.linalg import norm


def draw_poisson_process():
    deltatime = np.random.poisson(3,30)
    t = 0
    timing = []
    for dt in list(deltatime):
        t += dt
        timing.append(t)
    plt.scatter(timing,list(np.zeros(30)))
    plt.show()


def generate_poisson_process():
    deltatime = np.random.poisson(3,1000)
    t = 0
    timing = []
    for dt in list(deltatime):
        t += dt
        timing.append(t)
    return timing


def draw_firing_moving_averate(windowsize):
    timing = generate_poisson_process()
    _ = firing_moving_averate(timing,windowsize)
    plt.plot(zip(*_)[1])
    plt.show()


def firing_moving_averate(timing,windowsize):
    timing = np.array(timing)
    start = 0
    end = start + windowsize
    moving_averate = []
    while end<timing[-1]:
        firing = np.size(np.where((start <= timing)&(timing <= end)))
        moving_averate.append((start,firing))
        start += 1
        end += 1
    return moving_averate

def echo_state_neuron(input):
    a = 0.3
    m = 0
    output = []
    for i in input:
        m = (1-a)*m + a*np.tanh(m+i)
        output.append(m)
    return output

def draw_output(output):
    plt.plot(range(len(output)),output)
    plt.xlabel("Time")
    plt.ylabel("output")
    plt.show()

if __name__=="__main__":
    #input = np.sin(np.array(range(3000))*0.01)
    input = [0 for i in range(100)] + [1 for i in range(200)] + [0 for i in range(100)] + [1 for i in range(200)]
    output = echo_state_neuron(input)
    draw_output(output)