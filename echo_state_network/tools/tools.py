#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt

def draw_output(ouput):
    plt.plot(range(len(output)),output)

def attractor(output):
    plt.plot(output[:-1],output[1:])

