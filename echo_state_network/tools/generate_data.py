#coding:utf-8
"""
create data
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse

import pdb

datadir = "../../data/"


def _run():
    function_option = [
        'control_sinwave', 'mackyglass', 'sinwave', 'long_short_wave', 'inpulse'
    ]
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--filename',default=None)
    parser.add_argument('-d','--display',default=False,action='store_true')
    parser.add_argument('-n','--nums',default=15000,type=int)
    parser.add_argument('-func','--function',
                        help=' function name option : %s'%(', '.join(function_option)))
    parser.add_argument('-o', '--option',type=str) # option parser : must be dictva
    args = parser.parse_args()
    if args.filename:
        params = {'filename':args.filename,'num':args.nums,'display':args.display}
    else:
        params = {'num':args.nums,'display':args.display}
    if args.function=='control_sinwave':
        control_sinwave(**params)
    elif args.function=="mackyglass":
        Macky_glass(**params)
    elif args.function=='sinwave':
        sinwave(**params)
    elif args.function=='long_short_wave':
        long_short_wave(**params)
    elif args.function=='inpulse':
        inpulse(**params)
    else:
        print('input correct option as -func [%s]'%(', '.join(function_option)))


def Macky_glass(num,filename = "macky_glass.csv",display=False):
    N = num #N must bigger than 100
    b = 0.1
    c = 0.2
    n_= 10
    tau = 17
    
    f = open(datadir + filename,"w")
    f.write("output\n")

    y = np.zeros(N+tau+100)
    for t in range(N+tau):
        if t<tau:
            y[t] = np.random.random()
        else:
            y[t+1] = y[t] - b*y[t] + c*y[t-tau]/(1+pow(y[t-tau],10))
        if t>100: #remove initial value
            f.write("%f\n"%y[t])
    f.close()

    if display:
        plt.figure()
        plt.plot(range(N+tau-100),y[100:N+tau])
        plt.figure()
        plt.plot(y[100:N+tau-1],y[101:N+tau])        
        plt.show()   


def long_short_wave(num,filename="longshortwave5.txt",display=False):
    N = num
    pi = np.pi
    long_period = 200
    short_period = 50
    t = np.array(range(N))
    l = np.sin(t*2*pi/long_period)
    s = np.sin(t*2*pi/short_period)
    res = l + s*0.3

    filename = datadir+filename
    f = open(filename,"w")
    f.write("output\n")
    for i in list(res):
        f.write("%f\n"%i)
    f.close()

    if display:
        plt.figure()
        plt.plot(range(1000),res[:1000])
        plt.show()


def sinwave(num,filename="sinwave.txt",display=False):
    N = num
    pi = np.pi
    t = np.array(range(N))
    res = np.sin(t*2*pi/200)

    filename = datadir+filename
    f = open(filename,"w")
    f.write("output\n")
    for i in res:
        f.write("%f\n"%i)
    f.close()
    if display:
        plt.figure()
        plt.plot(range(1000),res[:1000])
        plt.show()


def control_sinwave(num,filename="control_sinwave.txt",display=False):
    """returns a random step function with N changepoints
       and a sine wave signal that changes its frequency at
       each such step, in the limits given by min_ and max_period."""
    filename = datadir + filename
    N = num
    min_period = 2
    max_period = 20
    n_changepoints = N/200
    rng = np.random.RandomState(42)

    changepoints = np.insert(np.sort(rng.randint(0,N,n_changepoints)),[0,n_changepoints],[0,N])
    const_intervals = zip(changepoints,np.roll(changepoints,-1))[:-1]
    frequency_control = np.zeros((N,1))
    for (t0,t1) in const_intervals:
        frequency_control[t0:t1] = rng.rand()
    periods = frequency_control * (max_period - min_period) + max_period
    frequency_output = np.zeros((N,1))
    z = 0
    for i in range(N):
        z = z + 2 * np.pi / periods[i]
        frequency_output[i] = (np.sin(z) + 1)/2
    input,output =  map(lambda x:x[0],frequency_control),map(lambda x:x[0],frequency_output)
    pd.DataFrame(zip(input,output),columns=['input','output']).to_csv(filename,index=False)
    if display:
        display_range = 1000
        plt.figure(figsize =(12,1.5))
        plt.plot(range(display_range),input[:display_range])
        plt.plot(range(display_range),output[:display_range]) # output[:display_range]ではうまくいかない。なぜ？
        plt.show()

        
def inpulse(num,filename="inpulse.txt",display=False):
    filename = datadir + filename
    N = num
    prestart_time = int(N/5)
    inpulse_time = min(100,int(N/10))
    output = np.zeros(N)
    output[prestart_time:prestart_time+inpulse_time] = 2
    pd.DataFrame(output,columns=['input']).to_csv(filename,index=False)
    if display:
        plt.figure(figsize=(12,1.5))
        plt.plot(range(N),output)
        plt.show()
    
        
def MemoryCapacity(A,B,k=10):
    if len(A)!=len(B):
        print("MC error")
        return 0
    meanA,meanB = A.mean(),B.mean()
    varA,varB = A.var(),B.var()
    for j in range(0,20):
        covs = sum([(A[i-k]-meanA)*(B[i]-meanB) for i in range(k,len(A))])/(len(A)-k)
    return pow(covs,2)/np.sqrt(varA*varB)
    
        
if __name__=="__main__":
    _run()
