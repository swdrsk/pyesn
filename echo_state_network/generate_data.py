#coding:utf-8
"""
create data
"""
import numpy as np
import matplotlib.pyplot as plt

nums = 400
datadir = "./data/"

def circle_function(x):
    a = [0.5,0.5]
    r = 0.25
    y = 1 if np.linalg.norm(x-a)>r else 0
    return y

def create_data(filename = "circle.csv"):
    f = open(filename,"w")
    f.write("x1,x2,y\n")
    for i in range(nums):
        x = np.random.rand(2)
        y = circle_function(x)
        f.write("%f,%f,%d\n"%(x[0],x[1],y))
    f.close()
def Macky_glass(tau=20,filename = "macky_glass.csv",display=False):
    N = 15000 #N must bigger than 100
    b = 0.1
    c = 0.2
    n_= 10
    tau = 17
    
    f = open(filename,"w")
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

def long_short_wave(filename="longshortwave5.txt",display=False):
    N = 15000
    pi = np.pi
    t = np.array(range(N))
    l = np.sin(t*2*pi/200)
    s = np.sin(t*2*pi/50)
    res = l + s*0.3

    datadir = "./data/"
    filename = datadir+filename
    f = open(filename,"w")
    f.write("output\n")
    for i in res:
        f.write("%f\n"%i)
    f.close()
    
    if display:
        plt.figure()
        plt.plot(range(1000),res[0:1000])
        plt.show()


def sinwave(filename="sinwave.txt",display=False):
    N = 15000
    pi = np.pi
    t = np.array(range(N))
    res = np.sin(t*2*pi/200)

    datadir = "./data/"
    filename = datadir+filename
    f = open(filename,"w")
    f.write("output\n")
    for i in res:
        f.write("%f\n"%i)
    f.close()
    
    if display:
        plt.figure()
        plt.plot(range(1000),res[0:1000])
        plt.show()

    
def multiple_input(filename="multiple_input.txt",display=False):
    """returns a random step function with N changepoints
       and a sine wave signal that changes its frequency at
       each such step, in the limits given by min_ and max_period."""
    # vector of random indices < N, padded with 0 and N at the ends:
    changepoints = np.insert(np.sort(rng.randint(0,N,n_changepoints)),[0,n_changepoints],[0,N]) 
    # list of interval boundaries between which the control sequence should be constant:
    const_intervals = zip(changepoints,np.roll(changepoints,-1))[:-1]
    # populate a control sequence
    frequency_control = np.zeros((N,1))
    for (t0,t1) in const_intervals:
        frequency_control[t0:t1] = rng.rand()
    periods = frequency_control * (max_period - min_period) + max_period
    # run time through a sine, while changing the period length
    frequency_output = np.zeros((N,1))
    z = 0
    for i in range(N):
        z = z + 2 * np.pi / periods[i]
        frequency_output[i] = (np.sin(z) + 1)/2
    return np.hstack([np.ones((N,1)),1-frequency_control]),frequency_output
    """
    N = 15000
    t = np.array(range(N))
    in1 = 
    in2 = 
    
    datadir = "./data/"
    filename = datadir+filename
    f = open(filename,"w")
    """
    
         
def MemoryCapacity(A,B,k=10):
    if len(A)!=len(B):
        print "MC error"
        return 0
    meanA,meanB = A.mean(),B.mean()
    varA,varB = A.var(),B.var()
    for j in range(0,20):
        covs = sum([(A[i-k]-meanA)*(B[i]-meanB) for i in range(k,len(A))])/(len(A)-k)
    return pow(covs,2)/np.sqrt(varA*varB)
    
        
if __name__=="__main__":
    #create_data()
    #Macky_glass(display=False)
    #long_short_wave(display=False)
    #multiple_input(display=True)
    sinwave(display=True)
