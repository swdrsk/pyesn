import numpy as np
import matplotlib.pyplot as plt
from pyESN import ESN
from scipy import signal

import pdb

rng = np.random.RandomState(42)


def run1():
    FILE="./result/sparsity_pyESN.csv"
    f = open(FILE,"w")
    f.write("sparsity,mse,spectro_mse\n")
    for i in [0,0.001,0.005,0.007,0.009,0.01,0.05,0.1,0.5,0.9,0.95,0.99]:
        for step in range(20):
            print("==============sparsity:%f===================="%i)
            mse,s_mse = test_pyESN(sparsity=i)
            f.write("%f,%f,%f\n"%(i,mse,s_mse))
    f.close()

def run():
    test_pyESN(True,0.9)
    
def frequency_generator(N,min_period,max_period,n_changepoints):
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

def draw_spectogram(data):
    plt.specgram(data,Fs=4,NFFT=256,noverlap=150,cmap=plt.cm.hot,detrend=lambda x:(x-0.5))
    plt.gca().autoscale('x')
    plt.ylim([0,0.5])
    plt.ylabel("freq")
    plt.yticks([])
    plt.xlabel("time")
    plt.xticks([])

def evaluate_spectrogram(p_data,t_data,display=False):
    pf, pt, p_Sxx = signal.spectrogram(p_data,fs=4,nfft=256,noverlap=150,detrend=lambda x:(x-0.5))
    tf, tt, t_Sxx = signal.spectrogram(t_data,fs=4,nfft=256,noverlap=150,detrend=lambda x:(x-0.5))
    if display:
        plt.figure()
        plt.pcolormesh(pt,pf,p_Sxx)
        plt.title("predict_signal")
        plt.figure()
        plt.pcolormesh(tt,tf,t_Sxx)
        plt.title("teacher_signal")
    if pf.size!=tf.size or pt.size!=tt.size: return None
    mse = 0
    for psx,tsx in zip(p_Sxx,t_Sxx):
        mse += np.mean((psx-tsx)**2)
    return mse

def test_pyESN(drawflag=False,sparsity=0.8):    
    N = 15000 # signal length
    min_period = 2
    max_period = 10
    n_changepoints = N/200
    frequency_control,frequency_output = frequency_generator(N,min_period,max_period,n_changepoints)

    traintest_cutoff = np.ceil(0.7*N)
    
    train_ctrl,train_output = frequency_control[:traintest_cutoff],frequency_output[:traintest_cutoff]
    test_ctrl, test_output  = frequency_control[traintest_cutoff:],frequency_output[traintest_cutoff:]

    esn = ESN(n_inputs = 2,
              n_outputs = 1,
              n_reservoir = 200,
              spectral_radius = 0.99,
              sparsity = sparsity,
              noise = 0,#0.001,
              input_shift = [0,0],
              input_scaling = [0.01, 3],
              teacher_scaling = 1.12,
              teacher_shift = -0.7,
              out_activation = np.tanh,
              inverse_out_activation = np.arctanh,
              random_state = rng,
              silent = False)

    pred_train = esn.fit(train_ctrl,train_output,inspect=False)

    print("test error:")
    pred_test = esn.predict(test_ctrl)
    mse = np.sqrt(np.mean((pred_test - test_output)**2))
    print(mse)
    
    s_mse = evaluate_spectrogram(pred_test.flatten(),test_output.flatten())
    print("test spectrogram error:")
    print(s_mse)


    if drawflag:
        window_tr = range(len(train_output)/4,len(train_output)/4+2000)
        plt.figure(figsize=(10,1.5))
        plt.plot(train_ctrl[window_tr,1],label='control')
        plt.plot(train_output[window_tr],label='target')
        plt.plot(pred_train[window_tr],label='model')
        plt.legend(fontsize='x-small')
        plt.title('training (excerpt)')
        plt.ylim([-0.1,1.1])
        
        #window_test = range(2000)
        window_test = range(200)
        plt.figure(figsize=(10,1.5))
        plt.plot(test_ctrl[window_test,1],label='control')
        plt.plot(test_output[window_test],label='target')
        plt.plot(pred_test[window_test],label='model')
        plt.legend(fontsize='x-small')
        plt.title('test (excerpt)')
        plt.ylim([-0.1,1.1])

        plt.figure(figsize=(7,1.5))
        draw_spectogram(train_output.flatten())
        plt.title("training: target")
        plt.figure(figsize=(7,1.5))
        draw_spectogram(pred_train.flatten())
        plt.title("training: model")
        
        plt.figure(figsize=(3,1.5))
        draw_spectogram(test_output.flatten())
        plt.title("test: target")
        plt.figure(figsize=(3,1.5))
        draw_spectogram(pred_test.flatten())
        plt.title("test: model")

    
        plt.show()
        
    return mse,s_mse

if __name__=="__main__":
    run()
