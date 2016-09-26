import numpy as np
import matplotlib.pyplot as plt
from pyESN import ESN
from scipy import signal
import argparse


import pdb

datadir = "../data/"
resultdir = "../result/"
rng = np.random.RandomState(42)

def run():
    parser = argparse.ArgumentParser()
    f = open(FILE,"w")
    mse,s_mse = run_pyESN()
    f.write("%f,%f,%f\n"%(i,mse,s_mse))
    f.close()

def run_pyESN(drawflag=False):
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
              sparsity = 0.8,
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
