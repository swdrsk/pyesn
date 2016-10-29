import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pyESN import ESN
import argparse
import sys,os
import sklearn .decomposition

import pdb

datadir = "../data/"
resultdir = "../result/"
paramdir = "../params/"
rng = np.random.RandomState(42)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('-if','--inputfile',help='all [view all exist file]')
    parser.add_argument('-of','--outputfolder')
    parser.add_argument('-p','--parameter',help='all [view all exist file]')
    parser.add_argument('-d','--drawflag',default=False,action='store_true')
    args = parser.parse_args()
    if args.inputfile=="all":
        for _,__,filenames in os.walk(datadir):
            for i in filenames:
                print(i)
        sys.exit()
    if args.parameter=="all":
        for _,__,filenames in  os.walk(paramdir):
            for i in filenames:
                print(i)
        sys.exit()
    inputfile = datadir + args.inputfile
    outputfolder = resultdir + args.outputfolder
    parameter = paramdir + args.parameter
    drawflag = args.drawflag
    if not os.access(outputfolder, os.F_OK):
        os.mkdir(resultdir+outputfolder)
    f = open(outputfolder+'log.txt', 'w')
    f.write(args.inputfile)
    f.write(args.parameter)
    f.close()
    run_pyESN(inputfile,outputfolder,parameter,drawflag)


def load_parameter(filename):
    data = pd.read_csv(filename)
    param_dict = {}
    for key in data.keys():
        param_dict[key] = data[key][0]
    return param_dict


def run_pyESN(inputfile, outputfolder, parameter, drawflag=False):
    inputdata = pd.read_csv(inputfile)
    params = load_parameter(parameter)
    N = inputdata.shape[0]
    keysnum = inputdata.shape[1]
    traintest_cutoff = int(np.ceil(0.8*N))
    print N,traintest_cutoff
    n_input = params['n_inputs']
    n_output = params['n_outputs']
    if n_input+n_output != keysnum:
        print('Invalid input dimention')
        sys.exit()
    train_ctrl, test_ctrl = None, None
    train_output, test_output = None, None
    for key in inputdata.keys()[:n_input]:
        if train_ctrl is None:
            train_ctrl = np.array(inputdata[key][:traintest_cutoff])
            test_ctrl = np.array(inputdata[key][traintest_cutoff:])
        else:
            train_ctrl = np.array(zip(*np.vstack([train_ctrl,np.array(inputdata[key][:traintest_cutoff])])))
            test_ctrl = np.array(zip(*np.vstack([test_ctrl, np.array(inputdata[key][traintest_cutoff:])])))
    for key in inputdata.keys()[n_input:]:
        if train_output is None:
            train_output = np.array(inputdata[key][:traintest_cutoff])
            test_output = np.array(inputdata[key][traintest_cutoff:])
        else:
            train_output = np.array(zip(*np.vstack([train_output, np.array(inputdata[key][:traintest_cutoff])])))
            test_output = np.array(zip(*np.vstack([test_output, np.array(inputdata[key][traintest_cutoff:])])))
    if train_ctrl is None:
        train_ctrl = np.zeros(traintest_cutoff)
        test_ctrl = np.zeros(N-traintest_cutoff)
    if train_output is None:
        train_output = np.zeros(traintest_cutoff)
        test_output = np.zeros(N-traintest_cutoff)
    if params['n_inputs'] == 0:
        params['n_inputs'] = 1

    #esn = ESN(**params)
    esn = ESN(n_inputs = 1,
              n_outputs = 1,
              n_reservoir = 300,
              spectral_radius = 0.99,
              noise = 0,#0.001,
              input_shift = 0,
              # input_scaling = 3,
              # teacher_scaling = 1.35, #1.12,
              # teacher_shift = -0.9, # -0.7,
              out_activation = np.tanh,
              inverse_out_activation = np.arctanh,
              silent = False)

    pred_train = esn.fit(train_ctrl, train_output, inspect=False)

    reservoir_state = esn.get_reservoir_states()
    pca = sklearn.decomposition.PCA(3)
    pca.fit(reservoir_state)
    pcaindex = pca.transform(reservoir_state)
    pred_test = esn.predict(test_ctrl,inspect=True)
    reservoir_state = esn.get_reservoir_states()
    pcaindex_pred = pca.transform(reservoir_state)

    #pd.DataFrame(pred_test).to_csv(outputfile)
    if drawflag:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(*zip(*pcaindex),color="blue")
        ax.plot(*zip(*pcaindex_pred),color="red")
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
    if drawflag:
        window_tr = range(len(train_output)/4,len(train_output)/4+2000)
        if n_output<2:
            plt.figure(figsize=(12, 4))
            plt.subplot(211)
            if n_input!=0:
                plt.plot(train_ctrl[window_tr],label='control')
            if n_output<2:
                plt.plot(train_output[window_tr],label='target')
                plt.plot(pred_train[window_tr],label='model')
                plt.legend(fontsize='x-small')
                plt.title('training (excerpt)')
            window_test = range(2000)
            plt.subplot(212)
            if n_input!=0:
                plt.plot(test_ctrl[window_test],label='control')
            plt.plot(test_output[window_test],label='target')
            plt.plot(pred_test[window_test],label='model')
            plt.legend(fontsize='x-small')
            plt.title('test (excerpt)')
            plt.show()
        else:
            for wo in range(n_output):
                plt.figure(figsize=(12, 4))
                plt.subplot(211)
                if n_input != 0:
                    plt.plot(train_ctrl[window_tr], label='control')
                if n_output < 2:
                    plt.plot(train_output[window_tr, wo], label='target')
                    plt.plot(pred_train[window_tr, wo], label='model')
                    plt.legend(fontsize='x-small')
                    plt.title('training (excerpt)')
                window_test = range(300)
                plt.subplot(212)
                if n_input != 0:
                    plt.plot(test_ctrl[window_test], label='control')
                plt.plot(test_output[window_test, wo], label='target')
                plt.plot(pred_test[window_test, wo], label='model')
                plt.legend(fontsize='x-small')
                plt.title('test (excerpt)')
            plt.show()


if __name__=="__main__":
    run()
