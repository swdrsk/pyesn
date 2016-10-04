import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyESN import ESN
import argparse
import sys

import pdb

datadir = "../data/"
resultdir = "../result/"
paramdir = "../params/"
rng = np.random.RandomState(42)


def run():
    usage = 'Usage: python [--verbose] [--cat <file>] [--help]'.format(__file__)
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('-if','--inputfile')
    parser.add_argument('-of','--outputfile')
    parser.add_argument('-p','--parameter')
    parser.add_argument('-d','--drawflag',default=False,action='store_true')
    args = parser.parse_args()
    inputfile = datadir + args.inputfile
    outputfile = resultdir + args.outputfile
    parameter = paramdir + args.parameter
    drawflag = args.drawflag
    run_pyESN(inputfile,outputfile,parameter,drawflag)


def load_parameter(filename):
    data = pd.read_csv(filename)
    param_dict = {}
    for key in data.keys():
        param_dict[key] = data[key][0]
    return param_dict


def run_pyESN(inputfile,outputfile,parameter,drawflag=False):
    inputdata = pd.read_csv(inputfile)
    params = load_parameter(parameter)
    N = inputdata.shape[0]
    keysnum = inputdata.shape[1]
    traintest_cutoff = int(np.ceil(0.7*N))
    n_input = params['n_inputs']
    n_output = params['n_outputs']
    if n_input+n_output != keysnum:
        print('Invalid input dimention')
        sys.exit()
    train_ctrl,test_ctrl = [],[]
    train_output,test_output = [],[]
    for key in inputdata.keys()[:n_input]:
        if not train_ctrl:
            train_ctrl = np.array(inputdata[key][:traintest_cutoff])
            test_ctrl = np.array(inputdata[key][traintest_cutoff:])
        else:
            train_ctrl = np.vstack([train_ctrl,np.array(inputdata[key][:traintest_cutoff])])
            test_ctrl = np.vstack([test_ctrl, np.array(inputdata[key][traintest_cutoff:])])
    for key in inputdata.keys()[n_input:]:
        if not train_output:
            train_output = np.array(inputdata[key][:traintest_cutoff])
            test_output = np.array(inputdata[key][traintest_cutoff:])
        else:
            train_output = np.vstack([train_output,np.array(inputdata[key][:traintest_cutoff])])
            test_output = np.vstack([test_output, np.array(inputdata[key][traintest_cutoff:])])

    #train_ctrl,train_output = control[:traintest_cutoff],output[:traintest_cutoff]
    #test_ctrl, test_output  = control[traintest_cutoff:],output[traintest_cutoff:]

    esn = ESN(**params)
    pred_train = esn.fit(train_ctrl,train_output)
    pred_test = esn.predict(test_ctrl)
    #pd.DataFrame(pred_test).to_csv(outputfile)

    if drawflag:
        window_tr = range(len(train_output)/4,len(train_output)/4+2000)
        plt.subplot(211)
        plt.plot(train_ctrl[window_tr],label='control')
        plt.plot(train_output[window_tr],label='target')
        plt.plot(pred_train[window_tr],label='model')
        plt.legend(fontsize='x-small')
        plt.title('training (excerpt)')
        plt.ylim([-0.1,1.1])
        
        window_test = range(400)
        plt.subplot(212)
        plt.plot(test_ctrl[window_test],label='control')
        plt.plot(test_output[window_test],label='target')
        plt.plot(pred_test[window_test],label='model')
        plt.legend(fontsize='x-small')
        plt.title('test (excerpt)')
        plt.ylim([-0.1,1.1])

        plt.show()

if __name__=="__main__":
    run()
