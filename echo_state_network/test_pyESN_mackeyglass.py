import numpy as np
from pyESN import ESN
from matplotlib import pyplot as plt
import pandas as pd
import argparse

datadir = "../data/"
FILE = datadir+"macky_glass.csv"
#FILE = datadir+"sinwave.txt"
#FILE = datadir+"longshortwave3.txt"
#data = np.load('mackey_glass_t17.npy') #  http://minds.jacobs-university.de/mantas/code
data = pd.read_csv(FILE)["output"]
data = np.array(data)

esn = ESN(n_inputs = 1,
          n_outputs = 1,
          leakyrate = 0.2,
          sparsity = 0.001,
          n_reservoir = 300,
          spectral_radius = 0.95,
          random_state=42)

trainlen = 2000
future = 2000
pred_training = esn.fit(np.zeros(trainlen), data[:trainlen])

prediction = esn.predict(np.zeros(future))
print("test error: \n"+str(np.sqrt(np.mean((prediction.flatten() - data[trainlen:trainlen+future])**2))))

plt.figure(figsize=(11,1.5))
plt.plot(range(0,trainlen+future),data[0:trainlen+future],'k',label="target system")
plt.plot(range(trainlen,trainlen+future),prediction,'r', label="free running ESN")
lo,hi = plt.ylim()
plt.plot([trainlen,trainlen],[lo+np.spacing(1),hi-np.spacing(1)],'k:')
plt.legend(loc=(0.61,1.1),fontsize='x-small')
plt.show()
