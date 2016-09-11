#encoding:utf-8

from echo_state_network import ESN

#FILE = "./MackeyGlass_t17.txt"
FILE = "./macky_glass.csv"

if __name__=="__main__":
    outputfile = "./result/sparsity_test-2.csv"
    f = open(outputfile,"w")
    f.write("sparsity,mse\n")
    for sparsity in [0.999,0.995,0.99,0.95,0.9,0.5,0]:
    #    sparsity = 1-step*0.005
    #for variant in [0.001,0.01,0.1,0.5,1,5,10,100]:
        for loop in range(20):
            mse = ESN(FILE,False,sparsity=sparsity)
            f.write("%f,%f\n"%(sparsity,mse))
            print "sparsity: %f,mse: %f"%(sparsity,mse)
    f.close()
    
