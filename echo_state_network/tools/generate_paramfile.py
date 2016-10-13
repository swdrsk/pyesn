#coding:utf-8
import argparse
import csv

paramdir = "../../params/"
def generate_paramfile(filename,
                       n_inputs,
                       n_outputs,
                       n_reservoir,
                       leakyrate,
                       spectral_radius):
    params = {
        "n_inputs":n_inputs,
        "n_outputs":n_outputs,
        "n_reservoir":n_reservoir,
        "leakyrate":leakyrate,
        "spectral_radius":spectral_radius,
    }
    f=open(paramdir+filename,"w")
    w = csv.writer(f)
    w.writerow(params.keys())
    w.writerow(params.values())
    f.close()
    

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f","--filename",default="parameter.csv")
    parser.add_argument('-in','--n_inputs',default=1)
    parser.add_argument('-out','--n_outputs',default=1)
    parser.add_argument('-N','--n_reservoir',default=300)
    parser.add_argument('-l','--leakyrate',default=0.2)
    parser.add_argument('-r','--radius',default=0.95)
    parser.add_argument('-all','--runall',default=False,action='store_true')
    args = parser.parse_args()
    if args.runall:
        run_all()
    else:
        generate_paramfile(filename=args.filename,
                           n_inputs=args.n_inputs,
                           n_outputs=args.n_outputs,
                           n_reservoir=args.n_reservoir,
                           leakyrate=args.leakyrate,
                           spectral_radius=args.radius)


def run_all():
    n_reservoir = [300, 500, 1000, 1500, 2000]
    leakyrate = [0.2, 0.25, 0.3, 0.35]
    n_in = [0,1]

    for i in n_in:
        for l in leakyrate:
            for r in n_reservoir:
                filename = str(i)+str(1)+"_"+str(r)+'_p'+str(int(l*100))+".csv"
                generate_paramfile(filename=filename,
                                   n_inputs=i,
                                   n_outputs=1,
                                   n_reservoir=r,
                                   leakyrate=l)

if __name__=="__main__":
    run()