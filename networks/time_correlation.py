#coding:utf-8
from math import log
import sys
import random
import numpy as np
import pdb

#一次元時系列同士の相互情報量
def mic(xseq , yseq):
   xmax,xmin = max(xseq),min(xseq)
   ymax,ymin = max(yseq),min(yseq)
   N = min(len(xseq) , len(yseq))
   L, Nx , Ny = sys.float_info.max, 1, 1
   if xmax==xmin or ymax==ymin:
      return 0.0
   #-- 赤池情報量基準に基づいて分割数を決める
   for n in xrange(2,N):
      p = dict()
      for (x,y) in zip(xseq,yseq):
         nx = int(n*(x-xmin)/(xmax-xmin))
         ny = int(n*(y-ymin)/(ymax-ymin))
         p[(nx,ny)] = p.get( (nx,ny) , 0) + 1
      AIC = -(sum( [Nxy*log(float(Nxy)/N , 2) for Nxy in p.itervalues()] ) + N*log(n,2)*2) + (n*n-1)
      if AIC < L:
         L = AIC
         Nx = n
         Ny = n
   #-- 相互情報量の推定
   px,py,p = dict(),dict(),dict()
   for (x,y) in zip(xseq,yseq):
       nx = int( Nx*(x-xmin)/(xmax-xmin))
       ny = int( Ny*(y-ymin)/(ymax-ymin))
       px[nx] = px.get(nx,0)+1
       py[ny] = py.get(ny,0)+1
       p[(nx,ny)] = p.get( (nx,ny) , 0) + 1
   return sum([p0*log(N*float(p0)/(px[nx]*py[ny]) , 2) for ((nx,ny) , p0) in p.iteritems()])/(N*N)


def npmic(xseq, yseq):
    """
    mic numpy version
    """
    xmax, xmin = max(xseq), min(xseq)
    ymax, ymin = max(yseq), min(yseq)
    N = min(len(xseq), len(yseq))
    Nx = min(int(log(N, 2)+1),int(log(100,2))+1)
    Ny = Nx
    if xmax == xmin or ymax == ymin:
        return 0.0
    # -- 相互情報量の推定
    nxseq = np.floor(Nx * (xseq -xmin) / (xmax - xmin))
    nyseq = np.floor(Ny * (yseq - ymin)/ (ymax - ymin))
    nxyseq = np.floor(nxseq+nyseq*(Ny+1))
    histxseq = np.histogram(nxseq, bins=range(Nx+2))
    histyseq = np.histogram(nyseq, bins=range(Ny+2))
    histxyseq = np.histogram(nxyseq, bins=range(Ny*(Ny+2)+1))
    # return sum([histxyseq[0][i] * log(N * float(histxyseq[0][i]) / (histxseq[0][i%Ny] * histyseq[0][i/Ny]), 2) for i in histxyseq[0] if not i==0])/(N*N)
    return sum([hist * log(N*float(hist) /(histxseq[0][n%(Ny+1)]*histyseq[0][n/(Ny+1)]),2) for hist,n in zip(*histxyseq) if not (hist == 0 or histxseq[0][n%(Ny+1)] == 0 or histyseq[0][n/(Ny+1)])])/(N*N)

#一次元時系列同士のtransfer entropy
#xseq <- yseqを測定
#時系列は、後ろにいくほど古いとする
def tec(xseq , yseq):
   xmax,xmin = max(xseq),min(xseq)
   ymax,ymin = max(yseq),min(yseq)
   N = min(len(xseq) , len(yseq)) - 1
   Nx = int(log(N , 2)+1)
   Ny = Nx
   p , px , pxx , pxy = dict() , dict() , dict() , dict()
   for (x1 , (x,y)) in zip(xseq[:-1] , zip(xseq[1:] , yseq[1:])):
       nx1 = int(Nx*(x1-xmin)/(xmax-xmin))
       nx = int(Nx*(x-xmin)/(xmax-xmin))
       ny = int(Ny*(y-ymin)/(ymax-ymin))
       p[(nx1,nx,ny)] = p.get( (nx1,nx,ny) , 0) + 1
       px[nx] = px.get(nx , 0) + 1
       pxx[(nx1,nx)] = pxx.get( (nx1,nx) , 0) + 1
       pxy[(nx,ny)] = pxy.get( (nx,ny) , 0) + 1
   R=[p0*log( float(p0*px[nx])/(pxx[(nx1,nx)]*pxy[(nx,ny)]) , 2) for ((nx1,nx,ny) , p0) in p.iteritems()]
   return sum(R)/(N-1)

def test_sample():
   xs = [random.uniform(-1,1) for _ in xrange(1000)]
   ys = [2*x+0.01*random.random() for x  in xs[1:]]
   xs = xs[:-1]   #長さをysに合わせておく
   print "T(X<-Y) = %2.5f" % tec(xs , ys)
   print "T(X->Y) = %2.5f" % tec(ys , xs)
   print "I(X,Y) = %2.5f" % mic(xs , ys)
   print "I(X',Y) = %2.5f" % mic(xs[1:] , ys[:-1])
   print "I(X'',Y) = %2.5f" % mic(xs[2:] , ys[:-2])
   print "I(X,Y) = %2.5f" % npmic(np.array(xs), np.array(ys))
   print "I(X',Y) = %2.5f" % npmic(np.array(xs[1:]), np.array(ys[:-1]))

def npmic_test():
    theta = np.arange(0,10*np.pi,0.01*np.pi)
    xs = np.sin(theta)
    ys = np.cos(theta)
    zs = np.random.random(len(xs))+ys*0.1
    print npmic(xs, zs)

#-- test
if __name__=="__main__":
    npmic_test()
    test_sample()