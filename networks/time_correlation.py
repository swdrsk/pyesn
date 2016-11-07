#coding:utf-8
from math import log
import sys
import random


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
    import random
   xs = [random.uniform(-1,1) for _ in xrange(100)]
   ys = [2*x+0.01*random.random() for x  in xs[1:]]
   xs = xs[:-1]   #長さをysに合わせておく
   print "T(X<-Y) = %2.5f" % tec(xs , ys)
   print "T(X->Y) = %2.5f" % tec(ys , xs)
   print "I(X,Y) = %2.5f" % mic(xs , ys)
   print "I(X',Y) = %2.5f" % mic(xs[1:] , ys[:-1])
   print "I(X'',Y) = %2.5f" % mic(xs[2:] , ys[:-2])
    

#-- test
if __name__=="__main__":
    test_sample()
