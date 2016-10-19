#!/usr/bin/python

import numpy as np
import pdb

class lampel_ziv:
    def lz_complexity1d(self,s):
        i, k, l = 0, 1, 1
        k_max = 1
        n = len(s) - 1
        c = 1
        while True:
            if s[i + k - 1] == s[l + k - 1]:
                k = k + 1
                if l + k >= n - 1:
                    c = c + 1
                    break
            else:
                if k > k_max:
                    k_max = k
                i = i + 1
                if i == l:
                    c = c + 1
                    l = l + k_max
                    if l + 1 > n:
                        break
                    else:
                        i = 0
                        k = 1
                        k_max = 1
                else:
                    k = 1
        return c
    
    def lz_complexity2d(self,s):
        L2 = len(s)
        L1 = len(s[0])
        c,r,q,k,i = 1,0,0,1,1
        while True:
            a = i+k-1 if q==r else L1 #a = i+k-1
            """
            try:
                print s[r][i:i+k],s[q][0:a]
            except:
                print "r:%d,i:%d,k:%d,q:%d,a:%d"%(r,i,k,q,a)
            """    
            if self.search(s[r][i:i+k],s[q][0:a]):
                k = k+1
                if i+k>L1:
                    r=r+1
                    if r>L2-1:#r>L2
                        c=c+1
                        break
                    else:
                        i,q,k=0,r-1,1
                        continue
                else:
                    continue
                
            else:
                q = q-1
                if q<0:#q<1
                    c,i=c+1,i+k
                    if i+1>L1:
                        r=r+1
                        if r>L2-1:#r>L2
                            c=c+1
                            break
                        else:
                            i,q,k=0,r-1,1
                            continue
                    else:
                        q,k=r,1
                        continue
                else:
                    continue
        return c

    def search(self,arr,arr_dic):
        '''
        when there is arr pattern in arr_dic, return True
        arr, arr_dic: list
        '''
        if len(arr)>len(arr_dic): return False
        rst=False
        for i in range(len(arr_dic)-len(arr)):
            for j in range(len(arr)):
                if not arr_dic[j+i]==arr[j]:
                    break
                if j>=len(arr)-1:
                    rst=True
                    return rst
        return rst

    def normalized_lz(self,s):
        L2 = len(s)
        L1 = len(s[0])
        prob = 0
        for i in range(L2):
            prob += s[i].count(1)
        prob = float(prob)/(L1*L2)
        if 0<prob and prob<1:
            entropy = -prob*np.log2(prob) -(1-prob)*np.log2(1-prob)
            cl = self.lz_complexity2d(s)
            norm_cl = cl*np.log2(L1*L2)/(L1*L2*entropy)
        else:
            entropy = 0
            norm_cl = 0
        return norm_cl
        
    
def generate_random_binary(a,b,threshold):
    import random
    rst = [[1 if random.random()<threshold else 0 for i in range(b)] for j in range(a)]
    return rst

def generate_regular_binary(a,b,threshold):
    import random
    tmp = [1 if random.random()<threshold else 0 for i in range(b)]
    rst = [tmp for j in range(a)]
    return rst

def generate_impulse_binary(a,b):
    import random
    tmp = [0 for i in range(b)]
    tmp[1] = 1
    rst = [tmp for j in range(a)]
    return rst

def main():

    data1d = '1001111011000010'
    #data2d = [[0,0,0,1,0,1,0,1,1,0,0,0,1],[0,0,1,1,0,1,1,0,1,0,1,0,1],[1,1,0,1,0,0,1,1,1,0,1,0,0]]
    data2d = generate_regular_binary(5,20,0.1)

    print data2d
    #l = lampel_ziv()
    #lz = l.lz_complexity2d(data2d)
    #nz = l.normalized_lz(data2d)
    #assert lz == 6 
    print lz,nz


def test():
    import matplotlib.pyplot as plt
    l = lampel_ziv()
    nzs = []
    lzs = []
    for i in range(11):
        data2d = generate_regular_binary(10,100,i*0.1)
        lz = l.lz_complexity2d(data2d)
        nz = l.normalized_lz(data2d)
        lzs.append(lz)
        nzs.append(nz)
    print lzs
    print nzs
    nzs = []
    lzs = []
    for i in range(11):
        data2d = generate_random_binary(10,100,i*0.1)
        lz = l.lz_complexity2d(data2d)
        nz = l.normalized_lz(data2d)
        lzs.append(lz)
        nzs.append(nz)
    print lzs
    print nzs
    data2d = generate_impulse_binary(10,100)
    lz = l.lz_complexity2d(data2d)
    nz = l.normalized_lz(data2d)
    print lz
    print nz


def test2():
    l = lampel_ziv()
    for i in range(1,20):
        data1d = generate_random_binary(1,100,0.05*i)
        lz1d= l.lz_complexity1d(data1d[0])
        lz2d = l.lz_complexity2d(data1d)
        print lz1d,lz2d
    
if __name__ == '__main__':
    #main()
    test2()
    
