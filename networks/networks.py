#coding:utf-8

import networkx as nx
import matplotlib.pyplot as plt
from random import random,randint

N = 1000
K = 5
TMAX = 2

def test():
    G = nx.Graph()
    G.add_node(1)
    G.add_node(2)
    G.add_node(3)
    G.add_edge(1,2)
    G.add_edge(2,3)
    nx.draw(G)
    plt.savefig("./data/a.png")

def make_small_world_network(n,k,p):
    assert p >= 0 and p <=1, "p is reconnect probability"
    assert n > 0, "n is num nodes"
    assert k > 0, "2*k is num of link per one node"

    G = nx.Graph()
    G.add_nodes_from(range(1,n+1))

    for i in range(1,n+1):
        for j in range(1,k+1):
            v = i
            u = i+j 
            if u > n:
                u = u % (n+1) + 1
            G.add_edge(v,u)

    for i in range(1,n+1):
        for j in range(1,k+1):
            if random() < p:
                # reconnect edge
                v = i
                old_u = i+j
                if old_u > n:
                    old_u = old_u % (n+1) + 1
                new_u = randint(1,n)
                while v != new_u and G.has_edge(v,new_u):
                    new_u = randint(1,n)
                G.remove_edge(v,old_u)
                G.add_edge(v,new_u)
    return G

def sim(G,r):
    time_count = 0
    infected_count = 1
    while True:
        time_count += 1

        remove_candidate = []
        for n in G.nodes_iter():
            if 'infected' in G.node[n]:
                G.node[n]['infected'] += 1
                if G.node[n]['infected'] >= TMAX:
                    # dead
                    remove_candidate.append(n)
        # remove dead node
        for i in remove_candidate:
            G.remove_node(i)

        if G.order() <= 0:
            # all node are infected and died
            return (time_count,infected_count)

        infected_nodes = 0
        infected_candidate = set()
        for n in G.nodes_iter():
            if 'infected' in G.node[n]:
                infected_nodes += 1
                for u,v in G.edges_iter(n):
                    if random() < r:
                        if not 'infected' in G.node[v]:
                            infected_candidate.add(v)
        for i in infected_candidate:
            G.node[i]['infected'] = 0
            infected_count += 1

        if infected_nodes == 0 or infected_count == N:
            return (time_count,infected_count)

def main():
    ps = []
    ls = []
    rs = []
    ts = []
    for i in range(1,10):
        ps.append(0.0001*i)
    for i in range(1,10):
        ps.append(0.001*i)
    for i in range(1,10):
        ps.append(0.01*i)
    for i in range(1,10):
        ps.append(0.1*i)
    ps.append(1)

    G0 = make_small_world_network(N,K,0)
    L0 = nx.average_shortest_path_length(G0)
    G0.node[1]['infected'] = 0
    time_count,infected_count = sim(G0,1)
    T0 = time_count

    for p in ps:
        r = 0.01
        while True:
            G = make_small_world_network(N,K,p)
            G.node[1]['infected'] = 0
            time_count,infected_count = sim(G,r)
            if infected_count >= N/2:
                break
            r += 0.01
        rs.append(r)
        G = make_small_world_network(N,K,p)
        G.node[1]['infected'] = 0
        l = nx.average_shortest_path_length(G)
        ls.append(l/L0)
        time_count,infected_count = sim(G,1)
        ts.append(time_count/T0)
        print("{0} {1} {2} {3}".format(p,r,time_count/T0,l/L0))

    plt.xlabel("p")
    plt.ylabel("r_half")
    plt.semilogx(ps,rs,"rs")
    plt.savefig("./data/sim_a.png")

    plt.close()
    plt.xlabel("p")
    plt.semilogx(ps,ls,"rs",label="L(p)/L0")
    plt.semilogx(ps,ts,"bo",label="T(p)/T0")
    plt.legend()
    plt.savefig("./data/sim_b.png")

import networkx as nx
import matplotlib.pyplot as plt
from random import random,randint

N = 1000
K = 5

def make_small_world_network(n,k,p):
    assert p >= 0 and p <=1, "p is reconnect probability"
    assert n > 0, "n is num nodes"
    assert k > 0, "2*k is num of link per one node"

    G = nx.Graph()
    G.add_nodes_from(range(1,n+1))

    for i in range(1,n+1):
        for j in range(1,k+1):
            v = i
            u = i+j 
            if u > n:
                u = u % (n+1) + 1
            G.add_edge(v,u)

    for i in range(1,n+1):
        for j in range(1,k+1):
            if random() < p:
                # reconnect edge
                v = i
                old_u = i+j
                if old_u > n:
                    old_u = old_u % (n+1) + 1
                new_u = randint(1,n)
                while v != new_u and G.has_edge(v,new_u):
                    new_u = randint(1,n)
                G.remove_edge(v,old_u)
                G.add_edge(v,new_u)
    return G


def main():
    ps = []
    ls = []
    cs = []
    for i in range(1,10):
        ps.append(0.0001*i)
    for i in range(1,10):
        ps.append(0.001*i)
    for i in range(1,10):
        ps.append(0.01*i)
    for i in range(1,10):
        ps.append(0.1*i)
    ps.append(1)

    G0 = make_small_world_network(N,K,0)
    L0 = nx.average_shortest_path_length(G0)
    C0 = nx.average_clustering(G0)

    for p in ps:
        G = make_small_world_network(N,K,p)
        l = nx.average_shortest_path_length(G)
        c = nx.average_clustering(G)
        ls.append(l/L0)
        cs.append(c/C0)
        print("{0} {1} {2}".format(p,l/L0,c/C0))

    plt.xlabel("p")
    plt.semilogx(ps,ls,"rs",label="L(p)/L0")
    plt.semilogx(ps,cs,"bo",label="C(p)/C0")
    plt.legend()
    plt.savefig("small.png")

if __name__ == "__main__":
    main()

    
'''
if __name__ == "__main__":
    main()

''' 
