import matplotlib.pyplot as plt
import numpy as np

from brian2 import *
from numpy.random import randn
from numpy.linalg import norm


def draw_poisson_process():
    deltatime = np.random.poisson(3,30)
    t = 0
    timing = []
    for dt in list(deltatime):
        t += dt
        timing.append(t)
    plt.scatter(timing,list(np.zeros(30)))
    plt.show()


def generate_poisson_process():
    deltatime = np.random.poisson(3,1000)
    t = 0
    timing = []
    for dt in list(deltatime):
        t += dt
        timing.append(t)
    return timing


def draw_firing_moving_averate(windowsize):
    timing = generate_poisson_process()
    _ = firing_moving_averate(timing,windowsize)
    plt.plot(zip(*_)[1])
    plt.show()


def firing_moving_averate(timing,windowsize):
    timing = np.array(timing)
    start = 0
    end = start + windowsize
    moving_averate = []
    while end<timing[-1]:
        firing = np.size(np.where((start <= timing)&(timing <= end)))
        moving_averate.append((start,firing))
        start += 1
        end += 1
    return moving_averate

def echo_state_neuron(input):
    a = 0.3
    m = 0
    output = []
    for i in input:
        m = (1-a)*m + a*np.tanh(m+i)
        output.append(m)
    return output


def LIFensembles(input_flag = True):
    exneuron = 800
    inneuron = 100
    duration = 0.1 * second

    taum = 30 * ms
    taue = 5 * ms
    taui = 10 * ms
    Vt = -50 * mV # threshold
    Vr = -60 * mV # reset
    El = -49 * mV #

    eqs = '''
    dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
    dge/dt = -ge/taue : volt
    dgi/dt = -gi/taui : volt
    '''

    P = NeuronGroup(exneuron+inneuron, eqs, threshold='v>Vt', reset='v = Vr', refractory=5 * ms,
                    method='linear')
    P.v = 'Vr + rand() * (Vt - Vr)'
    P.ge = 0 * mV
    P.gi = 0 * mV

    we = (60 * 0.27 / 10) * mV  # excitatory synaptic weight (voltage)
    wi = (-20 * 4.5 / 10) * mV  # inhibitory synaptic weight
    Ce = Synapses(P, P, on_pre='ge += we')
    Ci = Synapses(P, P, on_pre='gi += wi')
    Ce.connect('i<exneuron', p=0.02)
    Ci.connect('i>=exneuron', p=0.02)

    s_mon = SpikeMonitor(P)
    p_mon = PopulationRateMonitor(P)

    if input_flag:
        Pg = PoissonGroup(exneuron,25*Hz)
        # ta = TimedArray(values=[0,1,0,1,0], dt=duration/5)
        # Pg = NeuronGroup(exneuron, 'vi = ta(t) : 1', threshold='vi>0.5', reset='vi=0',refractory=5*ms)
        weg = (60 * 0.27 / 10) * mV
        Ceg = Synapses(Pg, P, on_pre='ge += we')
        Ceg.connect('i<exneuron', p=0.02)
        pp_mon = PopulationRateMonitor(Pg)


    run(duration)

    if input_flag:
        subplot(311)
        plot(s_mon.t / ms, s_mon.i, '.k')
        xlabel('Time (ms)')
        ylabel('Neuron index')
        subplot(312)
        plot(pp_mon.t/ms, pp_mon.smooth_rate(window='gaussian', width=5*ms)/Hz)
        xlabel('Time (ms)')
        ylabel('Input Rate [Hz]')
        subplot(313)
        plot(p_mon.t/ms, p_mon.smooth_rate(window='gaussian', width=5*ms)/Hz, label='5ms')
        plot(p_mon.t/ms, p_mon.smooth_rate(window='gaussian', width=10*ms)/Hz, label='10ms')
        plot(p_mon.t/ms, p_mon.smooth_rate(window='gaussian', width=20*ms)/Hz, label='20ms')
        xlabel("Time (ms)")
        ylabel('Output Rate [Hz]')
    else:
        subplot(211)
        plot(s_mon.t / ms, s_mon.i, '.k')
        xlabel('Time (ms)')
        ylabel('Neuron index')
        subplot(212)
        plot(p_mon.t/ms, p_mon.smooth_rate(window='gaussian', width=5*ms)/Hz, label='5ms')
        plot(p_mon.t/ms, p_mon.smooth_rate(window='gaussian', width=10*ms)/Hz, label='10ms')
        plot(p_mon.t/ms, p_mon.smooth_rate(window='gaussian', width=20*ms)/Hz, label='20ms')
        xlabel("Time (ms)")
        ylabel('Output Rate [Hz]')

    show()


def draw_output(output):
    plt.plot(range(len(output)), output)
    plt.xlabel("Time")
    plt.ylabel("output")
    plt.show()

def draw_attractor(output):
    plt.plot(output[:-1],output[1:])
    plt.show()

if __name__=="__main__":
    input = np.sin(np.array(range(3000))*0.01)
    #input = [0 for i in range(100)] + [1 for i in range(200)] + [0 for i in range(100)] + [1 for i in range(200)]
    #output = echo_state_neuron(input)
    #draw_attractor(output)
    LIFensembles()