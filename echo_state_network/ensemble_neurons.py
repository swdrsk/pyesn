import matplotlib.pyplot as plt
import numpy as np
import sys
from brian2 import *
from numpy.random import randn
from numpy.linalg import norm
import pdb


neuron_num = 800

def echo_state_neuron(inputs):
    a = 0.8
    m = 0
    output = np.zeros(len(inputs))
    for time, inp in enumerate(inputs):
        m = (1-a)*m + a*np.tanh(m+inp)
        output[time] = m
    return output


def LIFensemble_exp(hz):
    exneuron = neuron_num
    inneuron = neuron_num/8
    duration = 1 * second

    taum = 30 * ms
    taue = 10 * ms # 5
    taui = 10 * ms # 10
    Vt = -48.5 * mV # threshold 50
    Vr = -55 * mV # reset
    El = -49 * mV # -49 * mV

    eqs = '''
    dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
    dge/dt = -ge/taue : volt
    dgi/dt = -gi/taui : volt
    '''

    P = NeuronGroup(exneuron+inneuron, eqs, threshold='v>Vt', reset='v = Vr', refractory=3 * ms,
                    method='linear')
    P.v = 'Vr + rand() * (Vt - Vr)'
    P.ge = 0 * mV
    P.gi = 0 * mV

    we = (60 * 0.27 / 10) * mV  # excitatory synaptic weight (voltage)
    wi = (-20 * 4.5 / 10) * mV  # inhibitory synaptic weight
    Ce = Synapses(P, P, on_pre='ge += we')
    Ci = Synapses(P, P, on_pre='gi += wi')
    Ce.connect('i<exneuron', p=0.01)
    Ci.connect('i>=exneuron', p=0.01)

    Pg = PoissonGroup(exneuron, hz * Hz)
    weg = (60 * 0.27 / 10) * mV
    Ceg = Synapses(Pg, P, on_pre='ge += we')
    Ceg.connect('i<exneuron', p=0.001)
    pp_mon = PopulationRateMonitor(Pg)
    run(duration)
    out_hz = np.average(pp_mon.smooth_rate(window="gaussian", width=10 * ms))
    return out_hz


def experiment():
    filename = "../result/LIF_neuron.csv"
    f = open(filename, "w")
    ctrl_hz = range(0, 80, 5)
    out_hz = np.zeros(len(ctrl_hz))
    for ci,hz in enumerate(ctrl_hz):
        out_hz[ci] = LIFensemble_exp(hz)
        f.write("%f,%f\n" % (hz, out_hz[ci]))
    f.close()
    plt.plot(ctrl_hz, out_hz)
    plt.scatter(ctrl_hz, out_hz, color="r")
    plt.ylim(0,max(out_hz)+10)
    plt.xlim(0,max(ctrl_hz)+10)
    plt.title('relation between input and output')
    plt.ylabel('post neurons [Hz]')
    plt.xlabel('pre neurons [Hz]')
    plt.savefig('../result/LIF_neurons.png')
    plt.show()


def LIFensembles(input_rate,input_flag=True):
    exneuron = neuron_num
    inneuron = neuron_num/8
    duration = 1 * second

    taum = 30 * ms
    taue = 12 * ms # 5
    taui = 24 * ms # 10
    Vt = -48.5 * mV # threshold 50
    Vr = -55 * mV # reset
    El = -49 * mV # -49 * mV

    eqs = '''
    dv/dt  = (ge+gi-(v-El))/taum : volt (unless refractory)
    dge/dt = -ge/taue : volt
    dgi/dt = -gi/taui : volt
    '''

    P = NeuronGroup(exneuron+inneuron, eqs, threshold='v>Vt', reset='v = Vr', refractory=3 * ms,
                    method='linear')
    P.v = 'Vr + rand() * (Vt - Vr)'
    P.ge = 0 * mV
    P.gi = 0 * mV

    we = (60 * 0.27 / 10) * mV  # excitatory synaptic weight (voltage)
    wi = (-20 * 4.5 / 10) * mV  # inhibitory synaptic weight
    Ce = Synapses(P, P, on_pre='ge += we')
    Ci = Synapses(P, P, on_pre='gi += wi')
    Ce.connect('i<exneuron', p=0.01)
    Ci.connect('i>=exneuron', p=0.01)

    s_mon = SpikeMonitor(P)
    p_mon = PopulationRateMonitor(P)


    if input_flag:
        inputtype = "control_input"
        if inputtype == "poisson":
            Pg = PoissonGroup(exneuron,25*Hz)
        elif inputtype == "control_input":
            values = input_firing_rate2discrete_spike(input_rate, neuron_num=exneuron)
            ta = TimedArray(values=values, dt=duration/values.shape[0])
            Pg = NeuronGroup(exneuron, 'vi = ta(t,i) : 1', threshold='vi > 0.5', reset='vi = 0', refractory=0 * ms)
        else:
            values = [0,1]
            ta = TimedArray(values=values, dt=duration/len(values))
            Pg = NeuronGroup(exneuron, 'vi = ta(t) : 1', threshold='vi > 0.5', reset='v = 0', refractory=10 * ms)
        weg = (60 * 0.27 / 10) * mV
        Ceg = Synapses(Pg, P, on_pre='ge += we')
        Ceg.connect('i<exneuron', p=0.005)
        pp_mon = PopulationRateMonitor(Pg)

    run(duration)

    if input_flag:
        subplot(311)
        plot(s_mon.t / ms, s_mon.i, '.k')
        xlabel('Time (ms)')
        ylabel('Neuron index')
        subplot(312)
        plot(pp_mon.t/ms, pp_mon.smooth_rate(window='gaussian', width=10*ms)/Hz)
        xlabel('Time (ms)')
        ylabel('Input Rate [Hz]')
        subplot(313)
        #plot(p_mon.t/ms, p_mon.smooth_rate(window='gaussian', width=5*ms)/Hz, label='5ms')
        #plot(p_mon.t/ms, p_mon.smooth_rate(window='gaussian', width=10*ms)/Hz, label='10ms')
        plot(p_mon.t/ms, p_mon.smooth_rate(window='gaussian', width=10*ms)/Hz, label='10ms')
        xlabel("Time (ms)")
        ylabel('Output Rate [Hz]')
    else:
        subplot(211)
        plot(s_mon.t / ms, s_mon.i, '.k')
        xlabel('Time (ms)')
        ylabel('Neuron index')
        subplot(212)
        #plot(p_mon.t/ms, p_mon.smooth_rate(window='gaussian', width=5*ms)/Hz, label='5ms')
        plot(p_mon.t/ms, p_mon.smooth_rate(window='gaussian', width=10*ms)/Hz, label='10ms')
        plot(p_mon.t/ms, p_mon.smooth_rate(window='gaussian', width=20*ms)/Hz, label='20ms')
        legend()
        xlabel("Time (ms)")
        ylabel('Output Rate [Hz]')




def input_firing_rate2discrete_spike(input_rate, neuron_num=1,display=False):

    def event2raster(event_times_list):
        return zip(*np.where(event_times_list == 1))

    def poisson_spike(rate):
        ''' rate [Hz] return period [0.1*ms]'''
        prob = rate*1./10000
        return 1 if np.random.random() < prob else 0

    if neuron_num == 1:
        spikes = map(lambda x:poisson_spike(x), input_rate)
    else:
        spikes = np.zeros([len(input_rate), neuron_num])
        for i in xrange(neuron_num):
            spikes[:,i] = map(lambda x:poisson_spike(x), input_rate)
    if display:
        plt.subplot(211)
        plt.plot(range(len(input_rate)), input_rate)
        plt.ylim(0, max(input_rate) * 1.2)
        plt.subplot(212)
        plt.scatter(*np.where(spikes == 1))
        plt.xlim(0, len(input_rate))
        plt.show()
    return spikes


def draw_output(output):
    plt.figure()
    plt.plot(range(len(output)), output)
    plt.xlabel("Time")
    plt.ylabel("output")
    ylim(min(output)-0.2, max(output)+0.2)

def draw_attractor(output):
    plt.plot(output[:-1],output[1:])

if __name__=="__main__":
    volt = 40
    input = np.sin(np.array(range(3000))*0.01)
    _input = [0 for i in range(100)] + [volt for i in range(200)] + [0 for i in range(100)] + [volt for i in range(200)]
    # _input = [0 for i in range(200)] + [volt for i in range(400)]
    #output = echo_state_neuron(input)
    #draw_attractor(output)
    #_input = (np.sin(np.array(range(1000))*0.01) + 1.1) * 10
    #spikes = input_firing_rate2discrete_spike(_input, neuron_num)
    #LIFensembles(_input)
    #echo_out = echo_state_neuron(np.array(_input)/volt)
    #draw_output(echo_out)
    #plt.show()
    experiment()
