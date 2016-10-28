# coding:utf-8
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import numpy as np
import scipy.optimize
import pandas as pd


class Arm2Link:
    def __init__(self, q=None, q0=None, L=None):
        self.q = [math.pi/4, math.pi/4] if q is None else q
        self.q0 = np.array([math.pi/4, math.pi/4]) if q0 is None else q0
        self.L = np.array([100, 100]) if L is None else L

        self.max_angles = [math.pi, math.pi/4]
        self.min_angles = [0, -math.pi/4]

    def get_handpos(self, q=None):
        if q is None:
            q = self.q
        x = self.L[0]*np.cos(q[0]) + self.L[1]*np.cos(q[0]+q[1])
        y = self.L[0]*np.sin(q[0]) + self.L[1]*np.sin(q[0]+q[1])
        return [x, y]

    def get_jointpos(self, q=None):
        if q is None:
            q = self.q
        x = self.L[0] * np.cos(q[0])
        y = self.L[0] * np.sin(q[0])
        return [x, y]

    def get_startpos(self):
        return [0, 0]

    def set_joints(self, joints):
        self.q = list(joints)


class ArmOrbit(Arm2Link):
    def __init__(self,q=None, q0=None, L=None):
        Arm2Link.__init__(self, q=q, q0=q0, L=L)
        self.orbit = None
        self.rst = []
        self.step = 0

    def set_orbit(self, orbit):
        self.orbit = orbit

    def inv_kin(self, xy):
        def distance_to_default(q, *args):
            # weights found with trial and error, get some wrist bend, but not much
            weight = [1, 1]
            return np.sqrt(np.sum([(qi - q0i) ** 2 * wi
                                   for qi, q0i, wi in zip(q, self.q0, weight)]))

        def x_constraint(q, xy):
            x = (self.L[0] * np.cos(q[0]) + self.L[1] * np.cos(q[0] + q[1])) - xy[0]
            return x

        def y_constraint(q, xy):
            y = (self.L[0] * np.sin(q[0]) + self.L[1] * np.sin(q[0] + q[1])) - xy[1]
            return y

        return scipy.optimize.fmin_slsqp(
            func=distance_to_default,
            x0=self.q,
            eqcons=[x_constraint,
                    y_constraint],
            args=(xy,),
            iprint=0)  # iprint=0 suppresses output

    def get_jointorbit(self):
        if self.orbit is None:
            return None
        rst = []
        for x, y in self.orbit:
            rst.append(self.inv_kin(xy=[x, y]))
        return rst

    def yield_jointorbit(self):
        if self.orbit is None:
            pass
        xy = self.orbit[self.step]
        self.step += 1
        return self.inv_kin(xy)

    def save_orbit(self, filename):
        if self.orbit is None:
            print('let set_orbit')
            return None
        else:
            rst = self.get_jointorbit()
            pd.DataFrame(rst, columns=['output1', 'output2']).to_csv(filename, index=False)
            return rst


filename = "../data/arms.csv"
arm = ArmOrbit(L=[100,100])
step_angle = np.arange(0, 100*np.pi, 0.05*np.pi)
orbit = [[110+30*np.sin(i) for i in step_angle],[110+30*np.cos(i) for i in step_angle]]
arm.set_orbit(zip(*orbit))
arm.save_orbit(filename)
fig = plt.figure()
Lmax = np.sum(arm.L)
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                     xlim=(-Lmax,Lmax), ylim=(-Lmax,Lmax))
ax.grid()
line, = ax.plot([], [], 'o-', lw=3)

def init():
    line.set_data([], [])
    return line,

def animation0(i):
    global arm
    joints = arm.yield_jointorbit()
    arm.set_joints(joints)
    startpos = arm.get_startpos()
    jointpos = arm.get_jointpos()
    handpos = arm.get_handpos()
    line.set_data(*zip(startpos, jointpos, handpos))
    #line.set_data((0,100,100), (0,0,100))
    return line,

ani = animation.FuncAnimation(fig, animation0, frames=1000, blit=True, interval=100, init_func=init)
plt.show()

#print arm.get_jointorbit()