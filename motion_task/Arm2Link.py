# coding:utf-8
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import numpy as np




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

arm = Arm2Link(L=[100,100])
la = Lorenz(10,28,8./3)
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
    global arm,la
    j1,j2,j3 = la.step()
    arm.set_joints([j1*math.pi, j2*math.pi])
    startpos = arm.get_startpos()
    jointpos = arm.get_jointpos()
    handpos = arm.get_handpos()
    line.set_data(*zip(startpos, jointpos, handpos))
    #line.set_data((0,100,100), (0,0,100))
    return line,

ani = animation.FuncAnimation(fig, animation0, frames=1000, blit=True, interval=300, init_func=init)
plt.show()
