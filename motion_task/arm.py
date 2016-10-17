# coding:utf-8


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import numpy as np
import scipy.optimize
import pyglet


class Arm3Link:
    """
    cited from : https://studywolf.wordpress.com/2013/04/11/inverse-kinematics-of-3-link-arm-with-constrained-minimization-in-python/
    """
    def __init__(self, q=None, q0=None, L=None):
        """Set up the basic parameters of the arm.
        All lists are in order [shoulder, elbow, wrist].

        :param list q: the initial joint angles of the arm
        :param list q0: the default (resting state) joint configuration
        :param list L: the arm segment lengths
        """
        # initial joint angles
        if q is None:
            q = [.3, .3, 0]
        self.q = q
        # some default arm positions
        if q0 is None:
            q0 = np.array([math.pi / 4, math.pi / 4, math.pi / 4])
        self.q0 = q0
        # arm segment lengths
        if L is None:
            L = np.array([1, 1, 1])
        self.L = L

        self.max_angles = [math.pi, math.pi, math.pi / 4]
        self.min_angles = [0, 0, -math.pi / 4]

    def get_xy(self, q=None):
        if q is None:
            q = self.q

        x = self.L[0] * np.cos(q[0]) + \
            self.L[1] * np.cos(q[0] + q[1]) + \
            self.L[2] * np.cos(np.sum(q))

        y = self.L[0] * np.sin(q[0]) + \
            self.L[1] * np.sin(q[0] + q[1]) + \
            self.L[2] * np.sin(np.sum(q))

        return [x, y]

    def get_joint_posision(self):
        x = np.array([0,
                      self.L[0] * np.cos(self.q[0]),
                      self.L[0] * np.cos(self.q[0]) + self.L[1] * np.cos(self.q[0] + self.q[1]),
                      self.L[0] * np.cos(self.q[0]) + self.L[1] * np.cos(self.q[0] + self.q[1]) +
                      self.L[2] * np.cos(np.sum(self.q))]) # + window.width / 2

        y = np.array([0,
                      self.L[0] * np.sin(self.q[0]),
                      self.L[0] * np.sin(self.q[0]) + self.L[1] * np.sin(self.q[0] + self.q[1]),
                      self.L[0] * np.sin(self.q[0]) + self.L[1] * np.sin(self.q[0] + self.q[1]) +
                      self.L[2] * np.sin(np.sum(self.q))])

        return np.array([x, y])

    def inv_kin(self, xy):

        def distance_to_default(q, *args):
            # weights found with trial and error, get some wrist bend, but not much
            weight = [1, 1, 1.3]
            return np.sqrt(np.sum([(qi - q0i) ** 2 * wi for qi, q0i, wi in zip(q, self.q0, weight)]))

        def x_constraint(q, xy):
            x = (self.L[0] * np.cos(q[0]) + self.L[1] * np.cos(q[0] + q[1]) +
                 self.L[2] * np.cos(np.sum(q))) - xy[0]
            return x

        def y_constraint(q, xy):
            y = (self.L[0] * np.sin(q[0]) + self.L[1] * np.sin(q[0] + q[1]) +
                 self.L[2] * np.sin(np.sum(q))) - xy[1]
            return y

        return scipy.optimize.fmin_slsqp(func=distance_to_default,
                                         x0=self.q, eqcons=[x_constraint, y_constraint],
                                         args=[xy], iprint=0)  # iprint=0 suppresses output


def plot():
    """A function for plotting an arm, and having it calculate the
    inverse kinematics such that given the mouse (x, y) position it
    finds the appropriate joint angles to reach that point."""

    # create an instance of the arm
    arm = Arm3Link(L=np.array([300, 200, 100]))

    # make our window for drawin'
    window = pyglet.window.Window()

    label = pyglet.text.Label('Mouse (x,y)', font_name='Times New Roman',
                              font_size=36, x=window.width // 2, y=window.height // 2,
                              anchor_x='center', anchor_y='center')

    def get_joint_positions():
        """This method finds the (x,y) coordinates of each joint"""

        x = np.array([0,
                      arm.L[0] * np.cos(arm.q[0]),
                      arm.L[0] * np.cos(arm.q[0]) + arm.L[1] * np.cos(arm.q[0] + arm.q[1]),
                      arm.L[0] * np.cos(arm.q[0]) + arm.L[1] * np.cos(arm.q[0] + arm.q[1]) +
                      arm.L[2] * np.cos(np.sum(arm.q))]) + window.width / 2

        y = np.array([0,
                      arm.L[0] * np.sin(arm.q[0]),
                      arm.L[0] * np.sin(arm.q[0]) + arm.L[1] * np.sin(arm.q[0] + arm.q[1]),
                      arm.L[0] * np.sin(arm.q[0]) + arm.L[1] * np.sin(arm.q[0] + arm.q[1]) +
                      arm.L[2] * np.sin(np.sum(arm.q))])

        return np.array([x, y]).astype('int')

    window.jps = get_joint_positions()

    @window.event
    def on_draw():
        window.clear()
        label.draw()
        for i in range(3):
            pyglet.graphics.draw(2, pyglet.gl.GL_LINES, ('v2i',
                                                         (window.jps[0][i], window.jps[1][i],
                                                          window.jps[0][i + 1], window.jps[1][i + 1])))

    @window.event
    def on_mouse_motion(x, y, dx, dy):
        # call the inverse kinematics function of the arm
        # to find the joint angles optimal for pointing at
        # this position of the mouse
        label.text = '(x,y) = (%.3f, %.3f)' % (x, y)
        arm.q = arm.inv_kin([x - window.width / 2, y])  # get new arm angles
        window.jps = get_joint_positions()  # get new joint (x,y) positions

    pyglet.app.run()


def test():
    ############Test it!##################

    arm = Arm3Link()

    # set of desired (x,y) hand positions
    x = np.arange(-.75, .75, .05)
    y = np.arange(0, .75, .05)

    # threshold for printing out information, to find trouble spots
    thresh = .025

    count = 0
    total_error = 0
    # test it across the range of specified x and y values
    for xi in range(len(x)):
        for yi in range(len(y)):
            # test the inv_kin function on a range of different targets
            xy = [x[xi], y[yi]]
            # run the inv_kin function, get the optimal joint angles
            q = arm.inv_kin(xy=xy)
            # find the (x,y) position of the hand given these angles
            actual_xy = arm.get_xy(q)
            # calculate the root squared error
            error = np.sqrt((np.array(xy) - np.array(actual_xy)) ** 2)
            # total the error
            total_error += error

            # if the error was high, print out more information
            if np.sum(error) > thresh:
                print '-------------------------'
                print 'Initial joint angles', arm.q
                print 'Final joint angles: ', q
                print 'Desired hand position: ', xy
                print 'Actual hand position: ', actual_xy
                print 'Error: ', error
                print '-------------------------'

            count += 1

    print '\n---------Results---------'
    print 'Total number of trials: ', count
    print 'Total error: ', total_error
    print '-------------------------'


def visualize():
    arm = Arm3Link(L = np.array([100, 100, 100]))

    fig = plt.figure()
    x = np.arange(0, 10, 0.1)
    ims = []
    for a in range(50):
        y = np.sin(x-a)
        im = plt.plot(x,y,"r")
        ims.append(im)

    ani = animation.ArtistAnimation(fig, ims)
    plt.show()


if __name__=="__main__":
    test()