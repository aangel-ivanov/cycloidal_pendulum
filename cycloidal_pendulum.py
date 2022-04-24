from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation
from collections import deque

R = 1.0
g = 9.8  # acceleration due to gravity, in m/s^2
L = 4 * R  # length of pendulum in m
m = 1.0  # mass of pendulum in kg
t_stop = 5  # how many seconds to simulate
history_len = 500  # how many trajectory points to display


def derivs(state, t): # given positions and velocities return accelerations 

    dydx = np.zeros_like(state)
    dydx[0] = state[1]
    
    sub1 = -(g / 4 * R) * np.sin(state[0])
    sub2 = state[1] * state[1] * np.sin(state[0])
    
    dydx[1] = (sub1 + sub2) / np.cos(state[0])
    
    return dydx

# create a time array from 0..t_stop sampled at 0.02 second steps
dt = 0.02
t = np.arange(0, t_stop, dt)

# th is the initial angle (degrees)
# w is the initial angular velocity (degrees per second)
th = 120.0
w = 0.0

# initial state
state = np.radians([th, w])

# integrate your ODE using scipy.integrate.
integrate = integrate.odeint(derivs, state, t)

x = L*sin(integrate[:, 0])
y = L*cos(integrate[:, 0])

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
ax.set_aspect('equal')
ax.grid()

#line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)


def animate(i):
    thisx = [0, x[i]]
    thisy = [0, y[i]]

    if i == 0:
        history_x.clear()
        history_y.clear()

    history_x.appendleft(thisx[1])
    history_y.appendleft(thisy[1])

    #line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*dt))
    return trace, time_text


ani = animation.FuncAnimation(
    fig, animate, len(integrate), interval=dt*1000, blit=True)

ani.save('cycloidal_pendulum.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()
