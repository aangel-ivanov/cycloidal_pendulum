from numpy import sin, cos
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import matplotlib.animation as animation

R = 1.0
g = 9.81  # acceleration due to gravity, in m/s^2
L = 4 * R  # length of pendulum in m
# m = 1.0  # mass of pendulum in kg
t_stop = 10  # how many seconds to simulate
history_len = 1000  # how many trajectory points to display


def derivs(state, t): # given positions and velocities return accelerations 

    dydx = np.zeros_like(state)
    dydx[0] = state[1]
    
    sub1 = -(g / 4 * R) * sin(state[0])
    sub2 = state[1] * state[1] * sin(state[0])
    
    dydx[1] = (sub1 + sub2) / cos(state[0])
    
    return dydx

# create a time array from 0..t_stop sampled at 0.02 second steps
dt = 0.03
t = np.arange(0, t_stop, dt)

# th is the initial angle (degrees)
# w is the initial angular velocity (degrees per second)
th = 90.01
w = 0.0

# initial state
state = np.radians([th, w])

# integrate your ODE using scipy.integrate
y = integrate.odeint(derivs, state, t)

# first column of integrate matrix is position (angle) values
x1 = R * (2 * y[:, 0] + sin(2 * y[:, 0])) - (2 * np.pi)
y1 = R * (-3 - cos(2 * y[:, 0]))

# coordinate of contact point
x2 = R * (2 * y[:, 0] - sin(2 * y[:, 0])) - (2 * np.pi)
y2 = R * (-1 + cos(2 * y[:, 0]))

# cycloid parametric equations
n = np.linspace(-np.pi, np.pi, 1000)
x3 = R * (n - sin(n))
y3 = R * (-1 + cos(n))


# Define the meta data for the movie
FFMpegWriter = animation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='a cycloidal pendulum')
writer = FFMpegWriter(fps=15, metadata=metadata)

# Initialize the movie
fig = plt.figure()

# plot the cycloidal curve
plt.plot(x3, y3, 'b')

# plot the mass, contact points and line
mass, = plt.plot([], [], 'ro', markersize = 8)
contact_point, = plt.plot([], [], 'b', markersize = 0.1)
line, = plt.plot([], [], 'b', animated=True)

plt.xlim([-3.5, 3.5])
plt.ylim([-5, 0])

# Update the frames for the movie
with writer.saving(fig, "cycloidal_pendulum.mp4", 100):
    for i in range(300):
        x0 = x1[i]
        y0 = y1[i]
        x_c = x2[i]
        y_c = y2[i]
        mass.set_data(x0, y0)
        contact_point.set_data(x_c, y_c)
        line.set_data([x1[i], x2[i]], [y1[i], y2[i]])
        writer.grab_frame()
