#!/usr/bin/env python3
# Software License Agreement (BSD License)
#
# Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

"""
Description: Cartesian velocity control with real-time plotting
"""

import os
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from xarm.wrapper import XArmAPI

#######################################################
"""
Just for test example
"""
if len(sys.argv) >= 2:
    ip = sys.argv[1]
else:
    try:
        from configparser import ConfigParser
        parser = ConfigParser()
        parser.read('../robot.conf')
        ip = parser.get('xArm', 'ip')
    except:
        ip = input('Please input the xArm ip address:')
        if not ip:
            print('input error, exit')
            sys.exit(1)
########################################################

arm = XArmAPI(ip)
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)
time.sleep(1)

arm.reset(wait=True)

# Set cartesian velocity control mode
arm.set_mode(5)
arm.set_state(0)
time.sleep(1)

# Data for plotting
times = []
velocities = []

# Set up plot
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)

ax.set_xlim(0, 10)
ax.set_ylim(-1200, 1200)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Velocity (mm/s)')
ax.set_title('Cartesian Velocities of End Effector')

start_time = time.time()

# Function to calculate velocity
def calculate_velocity(prev_pos, curr_pos, dt):
    return [(curr_pos[i] - prev_pos[i]) / dt for i in range(3)]

prev_time = start_time
prev_pos = arm.get_position()[:3]  # Get initial position (x, y, z)

def update_plot(frame):
    global prev_time, prev_pos
    current_time = time.time()
    dt = current_time - prev_time

    if dt == 0:
        return line,

    curr_pos = arm.get_position()[:3]
    velocity = calculate_velocity(prev_pos, curr_pos, dt)
    velocities.append(velocity[0])  # Plotting X-axis velocity, you can add others if needed

    times.append(current_time - start_time)
    prev_time = current_time
    prev_pos = curr_pos

    line.set_data(times, velocities)
    ax.set_xlim(max(0, times[-1] - 10), times[-1] + 1)
    ax.set_ylim(min(velocities) - 100, max(velocities) + 100)
    return line,

# ani = animation.FuncAnimation(fig, update_plot, blit=True, interval=100)

# Set initial velocities and animate
arm.vc_set_cartesian_velocity([150, 0, 0, 0, 0, 0])

time.sleep(1)
arm.vc_set_cartesian_velocity([1100, 0, 0, 0, 0, 0])
time.sleep(0.55)

# Stop the arm
arm.vc_set_cartesian_velocity([0, 0, 0, 0, 0, 0])

# Disconnect arm
arm.disconnect()

plt.show()