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
import mujoco
import mujoco.viewer
import numpy as np
import time
# from scipy.integrate import simps
import matplotlib.pyplot as plt
import scipy.interpolate

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


# Integration timestep in seconds. This corresponds to the amount of time the joint
# velocities will be integrated for to obtain the desired joint positions.
integration_dt: float = 3

# Damping term for the pseudoinverse. This is used to prevent joint velocities from
# becoming too large when the Jacobian is close to singular.
damping: float = 1e-4

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95
Kori: float = 0.95

dt: float = 0.02

# Nullspace P gain.
Kn = np.asarray([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001])

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Maximum allowable joint velocity in rad/s.
max_angvel = 0.785

# PD control gains
Kp = 0.1  # Proportional gain
# Kp = 0.1
Kd = 1  # Derivative gain

## defining a function which which calculates the samples for the trajectory
def trajectory(initial_pos , final_pos, time_to_travel) :
    a1 =  (-32) * (final_pos - initial_pos) * (1/ (time_to_travel ** 4))
    a2 = -1 * a1 * time_to_travel
    number_of_waypoints = np.linalg.norm(final_pos - initial_pos) * 50 / 2 
    number_of_waypoints = int(number_of_waypoints)
    print("NO OF WAYPOINTS : ", number_of_waypoints)
    captured_waypoints = []
    velocity_waypoints = []
    acceleration_waypoints = []
    waypoints_index = 0
    while (waypoints_index <= number_of_waypoints) :
        time = (time_to_travel * waypoints_index) / number_of_waypoints
        if( time >=0 and time <= (time_to_travel / 2)) : 
            position = initial_pos + (a1/12) * (time ** 4) + (a2 / 6) * (time ** 3)
            captured_waypoints.append(np.array(position)) 
            velocity = (a1/3) * (time ** 3) + (a2/2) * (time ** 2)
            velocity_waypoints.append(np.array(velocity))
            acceleration = a1 * (time ** 2) + a2 * (time)
            acceleration_waypoints.append(np.array(acceleration))
        half_time = time_to_travel / 2
        vmax = (a1/3) * (half_time ** 3) + (a2/2) * ( half_time ** 2)
        x_t_half = initial_pos + (a1/12) * (half_time ** 4) + (a2 / 6) * (half_time ** 3)
        if( time > time_to_travel/2  and time <= time_to_travel) : 
            position = 2 * x_t_half - initial_pos + 2 * vmax * (time - half_time) - ((a1/12) * (time ** 4) + (a2 / 6) * (time ** 3))
            captured_waypoints.append(np.array(position))
            velocity = 2 * vmax - ((a1/3) * (time ** 3) + (a2/2) * (time ** 2))
            velocity_waypoints.append(np.array(velocity))
            acceleration = -1 * (a1 * (time ** 2) + a2 * (time))
            acceleration_waypoints.append(np.array(acceleration))
        waypoints_index = waypoints_index + 1
    return captured_waypoints, velocity_waypoints, acceleration_waypoints



def interpolate_trajectory(waypoints, velocities, accelerations, num_samples):
    times = np.linspace(0, 1, len(waypoints))
    interp_pos = scipy.interpolate.interp1d(times, waypoints, kind='cubic', axis=0)
    interp_vel = scipy.interpolate.interp1d(times, velocities, kind='cubic', axis=0)
    interp_acc = scipy.interpolate.interp1d(times, accelerations, kind='cubic', axis=0)
    sample_times = np.linspace(0, 1, num_samples)
    interp_positions = interp_pos(sample_times)
    interp_velocities = interp_vel(sample_times)
    interp_accelerations = interp_acc(sample_times)
    return interp_positions, interp_velocities, interp_accelerations

def calculate_jacobian(model , data , joint_angles , actuator_ids, dof_ids, site_id) :
    jac = np.zeros((6, model.nv))
    time_start = time.time()
    # while (True) : 
    #     step_start = time.time()
    #     data.ctrl[ : 7] = joint_angles[0 : 7]
    #     mujoco.mj_step(model, data)
    #     # viewer.sync()
        

    #     time_until_next_step = dt - (time.time() - step_start)
    #     if time_until_next_step > 0:
    #         time.sleep(time_until_next_step)
    #     if(time.time() - time_start > 1) : 
    #         break
    data.qpos[ : 7] = joint_angles
    mujoco.mj_step(model,data)
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
    jac1 = jac[:,:7]
    print("Jacobian : ", jac1) 
    return jac1


def main() -> None:
    # global dt
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."
    model = mujoco.MjModel.from_xml_path("/Users/sasivellanki/Desktop/xArm-Python-SDK-master/example/wrapper/common/mjctrl/ufactory_xarm7/xarm7_nohand.xml")
    data = mujoco.MjData(model)
    
    # Enable gravity compensation. Set to 0.0 to disable.
    model.body_gravcomp[:17] = float(gravity_compensation)
    model.opt.timestep = dt

    # End-effector site we wish to control.
    site_name = "attachment_site"
    site_id = model.site(site_name).id
    print("site_id: ", site_id)

    # Get the dof and actuator ids for the joints we wish to control. These are copied
    # from the XML file. Feel free to comment out some joints to see the effect on
    # the controller.
    joint_names = [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "joint7",
    ]
    dof_ids = np.array([model.joint(name).id for name in joint_names])
    actuator_ids = np.array([model.actuator(name).id for name in joint_names])
    print("dof_ids: ", dof_ids)

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = arm.get_servo_angle(is_radian=True)[1]
    # q0 = model.key(key_name).qpos
    print("Shape of servo angles : ", q0)
    #only till joints
    q01 = q0[:7]
    print("keyid:", key_id)
    print("qo:", q0)
    # Mocap body we will control with our mouse.
    # mocap_name = "target"
    # mocap_id = model.body(mocap_name).mocapid[0]
    # Define waypoints
    waypoints = [
    np.array([450, 0 , 50])  # Initial position (or another suitable starting point)
    # np.array([0.35 , 0.5 , 0.32]),
    # np.array([0.4, -0.5, 0.32]),
    # np.array([0.4,0 , 0.32]),
    # np.array([0.5, 0.3 , 0.32]),
    # np.array([0.4,0 , 0.3]),
    # np.array([0.5, 0 , 0.3]),
    # np.array([0.6, 0 , 0.3])
    ]
    current_waypoint_index = 0

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(7)
    eye = np.eye(7)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    Kvel= 0.5
    # target_velocity = np.array([2.0, 0.0, 0.0])
    acceleration = 10
    deceleration = -10
    # Tolerance = 0.01

    # Data collection
    time_data = []
    waypoint_data = [wp[0] for wp in waypoints]
    position_data_in_x = []
    position_data_in_y = []
    position_data_in_z = []
    velocity_data_in_x = []
    velocity_data_in_y = []
    velocity_data_in_z = []
    distance_data = []
    target_data_in_x = []
    target_data_in_y = []
    target_data_in_z = []

    previous_error = np.zeros(3)
    previous_time = time.time()
    desired_velocity = np.zeros(3)
    time_track = time.time()
    trajectory_flag = 1
    num_samples = 10
    sub_waypoint_index = 0
    while True : 
        step_start = time.time()
        # arm.set_mode(0)
        position_plus_orient = arm.get_position(is_radian=True)
        joint_states = arm.get_joint_states(is_radian=True,num = 3)
        current_joint_angles = joint_states[1][0]
        if(trajectory_flag == 1) : 
            initial_position = np.array(position_plus_orient[1][ : 3])
            final_position = waypoints[current_waypoint_index]
            time_to_travel = 100

            captured_waypoints , velocity_waypoints, acceleration_waypoints = trajectory(initial_position, final_position , time_to_travel)
            interp_positions, interp_velocities, interp_accelerations = interpolate_trajectory(captured_waypoints, velocity_waypoints, acceleration_waypoints, num_samples)
            trajectory_flag = 0
        print("WAYPOINTS : ", interp_positions)
        current_waypoint = np.array(interp_positions[sub_waypoint_index]) * 1000
        dx = (current_waypoint - position_plus_orient[1][:3])
        dx_main_waypoint = waypoints[current_waypoint_index] - position_plus_orient[1][ : 3]
        
        position = position_plus_orient[1][:3]
        print("POSITION : ", position)
        if  current_waypoint_index <= len(waypoints) - 1 :  # Tolerance of 1 cm to consider as reached
            Tolerance = 0.5
            if(np.linalg.norm(dx_main_waypoint) <= Tolerance) : 
                current_waypoint_index += 1
                trajectory_flag = 1
                sub_waypoint_index = 0

        if current_waypoint_index >= len(waypoints):
            time.sleep(1)
            arm.disconnect()

        if sub_waypoint_index <= num_samples - 1 : 
            Tolerance = 0.5
            if(np.linalg.norm(dx) <= Tolerance) : 
                sub_waypoint_index += 1

            

            
            

            print("Reached waypoint", current_waypoint_index)
               

        # print("waypoint : ",current_waypoint_index , "Distance on x : ", position_in_x)

        distance_to_waypoint = np.linalg.norm(dx)
        current_time = time.time()
        dt_1 = current_time - previous_time
        error = dx
        current_joint_velocities = joint_states[1][1]
        # print("Current joint velocities : ", current_joint_velocities)
        jac1 = np.array(calculate_jacobian(model, data, current_joint_angles, actuator_ids, dof_ids, site_id)) * 1000
        # print("JAC : ", jac)
        # jac1 = jac[:, :7]
        current_cartesian_velocities = np.array(jac1) @ (np.array(current_joint_velocities))
        print("Current cartesian velocity of end eff : ", current_cartesian_velocities)
        desired_velocity = np.array(interp_velocities[sub_waypoint_index]) * 1000 
        target_velocity =  desired_velocity + Kp * error
        print("TARGET VELOCITY = ", target_velocity) 
        # target_velocity = Kp * error + desired_velocity
        previous_error = error
        previous_time = current_time
        twist[:3] = Kpos * target_velocity / integration_dt
        # mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
        # mujoco.mju_negQuat(site_quat_conj, site_quat)
        # mujoco.mju_mulQuat(error_quat, data.site(site_id), site_quat_conj)
        # mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
        # twist[3:] *= Kori / integration_dt

        dq = np.linalg.solve(jac1.T @ jac1 + diag, jac1.T @ twist)

        dq += (eye - np.linalg.pinv(jac1) @ jac1) @ (Kn * (np.array(q0[ : 7]) - np.array(joint_states[1][0])))
        
        # Clamp maximum joint velocity.
        dq_abs_max = np.abs(dq).max()
        if dq_abs_max > max_angvel:
            dq *= max_angvel / dq_abs_max

        q = np.array((arm.get_joint_states(is_radian=True , num = 3))[1][0])
        # dq1 = np.append(dq, np.zeros(12))
        # q = np.append(q, np.zeros(13))
        mujoco.mj_integratePos(model, q, dq, integration_dt)
        np.clip(q[:14], *model.jnt_range.T, out=q[:14])
        print("DQ1 : ", dq)
        # data.ctrl[actuator_ids] = dq1[dof_ids]     # to uncomment this you need to enable velcoity actuators in xml file.
        
        arm.set_mode(4)
        arm.set_state(0)
        # arm.motion_enable(enable=True)
        # time.sleep(1)
        arm.vc_set_joint_velocity(dq[0 : 7],is_radian=True, duration = -1)
        # time.sleep(3)
        # arm.set_mode(0)
        time.sleep(0.001)

        # if(sub_waypoint_index < num_samples - 1) : 
        #     sub_waypoint_index += 1
        
        # time_until_next_step = dt - (time.time() - step_start)
        # if time_until_next_step > 0:
        #     time.sleep(time_until_next_step)

        # if(time.time() - time_track >= 20) : 
        #     break
        #     arm.disconnect()


if __name__ == "__main__":
    main()