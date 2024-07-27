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

# Cartesian impedance control gains.
impedance_pos = np.asarray([100, 100, 100])  # [N/m]
impedance_ori = np.asarray([10, 10, 10])  # [Nm/rad]

# Joint impedance control gains.
Kp_null = np.asarray([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

# Damping ratio for both Cartesian and joint impedance control.
damping_ratio = 2

# Gains for the twist computation. These should be between 0 and 1. 0 means no
# movement, 1 means move the end-effector to the target in one integration step.
Kpos: float = 0.95

# Gain for the orientation component of the twist computation. This should be
# between 0 and 1. 0 means no movement, 1 means move the end-effector to the target
# orientation in one integration step.
Kori: float = 0.95

# Integration timestep in seconds.
integration_dt: float = 3

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Simulation timestep in seconds.
dt: float = 0.02

# Maximum allowable joint velocity in rad/s.
max_angvel = 0.785

def convert_joint_torques_to_joint_Velocities(model , data , joint_angles , actuator_ids, dof_ids, site_id, gen_torques, joint_acc, curr_joint_vel,M_inv ) :
    jac = np.zeros((6, model.nv))
    time_start = time.time()
    while (True) : 
        step_start = time.time()
        data.ctrl[ : 7] = joint_angles[0 : 7]
        mujoco.mj_step(model, data)
        # viewer.sync()
        

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        if(time.time() - time_start > 1) : 
            break

    while(True) : 
        step_start = time.time()
        data.ctrl[ : 7] = curr_joint_vel[ : 7]

        mujoco.mj_step(model, data)
        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        if(time.time() - time_start > 1) :
            M_inv = M_inv[ : 7, :7]
            if abs(np.linalg.det(M_inv)) >= 1e-2:
                M = np.linalg.inv(M_inv)
            else:
                M = np.linalg.pinv(M_inv, rcond=1e-2) 
            C = np.zeros(model.nv)
            mujoco.mj_rne(model, data, 0, C)
            # if abs(np.linalg.det(C[ : 7])) >= 1e-2:
            #     C_inv = np.linalg.inv(C[ : 7])
            # else:
            #     C_inv = np.linalg.pinv(C[ : 7], rcond=1e-2) 

            
            # new_joint_velocities = C_inv @ (gen_torques - (M @ joint_acc))
            new_joint_velocities = C[ : 7] * (gen_torques - (M @ joint_acc))
            break

    return new_joint_velocities        


        

    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
    mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
    jac1 = jac[:,:7]
    print("Jacobian : ", jac1) 
    return jac1, M_inv


def calculate_jacobian_and_mass_matrix(model , data , joint_angles , actuator_ids, dof_ids, site_id) :
    M_inv = np.zeros((model.nv, model.nv))
    jac = np.zeros((6, model.nv))
    time_start = time.time()
    while (True) : 
        step_start = time.time()
        data.ctrl[ : 7] = joint_angles[0 : 7]
        mujoco.mj_step(model, data)
        # viewer.sync()
        

        time_until_next_step = dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
        if(time.time() - time_start > 1) : 
            break

    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
    mujoco.mj_solveM(model, data, M_inv, np.eye(model.nv))
    jac1 = jac[:,:7]
    print("Jacobian : ", jac1) 
    return jac1, M_inv

def main() -> None:
    assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."

    # Load the model and data.
    model = mujoco.MjModel.from_xml_path("/Users/sasivellanki/Desktop/xArm-Python-SDK-master/example/wrapper/common/mjctrl/ufactory_xarm7/scene_strike_act.xml")
    data = mujoco.MjData(model)

    model.opt.timestep = dt

    # Compute damping and stiffness matrices.
    damping_pos = damping_ratio * 2 * np.sqrt(impedance_pos)
    damping_ori = damping_ratio * 2 * np.sqrt(impedance_ori)
    Kp = np.concatenate([impedance_pos, impedance_ori], axis=0)
    Kd = np.concatenate([damping_pos, damping_ori], axis=0)
    Kd_null = damping_ratio * 2 * np.sqrt(Kp_null)



    # End-effector site we wish to control.
    site_name = "link_tcp"
    site_id = model.site(site_name).id

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

    # Initial joint configuration saved as a keyframe in the XML file.
    key_name = "home"
    key_id = model.key(key_name).id
    q0 = np.array(arm.get_servo_angle(is_radian=True)[1])
    
    waypoints = [
    np.array([450, 0, 120.5])  # Initial position (or another suitable starting point)
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
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    M_inv = np.zeros((model.nv, model.nv))
    Mx = np.zeros((6, 6))

    previous_error = np.zeros(3)
    previous_time = time.time()
    desired_velocity = np.array([20,0,0])
    time_track = time.time()
    current_joint_accelerations = np.zeros(7)
    previous_joint_velocities = np.zeros(7)

    while True : 
        step_start = time.time()
        # arm.set_mode(0)
        position_plus_orient = arm.get_position(is_radian=True)
        joint_states = arm.get_joint_states(is_radian=True,num = 3)
        current_joint_angles = np.array(joint_states[1][0])


        current_waypoint = waypoints[current_waypoint_index]
        dx = (current_waypoint - position_plus_orient[1][:3])
        
        position = position_plus_orient[1][:3]
        print("POSITION : ", position)
        if  current_waypoint_index <= len(waypoints) - 1 :  # Tolerance of 1 cm to consider as reached
                Tolerance = 0.5
                if(np.linalg.norm(dx) <= Tolerance) : 
                    current_waypoint_index += 1

        # if current_waypoint_index >= len(waypoints):
        #     time.sleep(1)
        #     arm.disconnect()

        distance_to_waypoint = np.linalg.norm(dx)
        
        error = dx
        current_joint_velocities = np.array(joint_states[1][1])
        jac, M_inv_func = calculate_jacobian_and_mass_matrix(model, data, current_joint_angles, actuator_ids, dof_ids, site_id)
        jac1 = np.array(jac) * 1000
        M_inv = np.array(M_inv_func) * 1000000
        current_cartesian_velocities = np.array(jac1) @ (np.array(current_joint_velocities))
        twist[:3] = Kpos * dx / integration_dt
        # mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
        # mujoco.mju_negQuat(site_quat_conj, site_quat)
        # mujoco.mju_mulQuat(error_quat, data.mocap_quat[mocap_id], site_quat_conj)
        # mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
        # twist[3:] *= Kori / integration_dt

        Mx_inv = jac1 @ M_inv[:7,:7] @ jac1.T
        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            Mx = np.linalg.inv(Mx_inv)
        else:
            Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

        tau = jac1.T @ Mx @ (Kp * twist - Kd * (jac1 @ current_joint_velocities))
        # Add joint task in nullspace.
        Jbar = M_inv[:7,:7] @ jac1.T @ Mx
        ddq = Kp_null * (q0[ : 7] - current_joint_angles) - Kd_null * current_joint_velocities
        tau += (np.eye(7) - jac1.T @ Jbar.T) @ ddq

        # # # Add gravity compensation.
        # if gravity_compensation:
        #     tau += data.qfrc_bias[dof_ids]
        np.clip(tau, *model.actuator_ctrlrange[:7].T, out=tau)

        # to give actuator to the hardware
        # to convert this generated torques into the to be generated joint
        # velocities

        # calculating current joint accelerations
        current_time = time.time()
        current_error = current_joint_velocities - previous_joint_velocities
        current_joint_accelerations = current_error/ (current_time - previous_time)
        previous_time = current_time
        previous_error = current_error
        dq = convert_joint_torques_to_joint_Velocities(model , data , current_joint_angles , actuator_ids, dof_ids, site_id, tau, current_joint_accelerations,current_joint_velocities,M_inv)

        # Clamp maximum joint velocity.
        dq_abs_max = np.abs(dq).max()
        if dq_abs_max > max_angvel:
            dq *= max_angvel / dq_abs_max

        q = np.array((arm.get_joint_states(is_radian=True , num = 3))[1][0])
        dq1 = np.append(dq, np.zeros(12))
        q = np.append(q, np.zeros(13))
        mujoco.mj_integratePos(model, q, dq1, integration_dt)
        np.clip(q[:14], *model.jnt_range.T, out=q[:14])

        arm.set_mode(4)
        arm.set_state(0)
        # arm.motion_enable(enable=True)
        # time.sleep(1)
        arm.vc_set_joint_velocity(dq1[0 : 7],is_radian=True, duration = -1)
        # time.sleep(3)
        # arm.set_mode(0)
        time.sleep(1)

if __name__ == "__main__":
    main()







