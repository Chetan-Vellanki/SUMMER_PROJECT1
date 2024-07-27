# #!/usr/bin/env python3
# # Software License Agreement (BSD License)
# #
# # Author: Vinman <vinman.wen@ufactory.cc> <vinman.cub@gmail.com>

# """
# Description: Cartesian velocity control with real-time plotting
# """

# import os
# import sys
# import time
# import mujoco
# import mujoco.viewer
# import numpy as np
# import time
# # from scipy.integrate import simps
# import matplotlib.pyplot as plt

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

# from xarm.wrapper import XArmAPI

# #######################################################
# """
# Just for test example
# """
# if len(sys.argv) >= 2:
#     ip = sys.argv[1]
# else:
#     try:
#         from configparser import ConfigParser
#         parser = ConfigParser()
#         parser.read('../robot.conf')
#         ip = parser.get('xArm', 'ip')
#     except:
#         ip = input('Please input the xArm ip address:')
#         if not ip:
#             print('input error, exit')
#             sys.exit(1)
# ########################################################

# arm = XArmAPI(ip)
# arm.motion_enable(enable=True)
# arm.set_mode(0)
# arm.set_state(state=0)
# time.sleep(1)

# # arm.reset(wait=True)


# # Integration timestep in seconds. This corresponds to the amount of time the joint
# # velocities will be integrated for to obtain the desired joint positions.
# integration_dt: float = 3

# # Damping term for the pseudoinverse. This is used to prevent joint velocities from
# # becoming too large when the Jacobian is close to singular.
# damping: float = 1e-4

# # Gains for the twist computation. These should be between 0 and 1. 0 means no
# # movement, 1 means move the end-effector to the target in one integration step.
# Kpos: float = 0.95
# Kori: float = 0.95

# dt: float = 0.02

# # Nullspace P gain.
# Kn = np.asarray([0.1, 0.01, 0.1, 0.1, 1.0, 1.0, 1.0])

# # Whether to enable gravity compensation.
# gravity_compensation: bool = True

# # Maximum allowable joint velocity in rad/s.
# max_angvel = 0.785

# # PD control gains
# Kp = 0.6  # Proportional gain
# # Kp = 0.1
# Kd = 1  # Derivative gain

# def calculate_jacobian(model , data , joint_angles , actuator_ids, dof_ids, site_id) :
#     jac = np.zeros((6, model.nv))
#     time_start = time.time()
#     # while (True) : 
#     #     step_start = time.time()
#     #     data.ctrl[ : 7] = joint_angles[0 : 7]
#     #     mujoco.mj_step(model, data)
#     #     # viewer.sync()
        

#     #     time_until_next_step = dt - (time.time() - step_start)
#     #     if time_until_next_step > 0:
#     #         time.sleep(time_until_next_step)
#     #     if(time.time() - time_start > 1) : 
#     #         break
#     data.qpos[ : 7] = joint_angles
#     mujoco.mj_step(model,data)
#     mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
#     jac1 = jac[:,:7]
#     print("Jacobian : ", jac1) 
#     return jac1


# def main() -> None:
#     # global dt
#     assert mujoco.__version__ >= "3.1.0", "Please upgrade to mujoco 3.1.0 or later."
#     model = mujoco.MjModel.from_xml_path("/Users/sasivellanki/Desktop/xArm-Python-SDK-master/example/wrapper/common/mjctrl/ufactory_xarm7/xarm7_nohand.xml")
#     data = mujoco.MjData(model)
    
#     # Enable gravity compensation. Set to 0.0 to disable.
#     model.body_gravcomp[:17] = float(gravity_compensation)
#     model.opt.timestep = dt

#     # End-effector site we wish to control.
#     site_name = "attachment_site"
#     site_id = model.site(site_name).id
#     print("site_id: ", site_id)

#     # Get the dof and actuator ids for the joints we wish to control. These are copied
#     # from the XML file. Feel free to comment out some joints to see the effect on
#     # the controller.
#     joint_names = [
#         "joint1",
#         "joint2",
#         "joint3",
#         "joint4",
#         "joint5",
#         "joint6",
#         "joint7",
#     ]
#     dof_ids = np.array([model.joint(name).id for name in joint_names])
#     actuator_ids = np.array([model.actuator(name).id for name in joint_names])
#     print("dof_ids: ", dof_ids)

#     # Initial joint configuration saved as a keyframe in the XML file.
#     key_name = "home"
#     key_id = model.key(key_name).id
#     q0 = arm.get_servo_angle(is_radian=True)[1]
#     # q0 = model.key(key_name).qpos
#     print("Shape of servo angles : ", q0)
#     #only till joints
#     q01 = q0[:7]
#     print("keyid:", key_id)
#     print("qo:", q0)
#     # Mocap body we will control with our mouse.
#     # mocap_name = "target"
#     # mocap_id = model.body(mocap_name).mocapid[0]
#     # Define waypoints
#     waypoints = [
#     np.array([480.6, -133.1 , 350])  # Initial position (or another suitable starting point)
#     # np.array([0.35 , 0.5 , 0.32]),
#     # np.array([0.4, -0.5, 0.32]),
#     # np.array([0.4,0 , 0.32]),
#     # np.array([0.5, 0.3 , 0.32]),
#     # np.array([0.4,0 , 0.3]),
#     # np.array([0.5, 0 , 0.3]),
#     # np.array([0.6, 0 , 0.3])
#     ]
#     current_waypoint_index = 0

#     # Pre-allocate numpy arrays.
#     jac = np.zeros((6, model.nv))
#     diag = damping * np.eye(7)
#     eye = np.eye(7)
#     twist = np.zeros(6)
#     site_quat = np.zeros(4)
#     site_quat_conj = np.zeros(4)
#     error_quat = np.zeros(4)
#     Kvel= 0.5
#     # target_velocity = np.array([2.0, 0.0, 0.0])
#     acceleration = 10
#     deceleration = -10
#     # Tolerance = 0.01

#     # Data collection
#     time_data = []
#     waypoint_data = [wp[0] for wp in waypoints]
#     position_data_in_x = []
#     position_data_in_y = []
#     position_data_in_z = []
#     velocity_data_in_x = []
#     velocity_data_in_y = []
#     velocity_data_in_z = []
#     distance_data = []
#     target_data_in_x = []
#     target_data_in_y = []
#     target_data_in_z = []

#     previous_error = np.zeros(3)
#     previous_time = time.time()
#     desired_velocity = np.zeros(3)
#     time_track = time.time()

#     while True : 
#         step_start = time.time()
#         # arm.set_mode(0)
#         position_plus_orient = arm.get_position(is_radian=True)
#         joint_states = arm.get_joint_states(is_radian=True,num = 3)
#         current_joint_angles = joint_states[1][0]


#         current_waypoint = waypoints[current_waypoint_index]
#         dx = (current_waypoint - position_plus_orient[1][:3])
        
#         position = position_plus_orient[1][:3]
#         print("POSITION : ", position)
#         if  current_waypoint_index <= len(waypoints) - 1 :  # Tolerance of 1 cm to consider as reached
#                 Tolerance = 0.5
#                 if(np.linalg.norm(dx) <= Tolerance) : 
#                     current_waypoint_index += 1

#         if current_waypoint_index >= len(waypoints):
#             time.sleep(1)
#             arm.disconnect()
            

            
            

#             print("Reached waypoint", current_waypoint_index)
               

#         # print("waypoint : ",current_waypoint_index , "Distance on x : ", position_in_x)

#         distance_to_waypoint = np.linalg.norm(dx)
#         current_time = time.time()
#         dt_1 = current_time - previous_time
#         error = dx
#         current_joint_velocities = joint_states[1][1]
#         # print("Current joint velocities : ", current_joint_velocities)
#         jac1 = np.array(calculate_jacobian(model, data, current_joint_angles, actuator_ids, dof_ids, site_id)) * 1000
#         # print("JAC : ", jac)
#         # jac1 = jac[:, :7]
#         current_cartesian_velocities = np.array(jac1) @ (np.array(current_joint_velocities))
#         print("Current cartesian velocity of end eff : ", current_cartesian_velocities)
#         target_velocity = desired_velocity + Kp * error
#         print("TARGET VELOCITY = ", target_velocity) 
#         # target_velocity = Kp * error + desired_velocity
#         previous_error = error
#         previous_time = current_time
#         twist[:3] = Kpos * target_velocity / integration_dt
#         # mujoco.mju_mat2Quat(site_quat, data.site(site_id).xmat)
#         # mujoco.mju_negQuat(site_quat_conj, site_quat)
#         # mujoco.mju_mulQuat(error_quat, data.site(site_id), site_quat_conj)
#         # mujoco.mju_quat2Vel(twist[3:], error_quat, 1.0)
#         # twist[3:] *= Kori / integration_dt

#         dq = np.linalg.solve(jac1.T @ jac1 + diag, jac1.T @ twist)

#         dq += (eye - np.linalg.pinv(jac1) @ jac1) @ (Kn * (np.array(q0[ : 7]) - np.array(joint_states[1][0])))
        
#         # Clamp maximum joint velocity.
#         dq_abs_max = np.abs(dq).max()
#         if dq_abs_max > max_angvel:
#             dq *= max_angvel / dq_abs_max

#         q = np.array((arm.get_joint_states(is_radian=True , num = 3))[1][0])
#         # dq1 = np.append(dq, np.zeros(12))
#         # q = np.append(q, np.zeros(13))
#         mujoco.mj_integratePos(model, q, dq, integration_dt)
#         np.clip(q[:14], *model.jnt_range.T, out=q[:14])
#         print("DQ1 : ", dq)
#         # data.ctrl[actuator_ids] = dq1[dof_ids]     # to uncomment this you need to enable velcoity actuators in xml file.
        
#         arm.set_mode(4)
#         arm.set_state(0)
#         # arm.motion_enable(enable=True)
#         # time.sleep(1)
#         arm.vc_set_joint_velocity(dq[0 : 7],is_radian=True, duration = -1)
#         # time.sleep(3)
#         # arm.set_mode(0)
#         time.sleep(0.1)
        
#         # time_until_next_step = dt - (time.time() - step_start)
#         # if time_until_next_step > 0:
#         #     time.sleep(time_until_next_step)

#         # if(time.time() - time_track >= 20) : 
#         #     break
#         #     arm.disconnect()


# if __name__ == "__main__":
#     main()




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
Kn = np.asarray([0.1, 0.01, 0.1, 0.1, 1.0, 1.0, 1.0])

# Whether to enable gravity compensation.
gravity_compensation: bool = True

# Maximum allowable joint velocity in rad/s.
max_angvel = 0.785

# PD control gains
Kp = 0.9  # Proportional gain
Kd = 1  # Derivative gain

def calculate_jacobian(model, data, joint_angles, actuator_ids, dof_ids, site_id):
    jac = np.zeros((6, model.nv))
    data.qpos[:7] = joint_angles
    mujoco.mj_step(model, data)
    mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
    jac1 = jac[:, :7]
    print("Jacobian: ", jac1)
    return jac1

def read_waypoints(file_path, bool_for_pick, bool_for_place, waypoints):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        if bool_for_pick and len(lines) > 0:
            pick_coords = lines[0].strip().split(',')
            if len(pick_coords) == 3:
                x = float(pick_coords[0].strip())
                y = float(pick_coords[1].strip())
                z = float(pick_coords[2].strip())
                print("Pick coordinates (x, y, z):", x, y, z)
                waypoints = [x, y, z]
            else:
                print("Error: First line should contain exactly three comma-separated values.")
        
        if bool_for_place and len(lines) > 1:
            place_coords = lines[1].strip().split(',')
            if len(place_coords) == 3:
                x = float(place_coords[0].strip())
                y = float(place_coords[1].strip())
                z = float(place_coords[2].strip())
                print("Place coordinates (x, y, z):", x, y, z)
                waypoints = [x, y, z]
            else:
                print("Error: Second line should contain exactly three comma-separated values.")
    
    return waypoints

def main() -> None:
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
    print("Shape of servo angles: ", q0)
    q01 = q0[:7]
    print("keyid:", key_id)
    print("qo:", q0)

    # Read waypoints from the file
    file_path = os.path.expanduser('~/Downloads/aruco_cam/aruco/ocenter_coordinates.txt')
    waypoints = read_waypoints(file_path, bool_for_pick=0, bool_for_place=0, waypoints=[])
    # read_waypoints(file_path)
    current_waypoint_index = 0

    # Pre-allocate numpy arrays.
    jac = np.zeros((6, model.nv))
    diag = damping * np.eye(7)
    eye = np.eye(7)
    twist = np.zeros(6)
    site_quat = np.zeros(4)
    site_quat_conj = np.zeros(4)
    error_quat = np.zeros(4)
    Kvel = 0.5

    # Data collection
    time_data = []
    # waypoint_data = [wp[0] for wp in waypoints]
    waypoint_data=waypoints
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
    bool_for_pick = 1
    bool_for_place = 0

    while True:
        step_start = time.time()
        position_plus_orient = arm.get_position(is_radian=True)
        joint_states = arm.get_joint_states(is_radian=True, num=3)
        current_joint_angles = joint_states[1][0]
        waypoints=read_waypoints(file_path, bool_for_pick, bool_for_place, waypoints)
        # waypoints = np.array([500, 134, 350])
        print("Waypoint:",waypoints)
        if(bool_for_pick == 1) : 
            current_waypoint = [waypoints[0]-110,waypoints[1],waypoints[2]]

        if(bool_for_place == 1) : 
            current_waypoint = [waypoints[0],waypoints[1],waypoints[2] - 70]
        # current_waypoint=[]
        print("desired Coordinates:",current_waypoint)
        print("pp:",position_plus_orient[1][:3])
        dx = (np.array(current_waypoint) - position_plus_orient[1][:3])
        
        position = position_plus_orient[1][:3]
        print("POSITION: ", position)
        if bool_for_pick == 1 and current_waypoint_index <= len(waypoints) - 1:  # Tolerance of 1 cm to consider as reached
            Tolerance = 1
            if np.linalg.norm(dx) <= Tolerance:
                current_waypoint_index += 1

        if current_waypoint_index >= len(waypoints):
            # time.sleep(1)
            # arm.disconnect()
            
            arm.set_gripper_mode(0)
            arm.set_gripper_enable(True)
            arm.set_gripper_speed(1000)
            code = arm.set_gripper_position(600, wait=True, auto_enable= True)
            print(code)
            print(waypoints)
            current_waypoint = [waypoints[0] - 100 , waypoints[1], 280]
            arm.motion_enable(enable=True)
            arm.set_mode(0)
            arm.set_state(state=0)

            arm.set_position(x=current_waypoint[0], y=current_waypoint[1], z=current_waypoint[2], roll=-180, pitch=0, yaw=0, speed=10, wait=True)
            # time.sleep(3)
            arm.set_gripper_position(450,wait=True, speed = 1000)
            bool_for_pick = 0
            bool_for_place = 1
            
            arm.set_position(x=current_waypoint[0], y=current_waypoint[1], z=current_waypoint[2] + 70 , roll=-180, pitch=0, yaw=0, speed=20, wait=True)
            current_waypoint_index = 0
            print("Reached waypoint", current_waypoint_index)
            # break


        waypoints=read_waypoints(file_path,bool_for_pick,bool_for_place,waypoints)
        if bool_for_place == 1 and current_waypoint_index <= len(waypoints) - 1:  # Tolerance of 1 cm to consider as reached
            Tolerance = 1
            if np.linalg.norm(dx) <= Tolerance:
                current_waypoint_index += 1

        if current_waypoint_index >= len(waypoints):
            # time.sleep(1)
            # arm.disconnect()
            
            
            print(code)
            print(waypoints)
            current_waypoint = [waypoints[0] , waypoints[1], waypoints[2]]
            # arm.motion_enable(enable=True)
            # arm.set_mode(0)
            # arm.set_state(state=0)

            # arm.set_position(x=current_waypoint[0], y=current_waypoint[1], z=current_waypoint[2] - 70, roll=-180, pitch=0, yaw=0, speed=20, wait=True)
            # time.sleep(3)
            # arm.set_gripper_position(450,wait=True, speed = 1000)
            arm.set_gripper_mode(0)
            arm.set_gripper_enable(True)
            arm.set_gripper_speed(1000)
            code = arm.set_gripper_position(600, wait=True, auto_enable= True)
            bool_for_pick = 1
            bool_for_place = 0
            arm.set_mode(0)
            arm.set_state(state=0)
            arm.set_position(x=current_waypoint[0], y=current_waypoint[1], z=current_waypoint[2] + 70, roll=-180, pitch=0, yaw=0, speed=20, wait=True)
            current_waypoint_index = 0

            time.sleep(5)

            # arm.disconnect()
            # break

        

        distance_to_waypoint = np.linalg.norm(dx)
        current_time = time.time()
        dt_1 = current_time - previous_time
        error = dx
        current_joint_velocities = joint_states[1][1]
        jac1 = np.array(calculate_jacobian(model, data, current_joint_angles, actuator_ids, dof_ids, site_id)) * 1000
        current_cartesian_velocities = np.array(jac1) @ (np.array(current_joint_velocities))
        print("Current cartesian velocity of end eff: ", current_cartesian_velocities)
        target_velocity = desired_velocity + Kp * error
        print("TARGET VELOCITY = ", target_velocity)
        previous_error = error
        previous_time = current_time
        twist[:3] = Kpos * target_velocity / integration_dt

        dq = np.linalg.solve(jac1.T @ jac1 + diag, jac1.T @ twist)
        dq += (eye - np.linalg.pinv(jac1) @ jac1) @ (Kn * (np.array(q0[:7]) - np.array(joint_states[1][0])))

        # Clamp maximum joint velocity.
        dq_abs_max = np.abs(dq).max()
        if dq_abs_max > max_angvel:
            dq *= max_angvel / dq_abs_max

        q = np.array((arm.get_joint_states(is_radian=True, num=3))[1][0])
        mujoco.mj_integratePos(model, q, dq, integration_dt)
        np.clip(q[:14], *model.jnt_range.T, out=q[:14])
        print("DQ1: ", dq)

        arm.set_mode(4)
        arm.set_state(0)
        arm.vc_set_joint_velocity(dq[:7], is_radian=True, duration=-1)
        time.sleep(0.1)

if __name__ == "__main__":
    main()
