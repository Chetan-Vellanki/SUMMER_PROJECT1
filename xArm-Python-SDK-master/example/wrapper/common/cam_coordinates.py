#!/usr/bin/env python

'''
Welcome to the ArUco Marker Pose Estimator!

This program:
  - Estimates the pose of an ArUco Marker
'''

from __future__ import print_function # Python 2/3 compatibility
import cv2 # Import the OpenCV library
import numpy as np # Import Numpy library
from scipy.spatial.transform import Rotation as R
import math # Math library
import pickle # For loading camera calibration parameters

# Project: ArUco Marker Pose Estimator
# Date created: 12/21/2021
# Python version: 3.8

# Dictionary that was used to generate the ArUco marker
aruco_dictionary_name = "DICT_4X4_1000"

# The different ArUco dictionaries built into the OpenCV library.

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}

# Side length of the ArUco marker in meters
aruco_marker_side_length = 0.067

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)

    return roll_x, pitch_y, yaw_z # in radians

def main():
    """
    Main method of the program.
    """
    # Check that we have a valid ArUco marker
    if ARUCO_DICT.get(aruco_dictionary_name, None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(aruco_dictionary_name))
        sys.exit(0)

    # Load the ArUco dictionary
    print("[INFO] detecting '{}' markers...".format(aruco_dictionary_name))
    this_aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_dictionary_name])
    this_aruco_parameters = cv2.aruco.DetectorParameters()

    # Load the camera matrix and distortion coefficients from the pickle files
    with open('cameraMatrix.pkl', 'rb') as f:
        mtx = pickle.load(f)

    with open('dist.pkl', 'rb') as f:
        dst = pickle.load(f)

    # Start the video stream
    cap = cv2.VideoCapture(6)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect ArUco markers in the video frame
        corners, marker_ids, rejected = cv2.aruco.detectMarkers(frame, this_aruco_dictionary, parameters=this_aruco_parameters)

        # Check that at least one ArUco marker was detected
        if marker_ids is not None:
            # Draw a square around detected markers in the video frame
            cv2.aruco.drawDetectedMarkers(frame, corners, marker_ids)

            # Get the rotation and translation vectors
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, aruco_marker_side_length, mtx, dst)

            # Print the pose for the ArUco marker
            for i, marker_id in enumerate(marker_ids):
                # Calculate the center of the ArUco marker in image pixels
                center = np.mean(corners[i][0], axis=0)
                center_x, center_y = center[0], center[1]
                print(f"Marker ID: {marker_id[0]}")
                print(f"Center in pixels: ({center_x:.2f}, {center_y:.2f})")
                if(marker_id == 0) :
                    base_x = center_x + 61.134
                    base_y = center_y 
                    print(f"Base in pixels :({base_x: .2f}, {base_y:.2f})")
                if(marker_id == 200) :
                    ocenter_x = center_x - base_x 
                    ocenter_y = center_y - base_y
                    print(f"Base in pixels :({ocenter_x: .2f}, {ocenter_y:.2f})")

                # Store the translation (i.e. position) information
                transform_translation_x = tvecs[i][0][0]
                transform_translation_y = tvecs[i][0][1]
                transform_translation_z = tvecs[i][0][2]

                # Store the rotation information
                rotation_matrix = np.eye(4)
                rotation_matrix[0:3, 0:3] = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
                r = R.from_matrix(rotation_matrix[0:3, 0:3])
                quat = r.as_quat()   

                # Quaternion format     
                transform_rotation_x = quat[0] 
                transform_rotation_y = quat[1] 
                transform_rotation_z = quat[2] 
                transform_rotation_w = quat[3] 

                # Euler angle format in radians
                roll_x, pitch_y, yaw_z = euler_from_quaternion(transform_rotation_x, 
                                                               transform_rotation_y, 
                                                               transform_rotation_z, 
                                                               transform_rotation_w)

                roll_x = math.degrees(roll_x)
                pitch_y = math.degrees(pitch_y)
                yaw_z = math.degrees(yaw_z)
                # print("transform_translation_x: {}".format(transform_translation_x))
                # print("transform_translation_y: {}".format(transform_translation_y))
                # print("transform_translation_z: {}".format(transform_translation_z))
                # print("roll_x: {}".format(roll_x))
                # print("pitch_y: {}".format(pitch_y))
                # print("yaw_z: {}".format(yaw_z))
                print()

                # Draw the axes on the marker
                cv2.drawFrameAxes(frame, mtx, dst, rvecs[i], tvecs[i], 0.05)

        frame_resized = cv2.resize(frame, (1920, 1080)) # Adjust the size as needed

        # Display the resulting frame
        cv2.imshow('frame', frame_resized)

        # If "q" is pressed on the keyboard, exit this loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close down the video stream
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print(__doc__)
    main()