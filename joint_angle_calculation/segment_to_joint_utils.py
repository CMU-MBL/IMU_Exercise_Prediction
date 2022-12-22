# name: segment_to_joint_utils.py
# description: customized functions for the calculation
# author: Vu Phan
# date: 2022/12/10

import math

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

from segment_to_joint_init import * 


# Load data 
def load_df(path):
	''' tbd '''

	df = pd.read_csv(path)
	# print(df)
	df = df.iloc[:, 1:] # remove the first column (i.e., index column)

	names 	= list(df.columns)
	names 	= [name.split('.')[0] for name in names]
	names_2 = df.iloc[0, :]
	names_3 = df.iloc[1, :]

	for i in range(len(names)):
		names[i] = names[i] + ' ' + names_2[i] + ' ' + names_3[i]

	df 			= df.iloc[2:, :] # remove the first 2 rows
	df.columns 	= names # update new column's names

	return df

# Keep considered data
def slice_df(df):
	''' tbd '''

	cols 		= sorted(df.columns)
	req_cols 	= [col for col in cols if col.split(' ')[0] in CONSIDERED_SEGMENTS] # body segment
	req_cols 	= [col for col in req_cols if col.split(' ')[1] in CONSIDERED_INFO] # data info, e.g., orientation
	df 			= df.loc[:, req_cols]

	return df

# Re-format data before exporting
# tbd

# Get rotation matrix (from the global to local coordinate)
def rot_x(theta):
	theta = np.deg2rad(theta) # convert to radian
	Rx = np.array([[1, 0, 0], [0, np.cos(theta), np.sin(theta)], [0, -np.sin(theta), np.cos(theta)]])

	return Rx

def rot_y(theta):
	theta = np.deg2rad(theta) # convert to radian
	Ry = np.array([[np.cos(theta), 0, -np.sin(theta)], [0, 1, 0], [np.sin(theta), 0, np.cos(theta)]])

	return Ry 

def rot_z(theta):
	theta = np.deg2rad(theta) # convert to radian
	Rz = np.array([[np.cos(theta), np.sin(theta), 0], [-np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

	return Rz

def get_rot(xyz_rot):
	''' tbd '''

	theta_x = xyz_rot[0]
	theta_y = xyz_rot[1]
	theta_z = xyz_rot[2]

	Rx = rot_x(theta_x)
	Ry = rot_y(theta_y)
	Rz = rot_z(theta_z)

	R = np.dot(Rz, Ry)
	R = np.dot(R, Rx)

	return R

# Get angles from the rotation matrix
def get_angle_from_rot_mat(R):
	''' tbd '''

	# print(type(R))

	angle_x = math.atan2(-R[2, 1], R[2, 2])
	# print(angle_x)
	angle_x = np.rad2deg(angle_x)
	angle_y = math.atan2(R[2, 0], np.sqrt(R[0, 0]**2 + R[0, 1]**2))
	angle_y = np.rad2deg(angle_y)
	angle_z = math.atan2(-R[1, 0], R[0, 0])
	angle_z = np.rad2deg(angle_z)

	# angle_y = math.asin(R[2, 0])	
	# angle_x = math.asin(-(R[2, 1]/np.cos(angle_y)))
	# angle_z = math.asin(-(R[1, 0]/np.cos(angle_y)))
	# angle_y = np.rad2deg(angle_y)
	# angle_x = np.rad2deg(angle_x)
	# angle_z = np.rad2deg(angle_z)

	return angle_x, angle_y, angle_z

# Get joint name from joint code
def get_joint_name(joint_code):
	''' tbd '''

	joint_name = None

	if joint_code == RIGHT_HIP:
		joint_name = 'RightHip'
	elif joint_code == RIGHT_KNEE:
		joint_name = 'RightKnee'
	elif joint_code == RIGHT_ANKLE:
		joint_name = 'RightAnkle'
	elif joint_code == LEFT_HIP:
		joint_name = 'LeftHip'
	elif joint_code == LEFT_KNEE:
		joint_name = 'LeftKnee'
	elif joint_code == LEFT_ANKLE:
		joint_name = 'LeftAnkle'
	else:
		pass # do nothing

	joint_name_arr = [joint_name + ' X', joint_name + ' Y', joint_name + ' Z']

	return joint_name_arr

# Get joint angles
def get_joint_angles(df, joint_code):
	''' tbd '''

	x_angles = [] # flexion
	y_angles = [] # ab/aduction
	z_angles = [] # internal/external

	id_1 = int((joint_code/10) - 1)
	id_2 = int((joint_code%10) - 1)

	segment_1 = JOINT_TRAJECTORY[id_1][id_2] # proximal segment
	segment_2 = JOINT_TRAJECTORY[id_1][id_2 + 1] # distal segment
	# print(segment_1 + ' ' + segment_2)

	cols 				= sorted(df.columns)
	cols_1 				= [col for col in cols if col.split(' ')[0] in segment_1]
	cols_2 				= [col for col in cols if col.split(' ')[0] in segment_2]
	segment_1_angles 	= df.loc[:, cols_1]
	# print(segment_1_angles)
	segment_1_angles 	= segment_1_angles.to_numpy(dtype = np.float)
	# print(segment_1_angles)
	# print(segment_1_angles.shape)
	segment_2_angles 	= df.loc[:, cols_2]
	# print(segment_2_angles)
	segment_2_angles 	= segment_2_angles.to_numpy(dtype = np.float)
	# print(segment_2_angles)
	# print(segment_2_angles.shape)

	num_samples = segment_1_angles.shape[0]
	for i in range(num_samples):
		xyz_rot_1 = segment_1_angles[i, :]
		xyz_rot_2 = segment_2_angles[i, :]

		rot_1 		= get_rot(xyz_rot_1)
		# print(rot_1)
		rot_2 		= get_rot(xyz_rot_2)
		# if (joint_code == RIGHT_ANKLE) or (joint_code == LEFT_ANKLE):
		# 	Rx_90 = rot_x(-90)
		# 	rot_2 = np.dot(rot_2, Rx_90)

		rot_joint 	= np.dot(rot_2, rot_1.T)
		# print(rot_joint)

		angle_x, angle_y, angle_z = get_angle_from_rot_mat(rot_joint)
		# print(angle_x)		

		x_angles.append(angle_x)
		y_angles.append(angle_y)
		z_angles.append(angle_z)

	joint_angles = [x_angles, y_angles, z_angles]
	joint_angles = np.array(joint_angles)
	joint_angles = joint_angles.T # (num_sampes x 3)
	# print(joint_angles)

	joint_name = get_joint_name(joint_code)

	return joint_name, joint_angles







