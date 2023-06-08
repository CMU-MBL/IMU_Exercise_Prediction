# name: segment_to_joint_init.py
# description: Initialize constants
# author: Vu Phan
# date: 2022/12/10


''' tbd '''
CONSIDERED_SEGMENTS = ['LeftFoot', 'LeftShank', 'LeftThigh', 'Pelvis', 'RightFoot', 'RightShank', 'RightThigh']
CONSIDERED_INFO 	= ['Orientation']

''' Code for joint angles '''
JOINT_TRAJECTORY = [['Pelvis', 'RightThigh', 'RightShank', 'RightFoot'], ['Pelvis', 'LeftThigh', 'LeftShank', 'LeftFoot']]
RIGHT_HIP 	= 11
RIGHT_KNEE 	= 12
RIGHT_ANKLE = 13
LEFT_HIP 	= 21
LEFT_KNEE 	= 22
LEFT_ANKLE 	= 23

''' Data storage '''
SEGMENT_PATH 	= 'parsed_MoCap_kinematics'
JOINT_PATH 		= 'parsed_joint_angles_'





