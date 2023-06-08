# name: segment_to_joint_angle.py
# description: Compute joint angles from segment angles (i.e., orientations)
# author: Vu Phan
# date: 2022/12/10

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os

from tqdm import tqdm

from segment_to_joint_utils import *


''' Read segmental orientations '''
# Subject folders
subs = sorted(list(os.listdir(SEGMENT_PATH))) # subjects
subs = subs[1:] # remove subject 1 from the calculation
# print('No. of subjects = ' + str(len(subs)))
# print(subs)

# Exercise folders
exercises 		= [sorted(os.listdir(SEGMENT_PATH + '\\' + sub)) for sub in subs] # exercises
exercise_types 	= [] # store exercise types
for ex in exercises:
	exercise_types.extend(ex)
	# print(ex)
exercise_types 		= np.array(exercise_types)
exercise_types, _ 	= np.unique(exercise_types, return_index=True)
exercise_types 		= exercise_types.tolist()
# print('No. of exercise types = ' + str(len(exercise_types)))
# print(exercise_types)

# Label exercises, e.g., 0 is BulgSq, ..., 36 is Walk
num_exercise 	= len(exercise_types) # get number of exercises
exercise_code 	= list(range(0, num_exercise))
label_code 		= dict(zip(exercise_types, exercise_code))
code_label 		= dict(zip(exercise_code, exercise_types))


''' Start the calculation '''
print('* Starting the calculation...\n')
sub_cnt = 0

# Loop through subjects
for sub in subs[:]:

	sub_cnt += 1 # subject counter
	print('** ' + str(sub_cnt) + '. Working on ' + sub + '...')

	# Loop through exercises
	for ex in tqdm(exercise_types):
		
		# Try the calculation if the subject performed the exercise
		try:
			# Get file path
			in_file_path 	= SEGMENT_PATH + '\\' + sub + '\\' + ex # input segmental angles (i.e., orientations)
			out_file_path	= JOINT_PATH + '\\' + sub + '\\' + ex # output joint angles
			filenames 		= os.listdir(in_file_path) # obtain all parsed data files

			# Create a folder for outputs if it does not exist
			is_exist = os.path.exists(out_file_path)
			if not is_exist:
				os.makedirs(out_file_path)

			# Perform calculation for all files
			for fn in filenames:
				# Get segmental data from a rep
				rep_path = in_file_path + '\\' + fn # path of a rep
				df = load_df(rep_path)
				df = slice_df(df)
				# print(df)

				# Calculate joint angles
				right_hip_name, angle_right_hip 	= get_joint_angles(df, RIGHT_HIP)				
				right_knee_name, angle_right_knee 	= get_joint_angles(df, RIGHT_KNEE)
				right_ankle_name, angle_right_ankle = get_joint_angles(df, RIGHT_ANKLE)
				left_hip_name, angle_left_hip 		= get_joint_angles(df, LEFT_HIP)
				left_knee_name, angle_left_knee 	= get_joint_angles(df, LEFT_KNEE)
				left_ankle_name, angle_left_ankle 	= get_joint_angles(df, LEFT_ANKLE)		

				# Frame the data frame
				name_col	= right_hip_name + right_knee_name + right_ankle_name + left_hip_name + left_knee_name + left_ankle_name
				angle_data 	= np.concatenate([angle_right_hip, angle_right_knee, angle_right_ankle, angle_left_hip, angle_left_knee, angle_left_ankle], axis = 1)
				out_df 		= pd.DataFrame(angle_data, columns = name_col)
				# print(out_df)

				# Export the data
				out_df.to_csv(out_file_path + '\\' + fn)
				# print('done')

				# break # FOR DEBUGGING

		except:
			pass # do nothing if this exercise was not performed by the subject

		# break # FOR DEBUGGING

	print('--> Finished!!!')








