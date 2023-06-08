# name: knee_flex_verification.py
# description: Verify calculation of right knee flexion angle with Ke
# author: Vu Phan
# date: 2022/12/11

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from segment_to_joint_utils import *


''' Constants '''
SEGMENT_PATH 	= 'parsed_MoCap_kinematics'
JOINT_PATH 		= 'parsed_joint_angles_'

''' Variables '''
sub = 'SUB03'
ex = 'Walk'

''' Main '''
# Get Ke's calculation
file_path_ke = 'test_ke_sub19_lunge_rep1.csv'
df_ke = pd.read_csv(file_path_ke)
df_ke = df_ke.loc[1:, :]
ke_knee_flex = df_ke['RightKneeAngle'].to_numpy(dtype = float)
# print(df_ke)


# Get my calculation
file_path_me = 'test_me_sub19_lunge_rep1.csv'
df_me = pd.read_csv(file_path_me)
me_knee_flex = df_me['RightKnee X'].to_numpy(dtype = float)
# print(df_me)

# Plot
plt.figure()
plt.plot(ke_knee_flex, color = 'b', linestyle = '-', linewidth = 1.8, label = 'Ke\'s reference')
plt.plot(me_knee_flex, color = 'r', linestyle = '--', linewidth = 1.2, label = 'My calculation')
plt.xlim([0, len(me_knee_flex)-1])
plt.xlabel('Time step')
plt.ylabel('Angle (degree)')
plt.legend(frameon=False)
plt.show()

