# name: preprocessing.py
# description: preprocess data (e.g., read, add labels, etc.)


import sys
import os
import numpy as np
import pandas as pd
import random

sys.path.append('/path/to/IMU_Exercise_Prediction')

import constants
import config


def mkfolder(pth):
	""" Make a new folder if not exist
	"""

	if not os.path.exists(pth):
		os.mkdir(pth)


def read(pth):
	""" Read data from a given path
	"""

	return pd.read_csv(pth)


def load_df(pth):
	""" Load and re-format data from a given path
	"""

	dtframe = read(pth)
	dtframe = dtframe.iloc[:, 3:] 

	names = list(dtframe.columns)
	names = [name.split('.')[0] for name in names]
	names_2 = dtframe.iloc[0, :]
	names_3 = dtframe.iloc[1, :]

	for i in range(len(names)):
		names[i] = names[i]+' '+names_2[i]+' '+names_3[i]

	dtframe = dtframe.iloc[2:, :]
	dtframe.columns = names

	return dtframe


def slice_df(dtframe):
	""" Slice data
	"""

	cols 		= sorted(dtframe.columns)
	req_cols 	= [col for col in cols if col.split(' ')[0] in config.CONSIDERED_IMU_POSITION]
	req_cols 	= [col for col in req_cols if col.split(' ')[1] not in config.NOT_CONSIDERED_INFO]
	dtframe 	= dtframe.loc[:, req_cols]

	return dtframe


def one_hot_encoding(label, num_clasess):
	""" Apply one-hot encoding to data label
	"""
	encoding = np.zeros(num_clasess)
	encoding[label] = 1

	return encoding


def one_hot_decoding(code):
	""" Decode a code
	"""

	if code.shape[0] > 0:
		label = np.array([np.where(row == 1) for row in code])
	else:
		label = np.argwhere(code == 1)

	return label


def normLength(arr, maxlength):
	""" Normalize data to have the same sample length for the network input
	"""

	new_arr = np.zeros((maxlength, arr.shape[-1]))
	for i in range(arr.shape[-1]):
		a = arr[:, i]
		k = a.shape[0]
		y = np.interp(np.linspace(0, 1, maxlength), np.linspace(0, 1, k), a)
		new_arr[:, i] = y

	return new_arr


def get_cluster_label(ex_code):
	""" Get label for exercise groups
	"""

	cluster_found = False
	cluster_id = 0

	while not cluster_found:
		if ex_code in constants.CLUSTER[cluster_id]:
			cluster_found = True
		else:
			cluster_id += 1

	return cluster_id


def losocv_split_train_list(all_subject_id, test_subject):  
	""" Split data for the LOSOCV scheme
	"""

	train_list = [m for m in all_subject_id if m != test_subject]

	return train_list





