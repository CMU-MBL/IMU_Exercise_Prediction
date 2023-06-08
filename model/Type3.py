# name: Type3.py
# description: models using type 3 block (i.e., a Conv1D followed by a BatchNorm, and then a MaxPool1D)


import torch
import torch.nn as nn


class CNN_Alter_Block(nn.Module):
	""" Model with only 1 Type-3 block
		Of note, num_classes can be 10 (exercise groups) or 37 (individual exercises)
	"""

	def __init__(self, num_in, num_out, kernel_size, stride, pool_size, num_classes):
		super(CNN_Alter_Block, self).__init__()
		self.conv1 		= nn.Conv1d(num_in, num_out, kernel_size, stride)
		self.relu1		= nn.ReLU()
		self.bnorm 		= nn.BatchNorm1d(num_out)
		self.pooling	= nn.MaxPool1d((pool_size))

		self.flatten	= nn.Flatten()
		self.dropout	= nn.Dropout(p = 0.5)
		self.fcl 		= nn.LazyLinear(out_features = num_classes)
		self.sfmx		= nn.Softmax(dim = 1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.bnorm(x)
		x = self.pooling(x)

		x = self.flatten(x)
		x = self.dropout(x)
		x = self.fcl(x)
		x = self.sfmx(x)

		return x


class CNN_Alter_Two_Blocks(nn.Module):
	""" Model with 2 Type-3 blocks
	"""

	def __init__(self, num_in, num_out, kernel_size, stride, pool_size, num_classes):
		super(CNN_Alter_Block, self).__init__()
		self.conv1 		= nn.Conv1d(num_in, num_out, kernel_size, stride)
		self.relu1		= nn.ReLU()
		self.bnorm1 	= nn.BatchNorm1d(num_out)
		self.pooling1	= nn.MaxPool1d((pool_size))

		self.conv2 		= nn.Conv1d(num_out, num_out, kernel_size, stride)
		self.relu2		= nn.ReLU()
		self.bnorm2 	= nn.BatchNorm1d(num_out)
		self.pooling2	= nn.MaxPool1d((pool_size))

		self.flatten	= nn.Flatten()
		self.dropout	= nn.Dropout(p = 0.5)
		self.fcl 		= nn.LazyLinear(out_features = num_classes)
		self.sfmx		= nn.Softmax(dim = 1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.bnorm1(x)
		x = self.pooling1(x)

		x = self.conv2(x)
		x = self.relu2(x)
		x = self.bnorm2(x)
		x = self.pooling2(x)

		x = self.flatten(x)
		x = self.dropout(x)
		x = self.fcl(x)
		x = self.sfmx(x)

		return x

