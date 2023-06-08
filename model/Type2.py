# name: Type2.py
# description: models using type 2 block (i.e., two Conv1D followed by a MaxPool1D)


import torch
import torch.nn as nn


class CNN_One_Deep_Block(nn.Module):
	""" Model with 1 Type-2 block
		Of note, num_classes can be 10 (exercise groups) or 37 (individual exercises)
	"""

	def __init__(self, num_in, num_out, kernel_size, stride, pool_size, num_classes):
		super(CNN_One_Deep_Block, self).__init__()
		self.conv1 		= nn.Conv1d(num_in, num_out, kernel_size, stride)
		self.relu1		= nn.ReLU()
		self.conv2 		= nn.Conv1d(num_out, num_out, kernel_size, stride)
		self.relu2		= nn.ReLU()
		self.pooling	= nn.MaxPool1d((pool_size))

		self.flatten	= nn.Flatten()
		self.dropout	= nn.Dropout(p = 0.5)
		self.fcl 		= nn.LazyLinear(out_features = num_classes)
		self.sfmx		= nn.Softmax(dim = 1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.pooling(x)

		x = self.flatten(x)
		x = self.dropout(x)
		x = self.fcl(x)
		x = self.sfmx(x)

		return x


class CNN_Two_Deep_Blocks(nn.Module):
	""" Model with 2 Type-2 blocks
	"""

	def __init__(self, num_in, num_out, kernel_size, stride, pool_size, num_classes):
		super(CNN_Two_Deep_Blocks, self).__init__()
		self.conv1 		= nn.Conv1d(num_in, num_out, kernel_size, stride)
		self.relu1		= nn.ReLU()
		self.conv2 		= nn.Conv1d(num_out, num_out, kernel_size, stride)
		self.relu2		= nn.ReLU()
		self.pooling1	= nn.MaxPool1d((pool_size))

		self.conv3 		= nn.Conv1d(num_out, num_out, kernel_size, stride)
		self.relu3		= nn.ReLU()
		self.conv4 		= nn.Conv1d(num_out, num_out, kernel_size, stride)
		self.relu4		= nn.ReLU()
		self.pooling2	= nn.MaxPool1d((pool_size))

		self.flatten	= nn.Flatten()
		self.dropout	= nn.Dropout(p = 0.5)
		self.fcl 		= nn.LazyLinear(out_features = num_classes)
		self.sfmx		= nn.Softmax(dim = 1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.pooling1(x)

		x = self.conv3(x)
		x = self.relu3(x)
		x = self.conv4(x)
		x = self.relu4(x)
		x = self.pooling2(x)

		x = self.flatten(x)
		x = self.dropout(x)
		x = self.fcl(x)
		x = self.sfmx(x)

		return x


