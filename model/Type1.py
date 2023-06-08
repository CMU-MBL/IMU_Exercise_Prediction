# name: Type1.py
# description: models using type 1 block (i.e., a Conv1D followed by a MaxPool1D)


import torch
import torch.nn as nn


class CNN_One_Block(nn.Module):
	""" Model with only 1 Type-1 block
		Of note, num_classes can be 10 (exercise groups) or 37 (individual exercises)
	"""

	def __init__(self, num_in, num_out, kernel_size, stride, pool_size, num_classes):
		super(CNN_One_Block, self).__init__()
		self.conv1 		= nn.Conv1d(num_in, num_out, kernel_size, stride)
		self.relu1		= nn.ReLU()
		self.pooling	= nn.MaxPool1d((pool_size))

		self.flatten	= nn.Flatten()
		self.dropout	= nn.Dropout(p = 0.5)
		self.fcl 		= nn.LazyLinear(out_features = num_classes)
		self.sfmx		= nn.Softmax(dim = 1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.pooling(x)

		x = self.flatten(x)
		x = self.dropout(x)
		x = self.fcl(x)
		x = self.sfmx(x)

		return x


class CNN_Two_Blocks(nn.Module):
	""" Model with 2 Type-1 blocks
	"""

	def __init__(self, num_in, num_out, kernel_size, stride, pool_size, num_classes):
		super(CNN_Two_Blocks, self).__init__()
		self.conv1 		= nn.Conv1d(num_in, num_out, kernel_size, stride)
		self.relu1		= nn.ReLU()
		self.pooling1	= nn.MaxPool1d((pool_size))

		self.conv2 		= nn.Conv1d(num_out, num_out, kernel_size, stride)
		self.relu2		= nn.ReLU()
		self.pooling2	= nn.MaxPool1d((pool_size))

		self.flatten	= nn.Flatten()
		self.dropout	= nn.Dropout(p = 0.5)
		self.fcl 		= nn.LazyLinear(out_features = num_classes)
		self.sfmx		= nn.Softmax(dim = 1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.pooling1(x)

		x = self.conv2(x)
		x = self.relu2(x)
		x = self.pooling2(x)

		x = self.flatten(x)
		x = self.dropout(x)
		x = self.fcl(x)
		x = self.sfmx(x)

		return x


class CNN_Three_Blocks(nn.Module):
	""" Model with 3 Type-1 blocks
	"""

	def __init__(self, num_in, num_out, kernel_size, stride, pool_size, num_classes):
		super(CNN_Three_Blocks, self).__init__()
		self.conv1 		= nn.Conv1d(num_in, num_out, kernel_size, stride)
		self.relu1		= nn.ReLU()
		self.pooling1	= nn.MaxPool1d((pool_size))

		self.conv2 		= nn.Conv1d(num_out, num_out, kernel_size, stride)
		self.relu2		= nn.ReLU()
		self.pooling2	= nn.MaxPool1d((pool_size))

		self.conv3 		= nn.Conv1d(num_out, num_out, kernel_size, stride)
		self.relu3		= nn.ReLU()
		self.pooling3	= nn.MaxPool1d((pool_size))

		self.flatten	= nn.Flatten()
		self.dropout	= nn.Dropout(p = 0.5)
		self.fcl 		= nn.LazyLinear(out_features = num_classes)
		self.sfmx		= nn.Softmax(dim = 1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.pooling1(x)

		x = self.conv2(x)
		x = self.relu2(x)
		x = self.pooling2(x)

		x = self.conv3(x)
		x = self.relu3(x)
		x = self.pooling3(x)

		x = self.flatten(x)
		x = self.dropout(x)
		x = self.fcl(x)
		x = self.sfmx(x)

		return x


class CNN_Parallel_Blocks(nn.Module):
	""" Model with 2 Type-1 blocks in parallel with 1 Type-1 block
	"""

	def __init__(self, num_in, num_out, kernel_size, stride, pool_size, num_classes):
		super(CNN_Parallel_Blocks, self).__init__()
		self.s_conv		= nn.Conv1d(num_in, num_out, kernel_size, stride)
		self.s_relu		= nn.ReLU()
		self.s_pooling	= nn.MaxPool1d((pool_size))

		self.d_conv1	= nn.Conv1d(num_in, num_out, kernel_size, stride)
		self.d_relu1	= nn.ReLU()
		self.d_pooling1	= nn.MaxPool1d((pool_size))
		self.d_conv2	= nn.Conv1d(num_out, num_out, kernel_size, stride)
		self.d_relu2	= nn.ReLU()
		self.d_pooling2	= nn.MaxPool1d((pool_size))

		self.flatten	= nn.Flatten()
		self.dropout	= nn.Dropout(p = 0.5)
		self.fcl 		= nn.LazyLinear(out_features = num_classes)
		self.sfmx		= nn.Softmax(dim = 1)

	def forward(self, x):
		ss = self.s_conv(x)
		ss = self.s_relu(ss)
		ss = self.s_pooling(ss)
		ss = self.flatten(ss)

		ds = self.d_conv1(x)
		ds = self.d_relu1(ds)
		ds = self.d_pooling1(ds)
		ds = self.d_conv2(ds)
		ds = self.d_relu2(ds)
		ds = self.d_pooling2(ds)
		ds = self.flatten(ds)

		y = torch.hstack((ss, ds))
		y = self.dropout(y)
		y = self.fcl(y)
		y = self.sfmx(y)

		return y

