# name: network.py
# description: functions for models


import sys
import os

sys.path.append('/path/to/IMU_Exercise_Prediction')

import constants
import config


class MyDataset(Dataset):
	""" Dataset handler
	"""

	def __init__(self, list_of_samples, to_size, num_classes):
		self.to_size = to_size
		list_of_samples = [normLength(sample, constants.NORM_SAMPLE_LENGTH).T for sample in list_of_samples]

		self.X = [sample[:ID_EXERCISE_LABEL, :] for sample in list_of_samples]

		if num_classes == 10:
			self.Y = [one_hot_encoding(int(sample[constants.ID_CLUSTER_LABEL, :][0]), num_classes) for sample in list_of_samples]
		else:
			self.Y = [one_hot_encoding(int(sample[constants.ID_EXERCISE_LABEL, :][0]), num_classes) for sample in list_of_samples]

	def __len__(self):
		return len(self.Y)

	def __getitem__(self, idx):
		x = torch.from_numpy(self.X[idx]).float()
		y = self.Y[idx]

		if device == 'cuda':
			x = x.to(device)
			y = torch.from_numpy(y)
			y = y.to(device)

		return x, y


def predict(some_tensor, labs, num_classes):
	""" Evaluate prediction
	"""

	some_tensor = some_tensor.cpu().detach().numpy()
	labs        = labs.cpu().detach().numpy()

	cm 		= np.zeros([num_classes, num_classes]) # for storing confusion matrix
	y_truth = []
	y_pred 	= []

	count = 0
	for i in range(some_tensor.shape[0]):
		temp_pred = np.argmax(some_tensor[i])
		temp_truth = np.argmax(labs[i])

		cm[temp_truth, temp_pred] = cm[temp_truth, temp_pred] + 1

		y_truth.append(temp_truth)
		y_pred.append(temp_pred)

		if temp_pred == temp_truth:
			count = count + 1
		else:
			pass

	return count, cm, y_truth, y_pred


def train_loop(dataloader, model, loss_fn, optimizer, num_classes):
	""" Training phase
	"""

	global train_mode
	train_mode = True

	size        = len(dataloader.dataset)
	num_batches = len(dataloader)
	train_loss, correct, sched_factor = 0, 0, 0

	cm = np.zeros([num_classes, num_classes]) 
	y_truth = []
	y_pred  = []

	for batch, (X, y) in enumerate(dataloader):
		pred = model(X)
		y    = y.type(torch.FloatTensor)

		if device == 'cuda': y = y.cuda()

		loss = loss_fn(pred, y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if batch % 20 == 0:
			loss, current = loss.item(), batch * len(X)
			# print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

		temp_correct, temp_cm, temp_y_truth, temp_y_pred = predict(pred, y, num_classes)
		correct     += temp_correct 
		cm          += temp_cm
		y_truth     += y_truth + temp_y_truth
		y_pred      += temp_y_pred
		train_loss  += loss_fn(pred, y).item()

	train_loss /= num_batches
	train_losses.append(train_loss)
	correct /= size
	scheduler.step(train_loss)
	# print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")

	return correct, cm, y_truth, y_pred


def test_loop(dataloader, model, loss_fn, num_classes):
	""" Testing phase
	"""

	global train_mode
	train_mode = False

	size        = len(dataloader.dataset)
	num_batches = len(dataloader)
	test_loss, correct, size = 0, 0, 0

	cm = np.zeros([num_classes, num_classes]) 
	y_truth = []
	y_pred = []

	with torch.no_grad():
		for X, y in dataloader:
			pred = model(X)
			y = y.type(torch.FloatTensor)
			if device == 'cuda': y = y.cuda()
			test_loss += loss_fn(pred, y).item()

			temp_correct, temp_cm, temp_y_truth, temp_y_pred = predict(pred, y, num_classes)
			correct += temp_correct
			cm      += temp_cm
			y_truth += temp_y_truth
			y_pred  += temp_y_pred
			size    += y.shape[0]

	test_loss /= num_batches
	correct /= size

	return correct, cm, y_truth, y_pred

