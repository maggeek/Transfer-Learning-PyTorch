import math
import numpy as np
import random
import pickle
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.models as models
from torch.autograd import Variable
from utils import *
from config import args
from sklearn.metrics import confusion_matrix


if __name__ == '__main__':

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cuda = True if torch.cuda.is_available() else False
	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

	# correct for all classifiers

	#The images have to be loaded in to a range of [0, 1]
	transform = transforms.Compose([
	 	transforms.Resize(256),
	 	transforms.CenterCrop(224),
	 	transforms.ToTensor(),
	 	transforms.Normalize(
		mean=[0.485, 0.456, 0.406],
		std=[0.229, 0.224, 0.225]
	 )])
	

	# load dataset
	trainset, validset, meta = load_dataset()

	trainloader = torch.utils.data.DataLoader(trainset,
											  batch_size=args.batch_size,
											  shuffle=True,
											  num_workers=args.workers)
	n_batches = math.ceil(len(trainloader)/args.batch_size)
	validloader = torch.utils.data.DataLoader(validset,
											  batch_size=math.ceil(len(validset)/n_batches),
											  shuffle=False,
											  num_workers=args.workers)

	# Get class names and number of classes
	# works to get labels for CIFAR10 * CIFAR100
	class_names = meta['label_names']
	num_classes = len(class_names)
	
	# load or create model
	if args.pretrained:
		# use pretrained model
		print("=> using pre-trained model '{}'".format(args.arch))
			# Freeze model weights
		model = models.__dict__[args.arch]()
		for param in model.parameters():
			param.requires_grad = False
		# overwrite the last layer with correct number of classes
		num_ftrs = model.fc.in_features
		print(num_ftrs)
		#model.classifier[6] = nn.Linear(num_ftrs, num_classes)
		model.fc = nn.Linear(num_ftrs, num_classes)
		# Add on classifier
		#model.fc = nn.Sequential(
		#                      nn.Linear(num_ftrs, 256),
		#                      nn.ReLU(),
		#                      nn.Dropout(0.4),
		#                      nn.Linear(256, num_classes))
		# Only training classifier[6]
		print(model.fc)
	else:
		# train model from scratch
		print("=> creating model '{}'".format(args.arch))
		model = models.__dict__[args.arch](pretrained=args.pretrained)
	print(model)
	
	print(class_names)
	print(num_classes)

	#valid = [validset[i] for i in range(3)][0]
	#valid = np.asarray(valid).type(Tensor)

	# Find total parameters and trainable parameters
	total_params = sum(p.numel() for p in model.parameters())
	print(f'{total_params:,} total parameters.')
	total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f'{total_trainable_params:,} training parameters.')

	# which one to use???
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01)
	#optimizer = torch.optim.SGD(model.parameters(),
	#							lr=args.lr,
	#                            momentum=args.momentum,
	#                            weight_decay=args.weight_decay)
	if cuda:
		criterion.cuda()
		model.cuda()
	
	total_train_loss, total_train_acc, total_valid_loss, total_valid_acc = [], [], [], []
	for epoch in range(args.n_epochs):

		n_train_correct, n_train_total, n_valid_correct, n_valid_total = 0, 0, 0, 0
		model.train()
		for batch, (images, gtruth) in enumerate(trainloader):

			input_var = Variable(images.type(Tensor))
			gtruth_var = Variable(gtruth.type(Tensor))

			# compute output
			output = model(input_var)
			gt = gtruth_var.long().cuda()
			loss = criterion(output, gt)

			# compute gradient and do SGD step
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# calculate accuracy of predictions in the current batch
			n_train_correct += (torch.max(output, 1)[1].view(gt.size()) == gt).sum().item()
			n_train_total += args.batch_size
			train_acc = 100. * n_train_correct / n_train_total

			# save loss and accuracy
			save_loss(epoch, batch, n_batches, loss.item(), train_acc, 'training')

		total_train_loss.append(loss.item())
		total_train_acc.append(train_acc)

		predicted = []
		actual = []
		valid = torch.zeros([100, 3, 224, 224])
		model.eval()
		for batch, (images, gtruth) in enumerate(validloader):

			input_var = Variable(images.type(Tensor))
			gtruth_var = Variable(gtruth.type(Tensor))

			if batch < 10:
				valid[batch*10:batch*10+10, :, :, :] = images.type(Tensor)
			
			# compute output
			output = model(input_var)
			gt = gtruth_var.long().cuda()
			loss = criterion(output, gt)
			actual.extend(gtruth.numpy())
			predicted.extend(torch.argmax(output, 1).cpu().numpy())

			# calculate accuracy of predictions in the current batch
			n_valid_correct += (torch.max(output, 1)[1].view(gt.size()) == gt).sum().item()
			n_valid_total += int(len(validset)/n_batches)
			valid_acc = 100. * n_valid_correct / n_valid_total

			# save loss and accuracy
			save_loss(epoch, batch, n_batches, loss.item(), valid_acc, 'validation')

		# save confusion matrix
		cm = confusion_matrix(actual, predicted)
		save_cm(cm, class_names, epoch)

		total_valid_loss.append(loss.item())
		total_valid_acc.append(valid_acc)

		#if epoch % 5 == 0:
		save_plot(epoch, total_train_loss, 'Training loss')
		save_plot(epoch, total_train_acc, 'Training accuracy')
		save_plot(epoch, total_valid_loss, 'Validation loss')
		save_plot(epoch, total_valid_acc, 'Validation accuracy')
		#torch.save(model.state_dict(), '{}/encoder.pth'.format(opt.folder_name))
		print('Plots saved')
		n_images = 100
		save_images(class_names, valid, predicted[0:n_images], epoch, int(math.sqrt(n_images)))