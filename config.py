import argparse
import torchvision.models as models
import datetime

# arguments
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
dataset_names = ["MNIST", "FashionMNIST", "LSUN", "ImageNet", "CIFAR10", "CIFAR100"]


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--n_epochs', default=25, type=int, help='number of epochs')
parser.add_argument('--n_channels', default=3, type=int, help='number of channels')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--arch', '-a', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--data', '-d', default='CIFAR10', choices=dataset_names,
                    help='dataset: ' + ' | '.join(dataset_names) + ' (default: CIFAR10)')
parser.add_argument('--pretrained', default=True, type=bool, help='use pre-trained model')
parser.add_argument('--folder_name', type=str, default='results', help='name of the folder to save the files')
args = parser.parse_args()

# open text file with append mode
#file = open('models.txt', 'a')
#file.write(str(datetime.datetime.now()) + '\n' + str(opt) + '\n')
#file.close()