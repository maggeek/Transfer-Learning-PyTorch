import numpy as np
import matplotlib
matplotlib.use('Agg')
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from torchvision.utils import save_image
from torchvision.utils import make_grid
from config import args
import torch


# load dataset
def load_dataset()
    if args.data == 'CIFAR10':
        trainset = datasets.CIFAR10(root="cifar10/train",
                                    train=True,
                                    download=True,
                                    transform=transform)
        validset = datasets.CIFAR10(root="cifar10/test",
                                    train=False,
                                    download=True,
                                    transform=transform)
        meta = pickle.load(open('cifar10/train/cifar-10-batches-py/batches.meta', 'rb'))
    elif args.data == 'CIFAR100':
        trainset = datasets.CIFAR100(root="cifar100/train",
                                    train=True,
                                    download=True,
                                    transform=transform)
        validset = datasets.CIFAR100(root="cifar100/test",
                                    train=False,
                                    download=True,
                                    transform=transform)
        meta = pickle.load(open('cifar100/train/cifar-100-batches-py/batches.meta', 'rb'))
    elif args.data == 'MNIST':
        trainset = datasets.MNIST(root="mnist/train",
                                    train=True,
                                    download=True,
                                    transform=transform)
        validset = datasets.MNIST(root="mnist/test",
                                    train=False,
                                    download=True,
                                    transform=transform)
        meta = pickle.load(open('mnist/train/mnist-batches-py/batches.meta', 'rb'))
    elif args.data == 'MNIST':
        trainset = datasets.MNIST(root="mnist/train",
                                    train=True,
                                    download=True,
                                    transform=transform)
        validset = datasets.MNIST(root="mnist/test",
                                    train=False,
                                    download=True,
                                    transform=transform)
        meta = pickle.load(open('cifar100/train/cifar-100-batches-py/batches.meta', 'rb'))

    return trainset, validset, meta


# save losses
def save_loss(epoch, batch, how_many_batches, loss, acc, name):
    file = open('{}/loss_{}.txt'.format(args.folder_name, name), 'a')
    file.write(
        "[Epoch %d/%d] [Batch %d/%d] [Loss: %f] [Accuracy: %f]" % (epoch, args.n_epochs, batch, how_many_batches, loss, acc) + "\n")
    file.close()


# save losses plots
def save_plot(epoch, loss, name):
    x = list(range(0, epoch + 1))
    plt.plot(x, loss, '-b', label='{} loss'.format(name))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('{} loss'.format(name))
    plt.legend(loc='upper left')
    plt.savefig('{}/loss_{}.png'.format(args.folder_name, name), bbox_inches='tight')
    plt.close()


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


# save confusion matrix
def save_cm(cm, class_names, epoch):
    print(cm)
    print(cm.shape)
    df_cm = pd.DataFrame(cm, index=[i for i in class_names],
                         columns=[i for i in class_names])
    plt.figure(figsize=(25, 25))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix'+ str(epoch) +'.jpg')
    plt.close()


# save images with labels
def save_images(class_names, imgs, predicted, epoch, n_row):
    print(predicted)
    print(class_names)
    print(len(imgs))
    print(imgs[0])
    mini = torch.min(imgs)
    maxi = torch.max(imgs)
    images = (imgs - mini) / (maxi - mini)
    labels = [class_names[predicted[i]] for i in predicted]

    fig, axs = plt.subplots(n_row, n_row, figsize=(150, 150))
    axs = axs.flatten()
    for img, ax, label in zip(images, axs, labels):
        arr = img.numpy()
        arr_im = np.moveaxis(arr, 0, 2)
        ax.imshow(arr_im)
        # make bigger font here
        ax.set_title(str(label))
        # get rid of axes etc
        plt.show()
    plt.savefig('predictions' + str(epoch) + '.jpg')