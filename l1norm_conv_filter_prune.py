import sys
import time
import os
import operator
import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable

torch.manual_seed(43) # This gives us stable randomness

def _make_pair(x):
    if hasattr(x, '__len__'):
        return x
    else:
        return (x, x)

class SparseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=1):
        super(SparseConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _make_pair(stride)
        self.padding = _make_pair(padding)

        # initialize weights of this layer 
        self._weight = nn.Parameter(torch.randn([self.out_channels, self.in_channels,
                                                        self.kernel_size, self.kernel_size]))
        stdv = 1. / math.sqrt(in_channels)
        self._weight.data.uniform_(-stdv, stdv)
        # initialize mask
        # Since we are going to zero out the whole filter, the number of 
        # elements in the mask is equal to the number of filters.
        self.register_buffer('_mask', torch.ones(out_channels))


    def forward(self, x):
        return F.conv2d(x, self.weight, stride=self.stride,
                        padding=self.padding)
                    
    @property
    def weight(self):
        # check out https://pytorch.org/docs/stable/notes/broadcasting.html 
        # to better understand the following line 
        return self._mask[:,None,None,None] * self._weight


def sparse_conv_block(in_channels, out_channels, kernel_size=3, stride=1,
                      padding=1):
    return nn.Sequential(
        SparseConv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

class SparseConvNet(nn.Module):
    def __init__(self):
        super(SparseConvNet, self).__init__()
        # PART 4.1: Implement!
        self.model = nn.Sequential(
            sparse_conv_block(3, 32),
            sparse_conv_block(32, 32),
            sparse_conv_block(32, 64, stride=2),
            sparse_conv_block(64, 64),
            sparse_conv_block(64, 64),
            sparse_conv_block(64, 128, stride=2),
            sparse_conv_block(128, 128),
            sparse_conv_block(128, 256),
            sparse_conv_block(256, 256),
            nn.AdaptiveAvgPool2d(1)
            )

        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        h = self.model(x)
        B, C, _, _ = h.shape
        h = h.view(B, C)
        return self.classifier(h)


def get_sparse_conv2d_layers(net):
    '''
    Helper function that returns all SparseConv2d layers in the neural network.
    '''
    sparse_conv_layers = []
    for layer in net.children():
        if isinstance(layer, SparseConv2d):
            sparse_conv_layers.append(layer)
        else:
            child_layers = get_sparse_conv2d_layers(layer)
            sparse_conv_layers.extend(child_layers)
    
    return sparse_conv_layers
    
def filter_l1_pruning(net, prune_percent):
    for i, layer in enumerate(get_sparse_conv2d_layers(net)):
        num_nonzero = layer._mask.sum().item()
        num_total = len(layer._mask)
        num_prune = round(num_total * prune_percent)
        sparsity = 100.0 * (1 - (num_nonzero / num_total))
        print(num_prune, num_total, prune_percent, sparsity)
        
        l1_norm_filters = []
        for i in range(num_total):
          l1_norm_filters.append((torch.sum(layer.weight[i,:,:,:]), 1))

        # Sort based on absolute weight sum of each filter while keeping track of which filter is which
        l1_norm_filters.sort(key = operator.itemgetter(0))
        for j in range(num_prune):
          layer._mask.data[l1_norm_filters[j][1]] = 0


###########################################################################
# Train our convolutional neural network on the CIFAR-10 dataset while 
# implementing a l1-norm filter pruning schedule of 10% every 10 epochs up 
# to the 50th epoch for every layer. Model will have 50% sparsity by the 
# end of pruning
###########################################################################

# Load training data
transform_train = transforms.Compose([                                   
    transforms.RandomCrop(32, padding=4),                                       
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                        download=True,
                                        transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=2)

# Load testing data
transform_test = transforms.Compose([                                           
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True,
                                       transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False,
                                         num_workers=2)
print('Finished loading datasets!')


device = 'cuda'
net = SparseConvNet()
net = net.to(device)

lr = 0.1
milestones = [25, 50, 75, 100]
prune_percentage = 0.1
prune_epoch = 10

epochs = 100

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                            weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=milestones,
                                                 gamma=0.1)

train_loss_tracker, train_acc_tracker = [], []
test_loss_tracker, test_acc_tracker = [], []

print('Training for {} epochs, with learning rate {} and milestones {}'.format(
      epochs, lr, milestones))

start_time = time.time()
for epoch in range(0, epochs):
    train(net, epoch, train_loss_tracker, train_acc_tracker)

    if (epoch + 1) % prune_epoch == 0 and epoch < 50:
        print('Pruning at epoch {}'.format(epoch))
        filter_l1_pruning(net, prune_percentage)
        
    test(net, epoch, test_loss_tracker, test_acc_tracker)
    scheduler.step()


total_time = time.time() - start_time
print('Total training time: {} seconds'.format(total_time))


# Plot training loss and test accuracy for SparseConvNet
plt.figure(figsize=(8, 6))

ma = moving_average(train_loss_tracker, n=100)

plt.plot([x for x in range(0, 0+len(ma))], ma, 'r-') # plot training loss (with  moving average)
plt.title(f'Training loss over iterations for SparseConvNet, scheduler {scheduler_name}, lr={lr}')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.savefig(f'12-tloss_{scheduler_name}.jpg')
plt.show()

plt.figure(figsize=(8, 6))

plt.plot([x for x in range(0, 0+len(test_acc_tracker))], test_acc_tracker, 'b-') #plot test accuracy
plt.title(f'Test accuracy over epochs for SparseConvNet, scheduler {scheduler_name}, lr={lr}')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.xticks(range(0, int(0+len(test_acc_tracker)), 10))

plt.savefig(f'12-tacc_{scheduler_name}.jpg')
plt.show()