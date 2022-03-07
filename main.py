'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data as data

import torchvision
import torchvision.transforms as transforms

import numpy as np
from PIL import Image

import os
import argparse
import pickle
import random
from models import *
from models.resnet_asff_eca import *
from models.resnet_asff_a2net import *
from models.resnet_asff_sknet import *
from models.resnet_asff_cbam import *
from models.resnet_asff_res import *
from utils import progress_bar

from sklearn.metrics import *
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.manifold import TSNE

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def setup_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False

setup_seed()
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',default=False,
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("the device is ", device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
print("the learning rate is ", args.lr)
# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    #transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
'''
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
print("train dataset is ", trainset)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
print("test dataset is ", testset)
'''
class MRI_img(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None,train=True):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.data = []
        self.targets = []

        if (self.train):
            choose_list = self.train_list
        else:
            choose_list = self.test_list

        for file_name in choose_list:
            file_path = os.path.join(self.root,file_name)
            with open(file_path,'rb') as f:
                entry = pickle.load(f, encoding='bytes')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels']-1)
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data).reshape(-1,1,64,64)
        #self.data = self.data.transpose(0, 2, 3, 1)

    def __getitem__(self, index):
        img,target = self.data[index], self.targets[index]


        if self.transform is not None:
            img = Image.fromarray(np.uint8(img))
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        img = np.asarray(img)

        return img,target

    def __len__(self):
        return len(self.data)

    train_list = ['train_64_5_to_1']
    test_list = ['test_64_5_to_1']

root = '/data/wangshuai/shuju_luanchao/ce2/dataset'
trainset = MRI_img(root= root, transform=None, target_transform=None,train=True)
testset = MRI_img(root=root, transform=None,target_transform=None,train=False)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=65, shuffle=False, num_workers=2)


# Model
print('==> Building model..')
#net = VGG('VGG19')
net = ResNet18_asff_res()
#net = FPN50()
#net = PreActResNet18()
#net = GoogLeNet()
#net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
#net = ShuffleNetG3_atten()
#net = SENet18_res()
#net = ShuffleNetV2(1.5)
#net = EfficientNetB0()
# net = RegNetX_200MF()
#net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print("load the model")

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                      momentum=0.9, weight_decay=5e-4)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
optimizer = torch.optim.Adam(net.parameters(),lr=args.lr,betas=(0.9,0.99),weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,step_size=20,gamma=0.1,last_epoch=-1)

from matplotlib import cm

def plot_with_labels(lowDWeights,labels,name):
    #plt.cla()
    X,Y = lowDWeights[:,0],lowDWeights[:,1]
    font = np.floor((X.max()-X.min())/3)
    for x,y,s in zip(X,Y,labels):
        if s == 0:
            c = cm.rainbow(int(255*s/3))
            type_0 = plt.scatter(x,y,c=c)
        if s == 1:
            c = cm.rainbow(int(255 * s / 3))
            type_1 = plt.scatter(x, y, c=c)
        #plt.text(x,y,s, backgroundcolor=c,fontsize=font)
    plt.xlim(X.min(),X.max())
    plt.ylim(Y.min(),Y.max())
    plt.legend((type_0,type_1),('benign','malignant'),loc=(0.005,0.90),prop = {'size':8})
    plt.xlabel('tSNE_1')
    plt.ylabel('tSNE_2')
    plt.title(name)
    plt.show()
    plt.pause(0.01)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    score_list = []
    label_list = []
    predicted_list = []

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        '''
        outputs, layer3, layer2, layer1 = net(inputs)
        loss4 = criterion(outputs, targets)
        loss3 = criterion(layer3, targets)
        loss2 = criterion(layer2, targets)
        loss1 = criterion(layer1, targets)
        loss = loss4
        '''
        outputs, fused,level3,level2,level1 = net(inputs)
        loss_fused = criterion(fused, targets)
        loss = criterion(outputs, targets) + loss_fused

        #outputs = net(inputs)
        #loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    acc_list = np.zeros(shape=[60])
    with torch.no_grad():
        score_list = []
        label_list = []
        predicted_list = []

        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            '''
            outputs, layer3, layer2, layer1 = net(inputs)
            loss4 = criterion(outputs, targets)
            loss3 = criterion(layer3, targets)
            loss2 = criterion(layer2, targets)
            loss1 = criterion(layer1, targets)
            loss = loss4 + loss3 + loss2 + loss1
            '''
            outputs, fused,level3,level2,level1 = net(inputs)
            loss_fused = criterion(fused, targets)
            loss = criterion(outputs, targets) + loss_fused

            #outputs = net(inputs)
            #loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            if epoch == 59:
                softmax_fn = nn.Softmax(dim=1)
                score_tmp = softmax_fn(outputs)
                score_list.extend(score_tmp.detach().cpu().numpy())
                label_list.extend(targets.cpu().numpy())
                predicted_list.extend(predicted.cpu().numpy())

    # Save checkpoint.
    acc = 100. * correct / total
    acc_list[epoch] = acc
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        #torch.save(state, './checkpoint/ckpt_Res18_ours_ce2_4th.pth')
        best_acc = acc
    elif epoch == 59:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, './checkpoint/ckpt_Res18_ours_ce2_final_seed_6.pth')
        best_acc = acc

    if epoch == 59:
        score_array = np.array(score_list)
        label_tensor = torch.tensor(label_list)
        label_tensor = label_tensor.reshape((label_tensor.shape[0], 1))
        label_onehot = torch.zeros(label_tensor.shape[0], 2)  # 2 is the number of class
        label_onehot.scatter_(dim=1, index=label_tensor, value=1)
        label_onehot = np.array(label_onehot)
        print('the shape of score_array is', score_array.shape)  # (batchsize, classnum)
        print('the shape of label_onthot is', label_onehot.shape)  # torch.size([batchsize, classnum])

        right_0 = 0
        wrong_0_to_1 = 0
        wrong_1_to_0 = 0
        right_1 = 0
        for i in range(len(predicted_list)):
            if label_list[i] == 0:
                if predicted_list[i] == 0:
                    right_0 += 1
                elif predicted_list[i] == 1:
                    wrong_0_to_1 += 1
            elif label_list[i] == 1:
                if predicted_list[i] == 0:
                    wrong_1_to_0 += 1
                elif predicted_list[i] == 1:
                    right_1 += 1
        print("the predict of 0 is ", right_0, wrong_0_to_1, right_0 + wrong_0_to_1,right_0/(right_0 + wrong_0_to_1))
        print("the predict of 1 is ", wrong_1_to_0, right_1, wrong_1_to_0 + right_1,right_1/(wrong_1_to_0 + right_1))
        print("the total is        ", right_0 + wrong_1_to_0, wrong_0_to_1 + right_1,
              right_0 + wrong_0_to_1 + wrong_1_to_0 + right_1)
        print("the acc is          ", right_0 / (right_0 + wrong_1_to_0), right_1 / (wrong_0_to_1 + right_1), )


        fpr_dict = dict()
        tpr_dict = dict()
        roc_auc_dict = dict()
        for i in range(2):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(label_onehot[:, i], score_array[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        # micro
        fpr_dict['micro'], tpr_dict['micro'], _ = roc_curve(label_onehot.ravel(), score_array.ravel())
        roc_auc_dict['micro'] = auc(fpr_dict['micro'], tpr_dict['micro'])

        # macro
        # first aggregrate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(2)]))
        # then interplolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(2):
            mean_tpr += interp(all_fpr, fpr_dict[i], tpr_dict[i])
        # finally average it and compute AUC
        mean_tpr /= 2
        fpr_dict['macro'] = all_fpr
        tpr_dict['macro'] = mean_tpr
        roc_auc_dict['macro'] = auc(fpr_dict['macro'], tpr_dict['macro'])

        # plt roc curves of classes
        plt.figure()
        lw = 2
        '''
        plt.plot(fpr_dict['micro'], tpr_dict['micro'],
                 label='micro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc_dict['micro']),
                 color='deeppink', linestyle=':', linewidth=4
                 )
        '''
        plt.plot(fpr_dict['macro'], tpr_dict['macro'],
                 label='average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc_dict['macro']),
                 color='navy', linestyle=':', linewidth=4
                 )
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(2), colors):
            plt.plot(fpr_dict[i], tpr_dict[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc_dict[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.5])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receive operating characteristic to our method')
        plt.legend(loc='lower right')
        #plt.show()
        print("the macro auc is (softmax) ", roc_auc_dict['macro'])
        print("the micro auc is (softmax) ", roc_auc_dict['micro'])
        print('plot showed, end')

        #print('the acc list is', acc_list)

for epoch in range(start_epoch, start_epoch + 60):
    train(epoch)
    test(epoch)
    scheduler.step()
