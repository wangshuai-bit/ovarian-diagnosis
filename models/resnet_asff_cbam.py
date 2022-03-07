'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import math


model_urls = {
  'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
  'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
  'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}
'''
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = x.repeat(1,3,1,1)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out,8)

        out = out.view(out.size(0), -1)
        #print("out.size is ", out.size())

        out = self.linear(out)
        return out
'''


def conv3x3(in_planes, out_planes, stride=1):
  "3x3 convolution with padding"
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
           padding=1, bias=False)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel,channel//reduction,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction,channel,bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        b,c,_,_ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b,c,1,1)
        return x*y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.fc1(self.avg_pool(x)))
        max_out = self.fc2(self.fc1(self.max_pool(x)))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


def add_conv(in_ch, out_ch, ksize, stride, leaky=False):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage


class ASFF(nn.Module):
    def __init__(self, level, expansion, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.expansion = expansion
        self.dim = [256*expansion, 128*expansion, 64*expansion]
        self.inter_dim = self.dim[self.level]

        if level==0:
            self.stride_level_1 = add_conv(self.dim[self.level+1], self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(self.dim[self.level+2], self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 1024, 3, 1)
        elif level==1:
            self.compress_level_0 = add_conv(self.dim[self.level-1], self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(self.dim[self.level+1], self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, 512, 3, 1)
        elif level==2:
            self.compress_level_0 = add_conv(self.dim[self.level-2], self.inter_dim, 1, 1)
            self.compress_level_1 = add_conv(self.dim[self.level-1], self.inter_dim,1,1)
            self.expand = add_conv(768, 768, 3, 1)

        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.CBAM_level0_ch = ChannelAttention(self.inter_dim)
        self.CBAM_level1_ch = ChannelAttention(self.inter_dim)
        self.CBAM_level2_ch = ChannelAttention(self.inter_dim)

        self.CBAM_level0_sa = SpatialAttention()
        self.CBAM_level1_sa = SpatialAttention()
        self.CBAM_level2_sa = SpatialAttention()

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)
        self.vis= vis


    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized =x_level_1
            level_2_resized =self.stride_level_2(x_level_2)
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_resized = self.compress_level_1(x_level_1)
            level_1_resized =F.interpolate(level_1_resized, scale_factor=2, mode='nearest')
            level_2_resized =x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        level_0_cbam = self.CBAM_level0_ch(level_0_resized)*level_0_resized
        level_0_cbam = self.CBAM_level0_sa(level_0_cbam)*level_0_cbam

        level_1_cbam = self.CBAM_level1_ch(level_1_resized)*level_1_resized
        level_1_cbam = self.CBAM_level1_sa(level_1_cbam)*level_1_cbam

        level_2_cbam = self.CBAM_level2_ch(level_2_resized)*level_2_resized
        level_2_cbam = self.CBAM_level2_sa(level_2_cbam)*level_2_cbam
        #fused_out_reduced = torch.cat((level_0_se,level_1_se,level_2_se),1)
        #fused_out_reduced = level_0_se + level_1_se + level_2_se

        fused_out_reduced = torch.cat((level_0_cbam* levels_weight[:,0:1,:,:], level_1_cbam* levels_weight[:,1:2,:,:], level_2_cbam*levels_weight[:,2:,:,:]), 1)

        '''
        fused_out_reduced = level_0_se * levels_weight[:,0:1,:,:]+ \
                            level_1_se * levels_weight[:,1:2,:,:]+ \
                            level_2_se * levels_weight[:,2:,:,:]
        '''

        out = fused_out_reduced
        #out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out



class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = nn.BatchNorm2d(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = nn.BatchNorm2d(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=1, downsample=None):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
                 padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(planes * 4)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out += residual
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=2):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                 bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change

    self.layer1 = self._make_layer(block, 64, layers[0])
    self.avgpool_layer1 = nn.AvgPool2d(int(64 / block.expansion))
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.avgpool_layer2 = nn.AvgPool2d(int(32 / block.expansion))
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.avgpool_layer3 = nn.AvgPool2d(int(16 / block.expansion))


    self.downsample = nn.Sequential(
        nn.Conv2d(256 * block.expansion, 512 * block.expansion,
                  kernel_size=3, stride=2, bias=False),
        nn.BatchNorm2d(512 * block.expansion),
    )

    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    self.ca_layer3 = ChannelAttention(512 * block.expansion)
    self.sa_layer3 = SpatialAttention()

    self.fusion = ASFF(level=2, expansion= block.expansion, rfb=False, vis=False)

    self.bn2 = nn.BatchNorm2d(192 * block.expansion)
    self.fc_fused = nn.Linear(192 * block.expansion, num_classes)
    # it is slightly better whereas slower to set stride = 1
    # self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
    self.avgpool_fused = nn.AvgPool2d(16)
    self.avgpool = nn.AvgPool2d(2)
    self.fc1 = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
              kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x = x.repeat(1, 3, 1, 1)
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x = self.layer1(x)
    x_layer1 = x
    with torch.no_grad():
        level1 = self.avgpool_layer1(x)
        level1 = level1.view(level1.size(0),-1)

    x = self.layer2(x)
    x_layer2 = x
    with torch.no_grad():
        level2 = self.avgpool_layer2(x)
        level2 = level2.view(level2.size(0), -1)

    x = self.layer3(x)
    x_layer3 = x
    with torch.no_grad():
        level3 = self.avgpool_layer3(x)
        level3 = level3.view(level3.size(0), -1)

    x_res = self.downsample(x)

    x = self.layer4(x)
    x = self.ca_layer3(x)
    x = self.sa_layer3(x)
    x = x + x_res

    fused = self.fusion(x_layer3,x_layer2,x_layer1)
    #print("fused is ", fused.shape)
    #fused = self.SENet_fusion(fused)
    fused = self.bn2(fused)
    fused = self.avgpool_fused(fused)
    fused = fused.view(fused.size(0), -1)
    fused = self.fc_fused(fused)


    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc1(x)



    return x,fused,level3,level2,level1




def ResNet18_asff_cbam():
    """Constructs a ResNet-18 model.
    Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k,v in model_zoo.load_url(model_urls['resnet18']).items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model

def ResNet34_asff_cbam():
    """Constructs a ResNet-18 model.
    Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k,v in model_zoo.load_url(model_urls['resnet34']).items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model

def ResNet50_asff_cbam():
    """Constructs a ResNet-18 model.
    Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k,v in model_zoo.load_url(model_urls['resnet50']).items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def ResNet101_asff_cbam():
    """Constructs a ResNet-18 model.
    Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model =  ResNet(Bottleneck, [3, 4, 23, 3])
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k,v in model_zoo.load_url(model_urls['resnet101']).items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model

def ResNet152_asff_cbam():
    """Constructs a ResNet-18 model.
    Args:
    pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model =  ResNet(Bottleneck, [3, 8, 36, 3])
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k,v in model_zoo.load_url(model_urls['resnet152']).items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


'''

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])
'''

def test():
    net = ResNet18()
    y = net(torch.randn(1, 1, 64, 64))
    print(y.size())

# test()
