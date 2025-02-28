import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from timm.models import create_model
from .backbones.model_convnext import convnext_tiny
from .backbones.resnet import Resnet
import numpy as np
from torch.nn import init
from torch.nn.parameter import Parameter

from sample4geo.Utils import init

class DEG_module(nn.Module):
    def __init__(self, channel, reduction=16,spatial_kernel_size=7,flag=False):
        super(MSE_module, self).__init__()
        self.FC11 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=1, bias=False, dilation=1)
        self.FC11.apply(weights_init_kaiming)
        self.FC12 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=2, bias=False, dilation=2)
        self.FC12.apply(weights_init_kaiming)
        self.FC13 = nn.Conv2d(channel, channel//4, kernel_size=3, stride=1, padding=3, bias=False, dilation=3)
        self.FC13.apply(weights_init_kaiming)
        self.FC1 = nn.Conv2d(channel//4, channel, kernel_size=1)
        self.FC1.apply(weights_init_kaiming)
        self.flag = flag
        self.dropout = nn.Dropout(p=0.01)
    def forward(self, x):
        x1 = (self.FC11(x) + self.FC12(x) + self.FC13(x))/3
        x1 = self.FC1(F.relu(x1))
        out1 = (x1 + x)/2
        out2 = (x1 + x)/2


        if self.flag:
            out1 = self.dropout(out1)
            out2 = self.dropout(out2)
            return out1, out2
        else:
            out1 = self.dropout(x1)
            return out1

class MultiScaleFusionDilated(nn.Module):
    def __init__(self, input_dim, num_bottleneck, groups=4):
        super(MultiScaleFusionDilated, self).__init__()
        # 使用膨胀卷积来扩展感受野
        self.dilated1 = nn.Conv1d(1, num_bottleneck // groups, kernel_size=3, dilation=1, padding=1)
        self.dilated2 = nn.Conv1d(1, num_bottleneck // groups, kernel_size=3, dilation=2, padding=2)
        self.dilated3 = nn.Conv1d(1, num_bottleneck // groups, kernel_size=3, dilation=3, padding=3)
        self.extra1 = nn.Sequential(nn.Conv1d(1,num_bottleneck // groups,kernel_size=1),
                                    nn.ReLU(inplace=True)
                                    )
    def forward(self, x):

        x = x.unsqueeze(1)  # (batch_size, 1, input_dim)

        # 使用不同膨胀率的卷积来提取不同尺度的特征
        x1 = self.dilated1(x)
        x2 = self.dilated2(x)
        x3 = self.dilated3(x)
        x4 = self.extra1(x)
        # 拼接不同尺度的特征，并通过全局池化汇聚
        out = torch.cat([x1, x2, x3,x4], dim=1)
        out = F.adaptive_avg_pool1d(out, 1).squeeze(-1)  # (batch_size, num_bottleneck)

        return out


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.5, relu=False, bnorm=True, num_bottleneck=512, linear=True,
                 return_f=False, groups=4):
        super(ClassBlock, self).__init__()
        self.return_f = return_f

        # 使用多尺度卷积模块进行特征提取
        self.multi_scale_block = MultiScaleFusionDilated(input_dim, num_bottleneck, groups=groups)

        add_block = []
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]  # Batch Normalization
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate > 0:
            add_block += [nn.Dropout(p=droprate)]  # Dropout 正则化
        self.add_block = nn.Sequential(*add_block)

        # 分类层
        self.classifier = nn.Linear(num_bottleneck, class_num)

    def forward(self, x):
        x = self.multi_scale_block(x)  # 提取多尺度特征
        x = self.add_block(x)
        if self.training:
            if self.return_f:
                f = x
                x = self.classifier(x)
                return x, f
            else:
                x = self.classifier(x)
                return x
        else:
            return x
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)


class SparseMLP1D(nn.Module):
    """
    Improved MLP with skip connections and sparse regularization (Dropout).
    Combines depth enhancement with feature sparsity control.
    """

    def __init__(self, in_channels, hid_channels, out_channels, norm_layer=None, bias=False, num_mlp=3, sparsity=0.05):
        super(SparseMLP1D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.sparsity = sparsity  # 控制稀疏度
        self.layers = nn.ModuleList()

        # 输入到隐藏层的降维
        self.layers.append(nn.Sequential(
            nn.Conv1d(in_channels, hid_channels, 1, bias=bias),
            norm_layer(hid_channels),
            nn.ReLU(inplace=True)
        ))

        # 增加深度的隐藏层
        for _ in range(num_mlp - 2):
            self.layers.append(nn.Sequential(
                nn.Conv1d(hid_channels, hid_channels, 1, bias=bias),
                norm_layer(hid_channels),
                nn.ReLU(inplace=True)
            ))

        # 隐藏层到输出层
        self.layers.append(nn.Sequential(
            nn.Conv1d(hid_channels, out_channels, 1, bias=bias)
        ))

    def init_weights(self, init_linear='kaiming'): # origin is 'normal'
        init.init_weights(self, init_linear)
    def forward(self, x):
        residual = x
        for layer in self.layers[:-1]:
            x = layer(x) # 引入跳跃连接
        x = self.layers[-1](x)
        # 稀疏正则化，通过Dropout进行稀疏性控制
        x = F.dropout(x, p=self.sparsity, training=self.training)
        return x

class MSF_module(nn.Module):
    def __init__(self, input_channels_list, output_channels):
        super(MSF_module, self).__init__()
        self.conv1 = nn.Conv1d(input_channels_list[0], output_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(input_channels_list[1], output_channels, kernel_size=1, stride=1, padding=0)
        # 融合后的特征压缩
        self.conv_fusion = nn.Conv1d(2*output_channels, output_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, W):
        # 先将通道对齐为相同的尺寸
        x = self.conv1(x)  # [16, output_channels, 144]
        W = self.conv2(W)  # [16, output_channels, 144]
        # 在通道维度上拼接特征
        fused = torch.cat([x, W], dim=1)  # [16, 2 * output_channels, 144]
        # 使用1x1卷积进行通道压缩
        fused = self.conv_fusion(fused)
        fused = self.bn(fused)
        fused = self.relu(fused)
        return fused
class FAT_module(nn.Module):
    def __init__(self, in_channels, out_channels, temperature=0.1):
        super(FAT_module, self).__init__()
        self.temperature = temperature
        self.proj = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        W = self.proj(x)
        # 使用温度缩放的Softmax
        W = F.softmax(W / self.temperature, dim=2)
        return W
class build_convnext(nn.Module):
    def __init__(self, num_classes, block=4, return_f=False, resnet=False):
        super(build_convnext, self).__init__()
        self.return_f = return_f
        if resnet:
            convnext_name = "resnet101"
            print('using model_type: {} as a backbone'.format(convnext_name))
            self.in_planes = 2048
            self.convnext = Resnet(pretrained=True)
        else:
            convnext_name = "convnext_tiny"
            print('using model_type: {} as a backbone'.format(convnext_name))
            if 'base' in convnext_name:
                self.in_planes = 1024
            elif 'large' in convnext_name:
                self.in_planes = 1536
            elif 'xlarge' in convnext_name:
                self.in_planes = 2048
            else:
                self.in_planes = 768
            self.convnext = create_model(convnext_name, pretrained=True)

        self.num_classes = num_classes
        self.classifier1 = ClassBlock(self.in_planes, num_classes, 0.5, return_f=return_f)
        self.block = block
        self.DEG = DEG_module(768,flag=True)
        self.MSF = MSF_module([768, 256], output_channels=512)
        for i in range(self.block):
            name = 'classifier_mcb' + str(i + 1)
            setattr(self, name, ClassBlock(self.in_planes, num_classes, 0.5, return_f=self.return_f))

        # define for Domain Space Alignment Module
        in_channels = 768
        hid_channels = 1536
        out_channels = 256
        norm_layer = None
        num_layers = 2
        self.proj = SparseMLP1D(in_channels, hid_channels, out_channels, norm_layer, num_mlp=num_layers)
        self.proj.init_weights()
        self.scale = 1.
        self.l2_norm = True
        self.feature_scaling = FAT_module(in_channels=256,out_channels=256,temperature=0.10)
    def forward(self, x):
        # -- backbone feature extractor
        gap_feature, part_features = self.convnext(x)
        # -- Training
        if self.training:
            # 1. Domain Space Alignment Module
            b, c, h, w = part_features.shape
            pfeat = part_features.flatten(2)  # (bs, c, h*w)
            W = self.proj(pfeat)  
            W = F.normalize(W, dim=1) if self.l2_norm else W
            W *= (1/self.scale)
            W = self.feature_scaling(W)
            W = F.softmax(W, dim=2)
            pfeat_align = self.MSF(pfeat,W)

            tri_features = self.DEG(part_features) 
            convnext_feature = self.classifier1(gap_feature) 
            tri_list = []
            for i in range(self.block):
                tri_list.append(tri_features[i].mean([-2, -1]))  
            triatten_features = torch.stack(tri_list, dim=2)  
            if self.block == 0:
                y = []
            else:
                y = self.part_classifier(self.block, triatten_features,
                                         cls_name='classifier_mcb')  
            y = y + [convnext_feature] 
            if self.return_f: 
                cls, features = [], []
                for i in y:
                    cls.append(i[0])
                    features.append(i[1])
                return pfeat_align, cls, features, gap_feature, part_features

        # -- Eval
        else:
            # ffeature = convnext_feature.view(convnext_feature.size(0), -1, 1)
            # y = torch.cat([y, ffeature], dim=2)
            pass

        return gap_feature, part_features

    def part_classifier(self, block, x, cls_name='classifier_mcb'):
        part = {}
        predict = {}
        for i in range(block):
            part[i] = x[:, :, i].view(x.size(0), -1)
            name = cls_name + str(i + 1)
            c = getattr(self, name)
            predict[i] = c(part[i])
        y = []
        for i in range(block):
            y.append(predict[i])
        if not self.training:
            return torch.stack(y, dim=2)
        return y

    def fine_grained_transform(self):

        pass


def make_convnext_model(num_class, block=4, return_f=False, resnet=False):
    print('===========building convnext===========')
    model = build_convnext(num_class, block=block, return_f=return_f, resnet=resnet)
    return model

