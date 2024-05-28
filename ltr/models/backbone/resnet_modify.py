import math
import torch.nn as nn
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls
from .base import Backbone
import torch
import torch.nn.functional as F

# class ChannelAttention(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
           
#         self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
#                                nn.ReLU(),
#                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_out = self.fc(self.avg_pool(x))
#         max_out = self.fc(self.max_pool(x))
#         out = avg_out + max_out
#         return self.sigmoid(out)

# class SKNet(nn.Module):
#     def __init__(self, in_planes, ratio=16):
#         super().__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)

#         self.fc = nn.Sequential(
#             nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
#             nn.ReLU(),
#         )
#         self.fcs = nn.ModuleList([])
#         for i in range(2):
#             self.fcs.append(
#                 nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
#             )
#         self.softmax = nn.Softmax(dim=1)


#     def forward(self, feature_map_rgb, feature_map_nir):
#         x = [feature_map_rgb, feature_map_nir]
#         x = torch.stack(x, dim=1)
#         attention = torch.sum(x, dim=1)

#         # attention = self.gap(attention)
#         # attention = self.fc(attention)
#         avg_out = self.fc(self.avg_pool(attention))
#         max_out = self.fc(self.max_pool(attention))
#         out = avg_out + max_out
        
#         attention = [fc(out) for fc in self.fcs]
#         attention = torch.stack(attention, dim=1)
#         attention = self.softmax(attention)
#         x = torch.sum(x * attention, dim=1)
#         return x

# class SpatialAttention(nn.Module):
    # def __init__(self, kernel_size=7):
    #     super(SpatialAttention, self).__init__()

    #     self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
    #     self.sigmoid = nn.Sigmoid()

    # def forward(self, x):
    #     avg_out = torch.mean(x, dim=1, keepdim=True)
    #     max_out, _ = torch.max(x, dim=1, keepdim=True)
    #     x = torch.cat([avg_out, max_out], dim=1)
    #     x = self.conv1(x)
    #     return self.sigmoid(x)

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


# class Attribute_Branch(nn.Module): # 参数太多
#     def __init__(self, input_channel):
#         super().__init__()
#         self.attribute_specific = nn.Sequential(nn.Conv2d(input_channel,input_channel,1,bias=False),nn.BatchNorm2d(input_channel),nn.ReLU(inplace=True))
#         self.branches_top = nn.Sequential(nn.Conv2d(input_channel,input_channel//2,1,bias=False),nn.BatchNorm2d(input_channel//2),nn.ReLU(inplace=True),nn.Conv2d(input_channel//2, input_channel//2, kernel_size=3, stride=1, padding=1,bias=False),nn.BatchNorm2d(input_channel//2),nn.ReLU(inplace=True))
#         self.branches_bottom = nn.Sequential(nn.Conv2d(input_channel,input_channel//2,1,bias=False),nn.BatchNorm2d(input_channel//2),nn.ReLU(inplace=True),nn.Conv2d(input_channel//2, input_channel//2, kernel_size=(1,3), stride=(1,1), padding=(0,1),bias=False),nn.BatchNorm2d(input_channel//2),nn.ReLU(inplace=True),nn.Conv2d(input_channel//2, input_channel//2, kernel_size=(3,1), stride=(1,1), padding=(1,0),bias=False),nn.BatchNorm2d(input_channel//2),nn.ReLU(inplace=True))
#     def forward(self, feature_map):
#         feature_map = self.attribute_specific(feature_map)
#         feature_map_top = self.branches_top(feature_map)
#         feature_map_bottom = self.branches_bottom(feature_map)
#         return torch.cat((feature_map_top, feature_map_bottom),dim=1)

# class Attribute_Branch(nn.Module):
#     def __init__(self, input_channel):
#         super().__init__()
#         '''
#         self.modality_specific = nn.Sequential(nn.Conv2d(input_channel,input_channel,1,bias=False),nn.BatchNorm2d(input_channel),nn.ReLU(inplace=True))
#         self.branches_top = nn.Sequential(nn.Conv2d(input_channel,input_channel//3,1,bias=False),nn.BatchNorm2d(input_channel//3),nn.ReLU(inplace=True),nn.Conv2d(input_channel//2, input_channel//2, kernel_size=3, stride=1, padding=1,bias=False),nn.BatchNorm2d(input_channel//2),nn.ReLU(inplace=True))
#         self.branches_bottom = nn.Sequential(nn.Conv2d(input_channel,input_channel//2,1,bias=False),nn.BatchNorm2d(input_channel//2),nn.ReLU(inplace=True),nn.Conv2d(input_channel//2, input_channel//2, kernel_size=(1,3), stride=(1,1), padding=(0,1),bias=False),nn.BatchNorm2d(input_channel//2),nn.ReLU(inplace=True),nn.Conv2d(input_channel//2, input_channel//2, kernel_size=(3,1), stride=(1,1), padding=(1,0),bias=False),nn.BatchNorm2d(input_channel//2),nn.ReLU(inplace=True))
#         '''
#         self.branch_pool = nn.Sequential(nn.Conv2d(input_channel,input_channel//8,1,bias=False),nn.BatchNorm2d(input_channel//8),nn.ReLU(inplace=True))

#         self.branch1 = nn.Sequential(nn.Conv2d(input_channel,3*input_channel//8,1,bias=False),nn.BatchNorm2d(3*input_channel//8),nn.ReLU(inplace=True))

#         self.branch3_pre = nn.Sequential(nn.Conv2d(input_channel,input_channel//4,1,bias=False),nn.BatchNorm2d(input_channel//4),nn.ReLU(inplace=True))

#         self.branch3_13 = nn.Sequential(nn.Conv2d(input_channel//4,input_channel//8,kernel_size=(1,3), stride=(1,1), padding=(0,1),bias=False),nn.BatchNorm2d(input_channel//8),nn.ReLU(inplace=True))
        
#         self.branch3_31 = nn.Sequential(nn.Conv2d(input_channel//4,input_channel//8,kernel_size=(3,1), stride=(1,1), padding=(1,0),bias=False),nn.BatchNorm2d(input_channel//8),nn.ReLU(inplace=True))
        
#         self.branch5_pre = nn.Sequential(nn.Conv2d(input_channel,input_channel//4,1,bias=False),nn.BatchNorm2d(input_channel//4),nn.ReLU(inplace=True),\
#         nn.Conv2d(input_channel//4,input_channel//4,kernel_size=(1,3), stride=(1,1), padding=(0,1),bias=False),nn.BatchNorm2d(input_channel//4),nn.ReLU(inplace=True),\
#         nn.Conv2d(input_channel//4,input_channel//4,kernel_size=(3,1), stride=(1,1), padding=(1,0),bias=False),nn.BatchNorm2d(input_channel//4),nn.ReLU(inplace=True))

#         self.branch5_13 = nn.Sequential(nn.Conv2d(input_channel//4,input_channel//8,kernel_size=(1,3), stride=(1,1), padding=(0,1),bias=False),nn.BatchNorm2d(input_channel//8),nn.ReLU(inplace=True))
        
#         self.branch5_31 = nn.Sequential(nn.Conv2d(input_channel//4,input_channel//8,kernel_size=(3,1), stride=(1,1), padding=(1,0),bias=False),nn.BatchNorm2d(input_channel//8),nn.ReLU(inplace=True))
        
        
#     def forward(self, feature_map):
#         # feature_map = self.modality_specific(feature_map)
#         # feature_map_top = self.branches_top(feature_map)
#         # feature_map_bottom = self.branches_bottom(feature_map)
#         # return torch.cat((feature_map_top, feature_map_bottom),dim=1)
#         branch_pool = F.avg_pool2d(feature_map, kernel_size=3,stride=1,padding=1)
#         branch_pool = self.branch_pool(branch_pool)

#         branch_1 = self.branch1(feature_map)

#         branch3_pre = self.branch3_pre(feature_map)
#         branch3_13 = self.branch3_13(branch3_pre)
#         branch3_31 = self.branch3_31(branch3_pre)

#         branch5_pre = self.branch5_pre(feature_map)
#         branch5_13 = self.branch5_13(branch5_pre)
#         branch5_31 = self.branch5_31(branch5_pre)

#         return torch.cat((branch_pool,branch_1,branch3_13,branch3_31,branch5_13,branch5_31),dim=1)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_bn=True):
        super(BasicBlock, self).__init__()
        self.use_bn = use_bn
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)

        if use_bn:
            self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)

        if self.use_bn:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if self.use_bn:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
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

# class Bottleneck_hierarchical(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
#         super(Bottleneck_hierarchical, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=dilation, bias=False, dilation=dilation)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.branch = Attribute_Branch(planes * 4)

#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         out = out + self.branch(out)

#         return out

# class Bottleneck_hierarchical_aggregation(nn.Module):
#     expansion = 4

#     def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
#         super(Bottleneck_hierarchical_aggregation, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=dilation, bias=False, dilation=dilation)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.branch_RGB = Attribute_Branch(planes * 4)
#         self.branch_NIR = Attribute_Branch(planes * 4)
#         self.downsample = downsample
#         self.stride = stride
#         for p in self.parameters(): 
#             p.requires_grad=False 
#         self.ca = SKNet(planes * 4)
#         self.sa = SpatialAttention()

        

#     def forward(self, x):
#         residual = x

#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)

#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)

#         out = self.conv3(out)
#         out = self.bn3(out)

#         if self.downsample is not None:
#             residual = self.downsample(x)

#         out += residual
#         out = self.relu(out)

#         # aggregation
#         rgb = self.branch_RGB(out)
#         nir = self.branch_NIR(out)
#         ca = self.ca(rgb, nir)
#         sa = self.sa(ca) * ca
#         out = out + sa

#         return out

class ResNet(Backbone):
    """ ResNet network module. Allows extracting specific feature blocks."""
    def __init__(self, block, layers, output_layers, num_classes=1000, inplanes=64, dilation_factor=1, frozen_layers=()):
        self.inplanes = inplanes
        super(ResNet, self).__init__(frozen_layers=frozen_layers)
        self.output_layers = output_layers
        self.conv1 = nn.Conv2d(3, inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        stride = [1 + (dilation_factor < l) for l in (8, 4, 2)]
        self.layer1 = self._make_layer(block, inplanes, layers[0], dilation=max(dilation_factor//8, 1))
        self.layer2 = self._make_layer(block, inplanes*2, layers[1], stride=stride[0], dilation=max(dilation_factor//4, 1))
        self.layer3 = self._make_layer(block, inplanes*4, layers[2], stride=stride[1], dilation=max(dilation_factor//2, 1))
        self.layer4 = self._make_layer(block, inplanes*8, layers[3], stride=stride[2], dilation=dilation_factor)

        out_feature_strides = {'conv1': 4, 'layer1': 4, 'layer2': 4*stride[0], 'layer3': 4*stride[0]*stride[1],
                               'layer4': 4*stride[0]*stride[1]*stride[2]}

        # TODO better way?
        if isinstance(self.layer1[0], BasicBlock):
            out_feature_channels = {'conv1': inplanes, 'layer1': inplanes, 'layer2': inplanes*2, 'layer3': inplanes*4,
                               'layer4': inplanes*8}
        elif isinstance(self.layer1[0], Bottleneck):
            base_num_channels = 4 * inplanes
            out_feature_channels = {'conv1': inplanes, 'layer1': base_num_channels, 'layer2': base_num_channels * 2,
                                    'layer3': base_num_channels * 4, 'layer4': base_num_channels * 8}
        # elif isinstance(self.layer1[0], Bottleneck_hierarchical):
        #     base_num_channels = 4 * inplanes
        #     out_feature_channels = {'conv1': inplanes, 'layer1': base_num_channels, 'layer2': base_num_channels * 2,
        #                             'layer3': base_num_channels * 4, 'layer4': base_num_channels * 8}
        # elif isinstance(self.layer1[0], Bottleneck_hierarchical_aggregation):
        #     base_num_channels = 4 * inplanes
        #     out_feature_channels = {'conv1': inplanes, 'layer1': base_num_channels, 'layer2': base_num_channels * 2,
        #                             'layer3': base_num_channels * 4, 'layer4': base_num_channels * 8}
        else:
            raise Exception('block not supported')

        self._out_feature_strides = out_feature_strides
        self._out_feature_channels = out_feature_channels

        # self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(inplanes*8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def out_feature_strides(self, layer=None):
        if layer is None:
            return self._out_feature_strides
        else:
            return self._out_feature_strides[layer]

    def out_feature_channels(self, layer=None):
        if layer is None:
            return self._out_feature_channels
        else:
            return self._out_feature_channels[layer]

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, output_layers=None):
        """ Forward pass with input x. The output_layers specify the feature blocks which must be returned """
        outputs = OrderedDict()

        if output_layers is None:
            output_layers = self.output_layers

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self._add_output_and_check('conv1', x, outputs, output_layers):
            return outputs

        x = self.maxpool(x)

        x = self.layer1(x)

        if self._add_output_and_check('layer1', x, outputs, output_layers):
            return outputs

        x = self.layer2(x)

        if self._add_output_and_check('layer2', x, outputs, output_layers):
            return outputs

        x = self.layer3(x)

        if self._add_output_and_check('layer3', x, outputs, output_layers):
            return outputs

        x = self.layer4(x)

        if self._add_output_and_check('layer4', x, outputs, output_layers):
            return outputs

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if self._add_output_and_check('fc', x, outputs, output_layers):
            return outputs

        if len(output_layers) == 1 and output_layers[0] == 'default':
            return x

        raise ValueError('output_layer is wrong.')


def resnet_baby(output_layers=None, pretrained=False, inplanes=16, **kwargs):
    """Constructs a ResNet-18 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(BasicBlock, [2, 2, 2, 2], output_layers, inplanes=inplanes, **kwargs)

    if pretrained:
        raise NotImplementedError
    return model


def resnet18(output_layers=None, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(BasicBlock, [2, 2, 2, 2], output_layers, **kwargs)

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet50(output_layers=None, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(Bottleneck, [3, 4, 6, 3], output_layers, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

# def resnet50_hierarchical(output_layers=None, pretrained=False, **kwargs):
#     """Constructs a ResNet-50 model.
#     """

#     if output_layers is None:
#         output_layers = ['default']
#     else:
#         for l in output_layers:
#             if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
#                 raise ValueError('Unknown layer: {}'.format(l))

#     model = ResNet(Bottleneck_hierarchical, [3, 4, 6, 3], output_layers, **kwargs)
#     if pretrained:
#         # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
#         pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
#         now_state_dict        = model.state_dict()
#         now_state_dict.update(pretrained_state_dict)
#         model.load_state_dict(now_state_dict)
#     return model

# def resnet50_hierarchical_aggregation(output_layers=None, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    """

    if output_layers is None:
        output_layers = ['default']
    else:
        for l in output_layers:
            if l not in ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer: {}'.format(l))

    model = ResNet(Bottleneck_hierarchical_aggregation, [3, 4, 6, 3], output_layers, **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        now_state_dict        = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model