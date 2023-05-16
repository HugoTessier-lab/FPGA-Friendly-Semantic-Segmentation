import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1):
        super(BasicBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 entry_downsampling_rate=1):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group

        if entry_downsampling_rate == 4:
            self.head = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                                bias=False),
                                      nn.BatchNorm2d(self.inplanes),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        elif entry_downsampling_rate == 2:
            self.head = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                                bias=False),
                                      nn.BatchNorm2d(self.inplanes),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        elif entry_downsampling_rate == 1:
            self.head = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                                bias=False),
                                      nn.BatchNorm2d(self.inplanes),
                                      nn.ReLU(inplace=True))
        else:
            print('ResNet-18: initial downsampling stage can only be 1, 2 or 4')
            raise ValueError

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1):
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.head(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x0, x2, x3, x4]


def _resnet(arch,
            block,
            layers,
            pretrained,
            progress,
            entry_downsampling_rate):
    model = ResNet(block, layers, entry_downsampling_rate=entry_downsampling_rate)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        del state_dict['fc.weight']
        del state_dict['fc.bias']

        if state_dict['conv1.weight'].shape != model.head[0].weight.shape:
            state_dict['conv1.weight'] = state_dict['conv1.weight'][:,:,2:-2,2:-2]

        state_dict['head.0.weight'] = state_dict['conv1.weight']
        del state_dict['conv1.weight']

        state_dict['head.1.weight'] = state_dict['bn1.weight']
        del state_dict['bn1.weight']

        state_dict['head.1.bias'] = state_dict['bn1.bias']
        del state_dict['bn1.bias']

        state_dict['head.1.running_mean'] = state_dict['bn1.running_mean']
        del state_dict['bn1.running_mean']

        state_dict['head.1.running_var'] = state_dict['bn1.running_var']
        del state_dict['bn1.running_var']

        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, entry_downsampling_rate=1):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, entry_downsampling_rate)


class UnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UnetBlock, self).__init__()

        self.transpose = nn.Conv2d(in_channels, out_channels, 3, bias=False, padding=1)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, bias=False, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, bias=False, padding=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, side_input):
        x = F.interpolate(x, scale_factor=2)
        x = self.transpose(x)
        x = self.bn1(x)
        x = x + side_input
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x


class Unet(nn.Module):
    def __init__(self, encoder, num_classes=19, exit_upsampling_rate=1.):
        super(Unet, self).__init__()

        self.encoder = encoder
        self.block1 = UnetBlock(512, 256)
        self.block2 = UnetBlock(256, 128)
        self.block3 = UnetBlock(128, 64)

        self.exit_upsampling_rate = exit_upsampling_rate
        if self.exit_upsampling_rate != 1:
            self.transpose = nn.Conv2d(64, 64, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(64, num_classes, 1, bias=False)

    def forward(self, x):
        outputs = self.encoder(x)
        x = outputs[3]
        x = self.block1(x, outputs[2])
        x = self.block2(x, outputs[1])
        x = self.block3(x, outputs[0])

        if self.exit_upsampling_rate != 1:
            x = F.interpolate(x, scale_factor=self.exit_upsampling_rate)
            x = self.transpose(x)
            x = self.bn1(x)
            x = self.relu(x)

        x = self.conv(x)
        return x


def resnet18_unet(num_classes, pretrained_encoder, output_downsampling_rate=4, entry_downsampling_rate=1):
    encoder = resnet18(
        pretrained=pretrained_encoder,
        entry_downsampling_rate=entry_downsampling_rate)
    unet = Unet(
        encoder=encoder,
        num_classes=num_classes,
        exit_upsampling_rate=entry_downsampling_rate / output_downsampling_rate)
    return unet
