import torch.nn as nn
import torch.nn.functional as F


class ResNet20(nn.Module):
    """
    ResNet20 implementation based on https://github.com/akamaster/pytorch_resnet_cifar10
    """

    def __init__(self, writer, input_grad, in_shape, out_dim, batch_norm):
        super(ResNet20, self).__init__()
        assert tuple(in_shape) == (3, 32, 32)
        self.writer = writer
        self.input_grad = input_grad
        self.out_dim = out_dim
        self.resnet = ResNet(BasicBlock, [3, 3, 3], num_classes=out_dim, batch_norm=batch_norm)
        self.batch_norm = batch_norm

    def forward(self, x):
        if self.input_grad:
            x.requires_grad = True
        x = self.resnet(x)
        return x


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, batch_norm=True):
        super(ResNet, self).__init__()
        self.batch_norm = batch_norm
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, batch_norm=batch_norm)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, batch_norm=batch_norm)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, batch_norm=batch_norm)
        self.fcout = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride, batch_norm):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride, option='A', batch_norm=batch_norm))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.fcout(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A', batch_norm=True):
        super(BasicBlock, self).__init__()
        self.batch_norm = batch_norm

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
