import torch
import torchvision

class ResNet(torchvision.models.ResNet):
    def __init__(self, block, layers, num_classes=10, width=64):
        """To make it possible to vary the width, we need to override the constructor of the torchvision resnet."""

        torch.nn.Module.__init__(self)  # Skip the parent constructor. This replaces it.
        self._norm_layer = torch.nn.BatchNorm2d
        self.inplanes = width
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        # The initial convolutional layer.
        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        # self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # The subsequent blocks.
        self.layer1 = self._make_layer(block, width, layers[0], stride=1)
        self.layer2 = self._make_layer(block, width*2, layers[1], stride=2, dilate=False)
        self.layer3 = self._make_layer(block, width*4, layers[2], stride=2, dilate=False)
        self.layer4 = self._make_layer(block, width*8, layers[3], stride=2, dilate=False)

        # The last layers.
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(width*8*block.expansion, num_classes)

        # Default init.
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
    
    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
