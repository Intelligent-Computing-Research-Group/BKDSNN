import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import timm

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class MyConv2d():
    def __init__(self, conv):
        super(MyConv2d, self).__init__()
        self.conv = conv
    def forward(self,x):
        weight_data = self.conv.weight.data
        bias_data = self.conv.bias.data
        max_output = torch.sum(weight_data[weight_data>0])+bias_data




class VGG(nn.Module):

    def __init__(self, cfg,num_classes=1000,batch_norm=False):
        super(VGG, self).__init__()
        self.cfg = cfg

        self.features = self.make_layers(cfg,batch_norm)

        if self.cfg[-1] == 'A':
            self.classifier = nn.Sequential(
                nn.Conv2d(512, 4096, kernel_size=(7, 7), stride=(1, 1)),
                nn.ReLU(inplace=True), # unreasonable
                nn.Dropout(),
                nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=1, stride=1),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(4096, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        # self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        # print("1 x.shape",x.shape)
        if self.cfg[-1] != 'A' :
            x = x.view(x.size(0), -1)
        # print("2 x.shape",x.shape)
        x = self.classifier(x)
        # print("3 x.shape",x.shape)
        return x

    def make_layers(self,cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for i,v in enumerate(cfg):
            if v == 'M' or v == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                if i == 0:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v),nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d,nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()





cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'A']
}


def vgg11(pretrained=False, model_root=None, **kwargs):
    """VGG 11-layer model (configuration "A")"""
    model = VGG(cfg['A'], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11'], model_root))
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(cfg['A'], batch_norm=True, **kwargs)


def vgg13(pretrained=False, model_root=None, **kwargs):
    """VGG 13-layer model (configuration "B")"""
    model = VGG(cfg['B'], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13'], model_root))
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(cfg['B'], batch_norm=True, **kwargs)


def vgg16(pretrained=False, model_root=None, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    model = VGG(cfg['D'], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16'], model_root))
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(cfg['D'], batch_norm=True, **kwargs)

def vgg16_bn_imagenet(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(cfg['F'], batch_norm=True, **kwargs)


def vgg19(pretrained=False, model_root=None, **kwargs):
    """VGG 19-layer model (configuration "E")"""
    model = VGG(cfg['E'], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19'], model_root))
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(cfg['E'], batch_norm=True, **kwargs)



def test_vgg16_imagenet():
    from load_pretrained import get_layers,replace_layers

    x = torch.rand(1,3,224,224)
    pretrain_model = timm.create_model("vgg16_bn",pretrained=True)
    model = vgg16_bn_imagenet(num_classes=1000)
    layer_instances = [nn.Conv2d,nn.Linear,nn.BatchNorm2d]
    get_layers(pretrain_model,layer_instances)
    replace_layers(model,layer_instances,dataset="cifar10")

    output = model(x)
    assert output.shape == (1,1000),"output shape is not correct"

def test_vgg16_cifar():
    from load_pretrained import get_layers,replace_layers

    x = torch.rand(1,3,32,32)
    pretrain_model = timm.create_model("vgg16_bn",pretrained=True)
    model = vgg16_bn(num_classes=10)
    # print(model)
    layer_instances = [nn.Conv2d,nn.Linear,nn.BatchNorm2d]
    get_layers(pretrain_model,layer_instances)
    replace_layers(model,layer_instances,dataset="cifar10")

    output = model(x)
    assert output.shape == (1,10),"output shape is not correct"

def test_vgg16_cifar100():
    from load_pretrained import get_layers,replace_layers

    x = torch.rand(1,3,32,32)
    pretrain_model = timm.create_model("vgg16_bn",pretrained=True)
    model = vgg16_bn(num_classes=100)
    layer_instances = [nn.Conv2d,nn.Linear,nn.BatchNorm2d]
    get_layers(pretrain_model,layer_instances)
    replace_layers(model,layer_instances,dataset="cifar10")

    output = model(x)
    assert output.shape == (1,100),"output shape is not correct"


