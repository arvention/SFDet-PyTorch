import torch
import torch.nn as nn
from layers.block import BasicConv
from utils.init import xavier_init
from layers.detection import Detect
from torchvision.models import (resnext50_32x4d, ResNeXt50_32X4D_Weights,
                                resnext101_32x8d, ResNeXt101_32X8D_Weights)


class SFDetResNeXt(nn.Module):

    def __init__(self,
                 mode,
                 base,
                 fusion_module,
                 pyramid_module,
                 head,
                 anchors,
                 class_count):

        super(SFDetResNeXt, self).__init__()
        self.mode = mode
        self.base = base
        self.base.avgpool = None
        self.base.fc = None

        self.fusion_module = nn.ModuleList(fusion_module)
        self.batch_norm = nn.BatchNorm2d(num_features=(512 + 128 + 32))
        self.pyramid_module = nn.ModuleList(pyramid_module)

        self.class_head = nn.ModuleList(head[0])
        self.loc_head = nn.ModuleList(head[1])
        self.anchors = anchors
        self.class_count = class_count

        if mode == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect.apply

    def forward(self, x):
        sources = []
        class_preds = []
        loc_preds = []

        b, _, _, _ = x.shape

        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        sources.append(x)

        x = self.base.layer3(x)
        sources.append(x)

        x = self.base.layer4(x)
        sources.append(x)

        features = []
        for i, layer in enumerate(self.fusion_module):
            features.append(layer(sources[i]))

        features = torch.cat(features, 1)
        x = self.batch_norm(features)

        feature_pyramid = []
        for layer in self.pyramid_module:
            x = layer(x)
            feature_pyramid.append(x)

        # apply multibox head to sources
        for (x, c, l) in zip(feature_pyramid, self.class_head, self.loc_head):
            class_preds.append(c(x).permute(0, 2, 3, 1).contiguous())
            loc_preds.append(l(x).permute(0, 2, 3, 1).contiguous())

        class_preds = torch.cat([pred.view(b, -1) for pred in class_preds], 1)
        loc_preds = torch.cat([pred.view(b, -1) for pred in loc_preds], 1)

        class_preds = class_preds.view(b, -1, self.class_count)
        loc_preds = loc_preds.view(b, -1, 4)

        if self.mode == 'test':
            output = self.detect(self.class_count,
                                 self.softmax(class_preds),
                                 loc_preds,
                                 self.anchors)

        else:
            output = (class_preds,
                      loc_preds)

        return output

    def init_weights(self,
                     model_save_path,
                     base_network):

        self.fusion_module.apply(fn=xavier_init)
        self.pyramid_module.apply(fn=xavier_init)
        self.class_head.apply(fn=xavier_init)
        self.loc_head.apply(fn=xavier_init)


def get_fusion_module(config,
                      in_channels):

    layers = []

    layers += [BasicConv(in_channels=in_channels[0],
                         out_channels=config[0][0],
                         kernel_size=config[0][1],
                         stride=config[0][2],
                         padding=config[0][3])]

    layers += [nn.Sequential(BasicConv(in_channels=in_channels[1],
                                       out_channels=config[1][0],
                                       kernel_size=config[1][1],
                                       stride=config[1][2],
                                       padding=config[1][3]),
                             nn.PixelShuffle(upscale_factor=2))]

    layers += [nn.Sequential(BasicConv(in_channels=in_channels[2],
                                       out_channels=config[2][0],
                                       kernel_size=config[2][1],
                                       stride=config[2][2],
                                       padding=config[2][3]),
                             nn.PixelShuffle(upscale_factor=4))]

    return layers


def get_pyramid_module(config):

    layers = []

    for layer in config:
        layers += [BasicConv(in_channels=layer[0],
                             out_channels=layer[1],
                             kernel_size=layer[2],
                             stride=layer[3],
                             padding=layer[4])]

    return layers


def multibox(config,
             class_count):

    class_layers = []
    loc_layers = []

    for in_channels, num_anchors in config:
        class_layers += [nn.Conv2d(in_channels=in_channels,
                                   out_channels=num_anchors * class_count,
                                   kernel_size=3,
                                   padding=1)]
        loc_layers += [nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_anchors * 4,
                                 kernel_size=3,
                                 padding=1)]

    return class_layers, loc_layers


fusion_in_channels = {
    'default': [512, 1024, 2048]
}


fusion_config = {
    '300': [[512, 3, 1, 0],
            [512, 2, 1, 0],
            [512, 2, 1, 0]],

    '512': [[512, 3, 1, 1],
            [512, 3, 1, 1],
            [512, 3, 1, 1]]
}

pyramid_config = {
    '300': [[(512 + 128 + 32), 512, 3, 1, 1],
            [512, 512, 3, 2, 1],
            [512, 256, 3, 2, 1],
            [256, 256, 3, 2, 1],
            [256, 256, 3, 1, 0],
            [256, 256, 3, 1, 0]],

    '512': [[(512 + 128 + 32), 512, 3, 1, 1],
            [512, 512, 3, 2, 1],
            [512, 256, 3, 2, 1],
            [256, 256, 3, 2, 1],
            [256, 256, 3, 2, 1],
            [256, 256, 3, 2, 1],
            [256, 256, 2, 1, 0]]
}

mbox_config = {
    '300': [(512, 6),
            (512, 6),
            (256, 6),
            (256, 6),
            (256, 4),
            (256, 4)],

    '512': [(512, 6),
            (512, 6),
            (256, 6),
            (256, 6),
            (256, 6),
            (256, 4),
            (256, 4)]
}


def build_SFDetResNeXt(mode,
                       new_size,
                       resnext_model,
                       anchors,
                       class_count):

    in_channels = fusion_in_channels['default']

    if resnext_model == '50_32x4d':
        base = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    elif resnext_model == '101_32x8d':
        base = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V1)

    fusion_module = get_fusion_module(config=fusion_config[str(new_size)],
                                      in_channels=in_channels)

    pyramid_module = get_pyramid_module(config=pyramid_config[str(new_size)])

    head = multibox(config=mbox_config[str(new_size)],
                    class_count=class_count)

    return SFDetResNeXt(mode=mode,
                        base=base,
                        fusion_module=fusion_module,
                        pyramid_module=pyramid_module,
                        head=head,
                        anchors=anchors,
                        class_count=class_count)
