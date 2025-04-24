import torch
import torch.nn as nn
from layers.block import BasicConv
from utils.init import xavier_init
from layers.detection import Detect
from utils.genutils import load_pretrained_model
from torchvision.models import (efficientnet_v2_s, EfficientNet_V2_S_Weights,
                                efficientnet_v2_m, EfficientNet_V2_M_Weights,
                                efficientnet_v2_l, EfficientNet_V2_L_Weights)


class SFDetEfficientNetV2(nn.Module):

    def __init__(self,
                 mode,
                 base,
                 fusion_module,
                 pyramid_module,
                 head,
                 anchors,
                 class_count):

        super(SFDetEfficientNetV2, self).__init__()
        self.mode = mode
        self.base = base
        self.base.avgpool = None
        self.base.classifier = None

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

        x = self.base.features[0](x)
        x = self.base.features[1](x)
        x = self.base.features[2](x)
        x = self.base.features[3](x)
        sources.append(x)

        x = self.base.features[4](x)
        x = self.base.features[5](x)
        sources.append(x)

        x = self.base.features[6](x)
        x = self.base.features[7](x)
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
    's': [64, 160, 1280],
    'm': [80, 176, 512],
    'l': [96, 224, 640]
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


def build_SFDetEfficientNetV2(mode,
                              new_size,
                              efficientnet_v2_model,
                              anchors,
                              class_count,
                              model_save_path,
                              pretrained_model,
                              output_txt):

    in_channels = fusion_in_channels[efficientnet_v2_model]

    if efficientnet_v2_model == 's':
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
        base = efficientnet_v2_s(weights=weights)
    elif efficientnet_v2_model == 'm':
        weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
        base = efficientnet_v2_m(weights=weights)
    elif efficientnet_v2_model == 'l':
        weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1
        base = efficientnet_v2_l(weights=weights)

    fusion_module = get_fusion_module(config=fusion_config[str(new_size)],
                                      in_channels=in_channels)

    pyramid_module = get_pyramid_module(config=pyramid_config[str(new_size)])

    if pretrained_model is not None:
        head = multibox(config=mbox_config[str(new_size)],
                        class_count=81)
        model = SFDetEfficientNetV2(mode=mode,
                                    base=base,
                                    fusion_module=fusion_module,
                                    pyramid_module=pyramid_module,
                                    head=head,
                                    anchors=anchors,
                                    class_count=class_count)
        load_pretrained_model(model=model,
                              model_save_path=model_save_path,
                              pretrained_model=pretrained_model,
                              output_txt=output_txt)
        head = multibox(config=mbox_config[str(new_size)],
                        class_count=class_count)
        model.class_head = nn.ModuleList(modules=head[0])
        model.loc_head = nn.ModuleList(modules=head[1])

    else:
        head = multibox(config=mbox_config[str(new_size)],
                        class_count=class_count)
        model = SFDetEfficientNetV2(mode=mode,
                                    base=base,
                                    fusion_module=fusion_module,
                                    pyramid_module=pyramid_module,
                                    head=head,
                                    anchors=anchors,
                                    class_count=class_count)

    return model
