import torch
import torch.nn as nn
from math import ceil
from layers.block import BasicConv
from utils.init import xavier_init
from layers.detection import Detect
from utils.genutils import load_pretrained_model
from torchvision.models import (resnet18, ResNet18_Weights,
                                resnet34, ResNet34_Weights,
                                resnet50, ResNet50_Weights,
                                resnet101, ResNet101_Weights,
                                resnet152, ResNet152_Weights)


class SFDetV2ResNet(nn.Module):

    def __init__(self,
                 mode,
                 base,
                 fusion_module,
                 pyramid_module,
                 head,
                 anchors,
                 class_count):

        super(SFDetV2ResNet, self).__init__()
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
            print(x.shape)
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


def get_fusion_module(in_channels):

    layers = []
    config = [[512, 3, 1, 1],
              [512, 3, 1, 1],
              [512, 3, 1, 1]]

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


def get_pyramid_module(input_size):

    layers = []

    layers += [BasicConv(in_channels=(512 + 128 + 32),
                         out_channels=512,
                         kernel_size=3,
                         stride=1,
                         padding=1)]

    in_channels = 512
    out_channels = 512

    while input_size > 1:
        print('pyramid', input_size)

        if input_size == 3:
            input_size //= 2
            layers += [BasicConv(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 stride=2,
                                 padding=0)]

        else:
            input_size = ceil(input_size / 2)
            layers += [BasicConv(in_channels=in_channels,
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)]

        in_channels = out_channels
        out_channels = 256

    return layers


def multibox(input_size,
             class_count):

    class_layers = []
    loc_layers = []

    i = 0
    num_anchors = 6
    in_channels = 512

    while input_size > 0:
        print('multibox', input_size)

        if i == 2:
            in_channels = 256

        if input_size <= 3:
            num_anchors = 4

        class_layers += [nn.Conv2d(in_channels=in_channels,
                                   out_channels=num_anchors * class_count,
                                   kernel_size=3,
                                   padding=1)]
        loc_layers += [nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_anchors * 4,
                                 kernel_size=3,
                                 padding=1)]

        if input_size <= 3:
            input_size //= 2
        else:
            input_size = ceil(input_size / 2)
        i += 1

    # class_layers += [nn.Conv2d(in_channels=in_channels,
    #                            out_channels=num_anchors * class_count,
    #                            kernel_size=3,
    #                            padding=1)]
    # loc_layers += [nn.Conv2d(in_channels=in_channels,
    #                          out_channels=num_anchors * 4,
    #                          kernel_size=3,
    #                          padding=1)]

    return class_layers, loc_layers


fusion_in_channels = {
    'basic': [128, 256, 512],
    'bottleneck': [512, 1024, 2048]
}


def build_SFDetV2ResNet(mode,
                        new_size,
                        resnet_model,
                        anchors,
                        class_count,
                        model_save_path,
                        pretrained_model,
                        output_txt):

    in_channels = fusion_in_channels['basic']

    if resnet_model == '18':
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif resnet_model == '34':
        base = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
    elif resnet_model == '50':
        in_channels = fusion_in_channels['bottleneck']
        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif resnet_model == '101':
        in_channels = fusion_in_channels['bottleneck']
        base = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    elif resnet_model == '152':
        in_channels = fusion_in_channels['bottleneck']
        base = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)

    fusion_module = get_fusion_module(in_channels=in_channels)

    pyramid_module = get_pyramid_module(input_size=new_size // 8)

    if pretrained_model is not None:
        head = multibox(input_size=new_size // 8,
                        class_count=81)
        model = SFDetV2ResNet(mode=mode,
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
        head = multibox(input_size=new_size // 8,
                        class_count=class_count)
        model = SFDetV2ResNet(mode=mode,
                              base=base,
                              fusion_module=fusion_module,
                              pyramid_module=pyramid_module,
                              head=head,
                              anchors=anchors,
                              class_count=class_count)

    return model
