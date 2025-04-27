import os
import torch
import torch.nn as nn
import os.path as osp
from backbone.vgg import VGG
from layers.block import BasicConv
from utils.init import xavier_init
from layers.detection import Detect
from utils.genutils import load_pretrained_model


class SFDetVGG(nn.Module):

    def __init__(self,
                 mode,
                 base,
                 extras,
                 fusion_module,
                 pyramid_module,
                 head,
                 anchors,
                 class_count):

        super(SFDetVGG, self).__init__()
        self.mode = mode
        self.base = nn.ModuleList(modules=base)
        self.extras = nn.ModuleList(modules=extras)
        self.fusion_module = nn.ModuleList(modules=fusion_module)
        self.batch_norm = nn.BatchNorm2d(num_features=(512 + 128 + 32))
        self.pyramid_module = nn.ModuleList(modules=pyramid_module)

        self.class_head = nn.ModuleList(modules=head[0])
        self.loc_head = nn.ModuleList(modules=head[1])
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

        # apply vgg up to conv4_3 relu
        for i in range(23):
            x = self.base[i](x)
        sources.append(x)

        # apply vgg up to fc7
        for i in range(23, len(self.base)):
            x = self.base[i](x)
        sources.append(x)

        # pass through extra layers
        for layer in self.extras:
            x = layer(x)
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

        if base_network:
            weights_path = osp.join(model_save_path, base_network)
            vgg_weights = torch.load(weights_path)
            self.base.load_state_dict(vgg_weights)

        else:
            self.base.apply(fn=xavier_init)

        self.extras.apply(fn=xavier_init)
        self.fusion_module.apply(fn=xavier_init)
        self.pyramid_module.apply(fn=xavier_init)
        self.class_head.apply(fn=xavier_init)
        self.loc_head.apply(fn=xavier_init)

    def load_weights(self,
                     base_file):

        other, ext = os.path.splitext(base_file)

        if ext == '.pkl' or ext == '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                            map_location=lambda storage,
                                            loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def get_extras(in_channels,
               batch_norm=False):

    layers = []

    layers += [nn.Conv2d(in_channels=in_channels,
                         out_channels=256,
                         kernel_size=(1, 1),
                         stride=(1, 1))]
    layers += [nn.ReLU(inplace=True)]

    layers += [nn.Conv2d(in_channels=256,
                         out_channels=512,
                         kernel_size=(3, 3),
                         stride=(1, 1))]
    layers += [nn.ReLU(inplace=True)]

    layers += [nn.Conv2d(in_channels=512,
                         out_channels=128,
                         kernel_size=(1, 1),
                         stride=(1, 1))]
    layers += [nn.ReLU(inplace=True)]

    layers += [nn.Conv2d(in_channels=128,
                         out_channels=256,
                         kernel_size=(3, 3),
                         stride=(2, 2),
                         padding=(1, 1))]
    layers += [nn.ReLU(inplace=True)]

    return layers


def get_fusion_module(config,
                      base,
                      extras):

    layers = []
    # conv4_3
    layers += [BasicConv(in_channels=base[24].out_channels,
                         out_channels=config[0][0],
                         kernel_size=config[0][1],
                         stride=config[0][2],
                         padding=config[0][3])]
    # fc_7
    layers += [nn.Sequential(BasicConv(in_channels=base[-2].out_channels,
                                       out_channels=config[1][0],
                                       kernel_size=config[1][1],
                                       stride=config[1][2],
                                       padding=config[1][3]),
                             nn.PixelShuffle(upscale_factor=2))]

    layers += [nn.Sequential(BasicConv(in_channels=extras[-2].out_channels,
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


fusion_config = {
    '300': [[512, 3, 1, 0],
            [512, 2, 1, 0],
            [512, 1, 1, 0]],

    '512': [[512, 3, 1, 1],
            [512, 3, 1, 1],
            [512, 2, 1, 1]]
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


def build_SFDetVGG(mode,
                   new_size,
                   anchors,
                   class_count,
                   model_save_path,
                   pretrained_model,
                   output_txt):

    base = VGG(in_channels=3)

    extras = get_extras(in_channels=1024)

    fusion_module = get_fusion_module(config=fusion_config[str(new_size)],
                                      base=base.layers,
                                      extras=extras)

    pyramid_module = get_pyramid_module(config=pyramid_config[str(new_size)])

    if pretrained_model is not None:
        head = multibox(config=mbox_config[str(new_size)],
                        class_count=81)
        model = SFDetVGG(mode=mode,
                         base=base.layers,
                         extras=extras,
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
        model = SFDetVGG(mode=mode,
                         base=base.layers,
                         extras=extras,
                         fusion_module=fusion_module,
                         pyramid_module=pyramid_module,
                         head=head,
                         anchors=anchors,
                         class_count=class_count)

    return model
