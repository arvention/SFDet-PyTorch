import os
import torch
import os.path as osp
import torch.nn as nn
from backbone.vgg import VGG
import torch.nn.functional as F
from layers.l2_norm import L2Norm
from utils.init import xavier_init
from layers.detection import Detect


class SSD(nn.Module):

    """
    SSD architecture
    """

    def __init__(self,
                 mode,
                 base,
                 extras,
                 head,
                 anchors,
                 class_count):
        super(SSD, self).__init__()

        self.mode = mode
        self.base = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
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
        # apply vgg up to conv4_3 relu
        for i in range(23):
            x = self.base[i](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for i in range(23, len(self.base)):
            x = self.base[i](x)
        sources.append(x)

        # apply extras
        for i, layer in enumerate(self.extras):
            x = F.relu(layer(x), inplace=True)
            if i % 2 == 1:
                sources.append(x)

        # apply multibox head to sources
        for (x, c, l) in zip(sources, self.class_head, self.loc_head):
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

    def init_weights(self, model_save_path, basenet):
        if basenet:
            weights_path = osp.join(model_save_path, basenet)
            vgg_weights = torch.load(weights_path)
            self.base.load_state_dict(vgg_weights)
        else:
            self.base.apply(fn=xavier_init)
        self.extras.apply(fn=xavier_init)
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


def get_extras(config, in_channels, batch_norm=False):
    layers = []
    flag = False

    for i, out_channels in enumerate(config):
        if in_channels != 'S':
            if out_channels == 'S':
                layers += [nn.Conv2d(in_channels=in_channels,
                                     out_channels=config[i + 1],
                                     kernel_size=(1, 3)[flag],
                                     stride=2,
                                     padding=1)]
            else:
                layers += [nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = out_channels

    return layers


def multibox(config, base, extra_layers, class_count):
    class_layers = []
    loc_layers = []
    vgg_source = [21, -2]

    for k, v in enumerate(vgg_source):
        class_layers += [nn.Conv2d(in_channels=base.layers[v].out_channels,
                                   out_channels=config[k] * class_count,
                                   kernel_size=3,
                                   padding=1)]
        loc_layers += [nn.Conv2d(in_channels=base.layers[v].out_channels,
                                 out_channels=config[k] * 4,
                                 kernel_size=3,
                                 padding=1)]

    for k, v in enumerate(extra_layers[1::2], start=2):
        class_layers += [nn.Conv2d(in_channels=v.out_channels,
                                   out_channels=config[k] * class_count,
                                   kernel_size=3,
                                   padding=1)]
        loc_layers += [nn.Conv2d(in_channels=v.out_channels,
                                 out_channels=config[k] * 4,
                                 kernel_size=3,
                                 padding=1)]

    return class_layers, loc_layers


extras_config = {
    '300': [256, 'S',
            512, 128, 'S',
            256, 128, 256, 128, 256],
    '512': [256, 'S',
            512, 128, 'S',
            256, 128, 'S',
            256, 128, 'S',
            256, 128, 'S',
            256]
}
mbox_config = {
    '300': [4, 6, 6, 6, 4, 4],
    '512': [4, 6, 6, 6, 6, 4, 4]
}


def build_SSD(mode,
              new_size,
              anchors,
              class_count):

    base = VGG(in_channels=3)

    extras = get_extras(config=extras_config[str(new_size)],
                        in_channels=1024)

    head = multibox(config=mbox_config[str(new_size)],
                    base=base,
                    extra_layers=extras,
                    class_count=class_count)

    return SSD(mode,
               base.layers,
               extras,
               head,
               anchors,
               class_count)
