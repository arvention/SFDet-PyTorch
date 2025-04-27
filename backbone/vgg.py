import torch.nn as nn


class VGG(object):
    """VGG classification architecture"""

    def __init__(self,
                 in_channels,
                 batch_norm=False):
        """Class constructor for VGG

        Arguments:
            config {string} -- represents the model configuration to be loaded
            as layers of the model
            in_channels {int} -- number of input channels for the first conv
            layer

        Keyword Arguments:
            batch_norm {bool} -- determines if the model uses batch
            normalization or not (default: {False})
        """

        super(VGG, self).__init__()
        self.config = [64, 64, 'M',
                       128, 128, 'M',
                       256, 256, 256, 'C',
                       512, 512, 512, 'M',
                       512, 512, 512]
        self.in_channels = in_channels
        self.batch_norm = batch_norm
        self.layers = self.get_layers()

    def get_layers(self):
        """Forms the layers of the model based on self.config

        Returns:
            list -- contains all layers of VGG model based on self.config
        """

        layers = []
        in_channels = self.in_channels

        for channels in self.config:
            if channels == 'M':
                layers += [nn.MaxPool2d(kernel_size=2,
                                        stride=2)]
            elif channels == 'C':
                layers += [nn.MaxPool2d(kernel_size=2,
                                        stride=2,
                                        ceil_mode=True)]
            else:
                conv = nn.Conv2d(in_channels=in_channels,
                                 out_channels=channels,
                                 kernel_size=3,
                                 padding=1)
                if self.batch_norm:
                    layers += [conv,
                               nn.BatchNorm2d(num_features=channels),
                               nn.ReLU(inplace=True)]
                else:
                    layers += [conv,
                               nn.ReLU(inplace=True)]
                in_channels = channels

        pool5 = nn.MaxPool2d(kernel_size=3,
                             stride=1,
                             padding=1)

        conv6 = nn.Conv2d(in_channels=512,
                          out_channels=1024,
                          kernel_size=3,
                          padding=6,
                          dilation=6)

        conv7 = nn.Conv2d(in_channels=1024,
                          out_channels=1024,
                          kernel_size=1)

        layers += [pool5,
                   conv6,
                   nn.ReLU(inplace=True),
                   conv7,
                   nn.ReLU(inplace=True)]

        return layers
