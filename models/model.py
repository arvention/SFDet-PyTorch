from models.ssd import build_SSD
from models.rssd import build_RSSD
from models.stdn import build_STDN
from models.stdn2 import build_STDN2
from models.rssdv2 import build_RSSDv2
from models.rssd_1c import build_RSSD_1C
from models.sfdet_vgg import build_SFDetVGG
from models.sfdet_resnet import build_SFDetResNet
from models.sfdet_resnext import build_SFDetResNeXt
from models.sfdet_densenet import build_SFDetDenseNet
# from torchvision.models.detection import ssd300_vgg16
# from torchvision.models.detection.ssd import SSDHead
# from torchvision.models.detection._utils import retrieve_out_channels


def get_model(config, anchors):
    """
    returns the model
    """

    model = None

    if config['model'] == 'SFDet-VGG':
        model = build_SFDetVGG(mode=config['mode'],
                               new_size=config['new_size'],
                               anchors=anchors,
                               class_count=config['class_count'])

    elif config['model'] == 'SFDet-ResNet':
        model = build_SFDetResNet(mode=config['mode'],
                                  new_size=config['new_size'],
                                  resnet_model=config['resnet_model'],
                                  anchors=anchors,
                                  class_count=config['class_count'])

    elif config['model'] == 'SFDet-DenseNet':
        model = build_SFDetDenseNet(mode=config['mode'],
                                    new_size=config['new_size'],
                                    densenet_model=config['densenet_model'],
                                    anchors=anchors,
                                    class_count=config['class_count'])

    elif config['model'] == 'SFDet-ResNeXt':
        model = build_SFDetResNeXt(mode=config['mode'],
                                   new_size=config['new_size'],
                                   resnext_model=config['resnext_model'],
                                   anchors=anchors,
                                   class_count=config['class_count'])

    elif config['model'] == 'SSD':
        # num_classes = config['class_count']
        # model = ssd300_vgg16(pretrained=True,
        #                      trainable_backbone_layers=5)
        # in_channels = retrieve_out_channels(model.backbone, (300, 300))
        # num_anchors = model.anchor_generator.num_anchors_per_location()
        # model.head = SSDHead(in_channels, num_anchors, num_classes)
        # model.transform = None
        model = build_SSD(mode=config['mode'],
                          new_size=config['new_size'],
                          anchors=anchors,
                          class_count=config['class_count'])

    elif config['model'] == 'RSSD':
        model = build_RSSD(mode=config['mode'],
                           new_size=config['new_size'],
                           anchors=anchors,
                           class_count=config['class_count'])

    elif config['model'] == 'RSSDv2':
        model = build_RSSDv2(mode=config['mode'],
                             new_size=config['new_size'],
                             anchors=anchors,
                             class_count=config['class_count'])

    elif config['model'] == 'RSSD_1C':
        model = build_RSSD_1C(mode=config['mode'],
                              new_size=config['new_size'],
                              anchors=anchors,
                              class_count=config['class_count'])

    elif config['model'] == 'STDN':
        model = build_STDN(mode=config['mode'],
                           new_size=config['new_size'],
                           anchors=anchors,
                           class_count=config['class_count'])

    elif config['model'] == 'STDN2':
        model = build_STDN2(mode=config['mode'],
                            new_size=config['new_size'],
                            anchors=anchors,
                            class_count=config['class_count'])

    return model
