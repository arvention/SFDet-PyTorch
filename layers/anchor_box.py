import torch
from math import sqrt, ceil


anchors_config = {
    'SSD-300': [[8, 16, 32, 64, 100, 300],
                [38, 19, 10, 5, 3, 1],
                [[2], [2, 3], [2, 3], [2, 3], [2], [2]]],
    'SSD-512': [[8, 16, 32, 64, 128, 256, 512],
                [64, 32, 16, 8, 4, 2, 1],
                [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]],
    'STDN-300': [[8, 16, 32, 64, 100, 300],  # change this
                 [36, 18, 9, 5, 3, 1],
                 [[1.6, 2, 3], [1.6, 2, 3], [1.6, 2, 3], [1.6, 2, 3],
                  [1.6, 2, 3], [1.6, 2, 3]]],
    'SFDet-300': [[8, 16, 32, 64, 100, 300],
                  [36, 18, 9, 5, 3, 1],
                  [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]],
    'SFDet-512': [[8, 16, 32, 64, 128, 256, 512],
                  [64, 32, 16, 8, 4, 2, 1],
                  [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]]
}


def get_steps(input_size,
              map_sizes):
    steps = []

    for map_size in map_sizes:
        steps += [input_size / map_size]

    return steps


def get_map_sizes(input_size):
    map_sizes = []
    while input_size > 1:
        map_sizes += [input_size]
        if input_size == 3:
            input_size //= 2
        else:
            input_size = ceil(input_size / 2)
    map_sizes += [1]
    return map_sizes


class AnchorBox(object):
    """Anchor box"""

    def __init__(self,
                 new_size,
                 config,
                 scale_initial,
                 scale_min,
                 scale_max):
        """Class constructor for AnchorBox

        Arguments:
            new_size {int} -- width & height of images
            config {string} -- represents the configuration of the anchor boxes
            to be loaded. The loaded configuration contains 3 lists - (1) the
            steps, (2) the sizes of output feature maps, and (3) the list of
            aspect ratios per feature map
        """

        super(AnchorBox, self).__init__()

        self.new_size = new_size
        self.map_sizes = get_map_sizes(new_size // 8)
        self.steps = get_steps(new_size,
                               self.map_sizes)
        self.aspect_ratios = [[2, 3] for i in range(len(self.map_sizes) - 2)]
        self.aspect_ratios += [[2], [2]]

        self.scales = self.get_scales(scale_initial=scale_initial,
                                      scale_min=scale_min,
                                      scale_max=scale_max)

    def get_scales(self,
                   scale_initial=0.1,
                   scale_min=0.2,
                   scale_max=1.05):
        """Computes the scales used in the computation of anchor boxes

        Keyword Arguments:
            scale_min {float} -- [description] (default: {0.2})
            scale_max {float} -- [description] (default: {1.05})

        Returns:
            list -- contains the scales
        """

        scales = [scale_initial]
        num_map_sizes = len(self.map_sizes)
        scale_diff = scale_max - scale_min

        for k in range(1, num_map_sizes + 1):
            scale = scale_min + scale_diff / (num_map_sizes - 1) * (k - 1)
            scales.append(scale)

        return scales

    def get_boxes(self):
        """Computes location of anchor boxes in center-offset form for each
        feature map

        Returns:
            tensor -- anchor boxes
        """

        boxes = []

        for k, map_size in enumerate(self.map_sizes):
            num_elements = map_size ** 2
            size = self.new_size / self.steps[k]
            for i in range(num_elements):
                row = i // map_size
                col = i % map_size

                cx = (col + 0.5) / size
                cy = (row + 0.5) / size

                scale = self.scales[k]
                scale_next = self.scales[k + 1]
                scale_next = sqrt(scale * scale_next)

                boxes += [cx, cy, scale, scale]
                boxes += [cx, cy, scale_next, scale_next]

                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    boxes += [cx, cy, scale * ratio, scale / ratio]
                    boxes += [cx, cy, scale / ratio, scale * ratio]

        output = torch.Tensor(boxes).view(-1, 4)
        output.clamp_(max=1, min=0)
        return output
