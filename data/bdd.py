import os
import cv2
import json
import torch
import pickle
import numpy as np
import os.path as osp
from torch.utils.data import Dataset
from utils.genutils import write_print


BDD_CLASSES = ('pedestrian', 'rider', 'car', 'truck', 'bus', 'train',
               'motorcycle', 'bicycle', 'traffic light', 'traffic sign')


class BDDAnnotationTransform(object):
    """This class converts the data from Annotation JSON files to a list
    of [xmin, ymin, xmax, ymax, class]"""

    def __init__(self):
        """Class constructor for BDDAnnotationTransform"""
        super(BDDAnnotationTransform, self).__init__()

    def __call__(self,
                 targets,
                 width,
                 height):
        """Executed when the class is called as a function

        Arguments:
            targets {list} -- list of targets
            width {int} -- Width of the corresponding image; Used for scaling
            the coordinates of bounding boxes
            height {int} -- Height of the corresponding image; Used for
            scaling the coordinates of bounding boxes

        Returns:
            list -- list of bounding boxes formatted as
            [xmin, ymin, xmax, ymax, class]
        """

        labels = []

        for target in targets:
            bbox = [(float(target[0])) / width,
                    (float(target[1])) / height,
                    (float(target[2])) / width,
                    (float(target[3])) / height,
                    int(target[4])]
            labels += [bbox]

        return labels


class BDD(Dataset):
    """BDD dataset

    Extends:
        Dataset
    """

    def __init__(self,
                 data_path,
                 version,
                 new_size,
                 mode,
                 image_transform):
        """Class constructor for BDD100K

        Arguments:
            data_path {string} -- path to the dataset
            version {string} -- either 10k or 100k
            new_size {int} -- new height and width of the image
            mode {string} -- experiment mode - either train or test
            image_transform {object} -- produces different dataset
            augmentation techniques
        """

        super(BDD, self).__init__()

        self.data_path = data_path
        self.version = version
        self.new_size = new_size
        self.mode = mode
        self.image_transform = image_transform
        self.target_transform = BDDAnnotationTransform()
        class_to_index = dict(zip(BDD_CLASSES, range(len(BDD_CLASSES))))

        if self.mode == 'test':
            self.mode = 'val'

        self.annotation_path = osp.join(self.data_path,
                                        'labels',
                                        'det_20',
                                        'det_' + self.mode + '.json')
        self.image_path = osp.join(self.data_path,
                                   'images',
                                   self.version,
                                   self.mode,
                                   '{}.jpg')

        self.ids = []
        self.dict_targets = {}
        with open(self.annotation_path) as file:
            data = json.load(file)
            for instance in data:
                image_id = instance['name'].split('.')[0].strip()

                if 'labels' in instance:
                    self.ids.append(image_id)
                    self.dict_targets[image_id] = []

                    for label in instance['labels']:
                        if label['category'] in BDD_CLASSES:
                            x_min = label['box2d']['x1']
                            y_min = label['box2d']['y1']
                            x_max = label['box2d']['x2']
                            y_max = label['box2d']['y2']
                            mapped_class = class_to_index[label['category']]
                            bbox = [x_min, y_min, x_max, y_max, mapped_class]
                            self.dict_targets[image_id] += [bbox]
                else:
                    print(image_id)

    def __len__(self):
        """Returns the number of images in the dataset

        Returns:
            int -- number of images in the dataset
        """

        return len(self.ids)

    def __getitem__(self,
                    index):
        """Gets the image and its corresponding annotation found in
        position index of the list of images in the dataset

        Arguments:
            index {int} -- index of the image in the list of images in the
            dataset

        Returns:
            torch.Tensor, np.ndarray, -- tensor representation of the image,
            list of bounding boxes of objects in the image formatted as
            [xmin, ymin, xmax, ymax, class]
        """

        image, target, _, _ = self.pull_item(index)
        return image, target

    def pull_item(self,
                  index):
        """Gets the image found in position index of the list of images in the
        dataset together with its corresponding annotation, its height, and its
        width

        Arguments:
            index {int} -- index of the image in the list of images in the
            dataset

        Returns:
            torch.Tensor, np.ndarray, int, int -- tensor representation of
            the image, list of bounding boxes of objects in the image
            formatted as [xmin, ymin, xmax, ymax, class], height, width
        """

        image_id = self.ids[index]

        # target_path = self.annotation_path.format(image_id)
        image_path = self.image_path.format(image_id)

        image = cv2.imread(image_path)
        height, width, _ = image.shape

        target = self.dict_targets[image_id]
        target = self.target_transform(target, width, height)

        if self.image_transform is not None:
            target = np.array(target)
            boxes = target[:, :4]
            labels = target[:, 4]
            image, boxes, labels = self.image_transform(image, boxes, labels)

            # to rgb
            image = image[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return torch.from_numpy(image).permute(2, 0, 1), target, height, width

    def pull_image(self,
                   index):
        """Gets the image found in position index of the list of images in the
        dataset

        Arguments:
            index {int} -- index of the image in the list of images in the
            dataset

        Returns:
            np.ndarray -- image
        """

        image_id = self.ids[index]
        image_path = self.image_path.format(image_id)
        return cv2.imread(image_path, cv2.IMREAD_COLOR)

    def pull_annotation(self,
                        index):
        """Gets the annotation of the image found in position index of the
        list of images in the dataset. The coordinates of the annotation is
        not scaled by the height and width of the image

        Arguments:
            index {int} -- index of the image in the list of images in the
            dataset

        Returns:
            string, list -- id of the image, list of bounding boxes of objects
            in the image formatted as [xmin, ymin, xmax, ymax, class]
        """

        image_id = self.ids[index]
        target = self.dict_targets[image_id]
        target = self.target_transform(target, 1, 1)

        return image_id, target

    def pull_tensor(self,
                    index):
        """Gets a tensor of the image found in position index of the list of
        images in the dataset

        Arguments:
            index {int} -- index of the image in the list of images in the
            dataset

        Returns:
            torch.Tensor -- tensor representation of the image
        """

        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)


def save_results(all_boxes,
                 dataset,
                 results_path,
                 output_txt):

    # for each class
    for class_i, class_name in enumerate(BDD_CLASSES):

        text = 'Writing {:s} BDD results file'.format(class_name)
        write_print(output_txt, text)
        filename = osp.join(results_path, class_name + '.txt')

        with open(filename, 'wt') as f:

            # get detections for the class in an image
            for image_i, image_id in enumerate(dataset.ids):
                detections = all_boxes[class_i + 1][image_i]
                # print('HELLO', detections)

                # if there are detections for the class in the image
                if len(detections) != 0:
                    for k in range(detections.shape[0]):
                        output = '{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'

                        # the VOCdevkit expects 1-based indices
                        output = output.format(image_id,
                                               detections[k, -1],
                                               detections[k, 0],
                                               detections[k, 1],
                                               detections[k, 2],
                                               detections[k, 3])

                        f.write(output)


def parse_annotation(instance_dict):
    """ Parse a BDD JSON file """
    objects = []

    if 'labels' in instance_dict:
        obj_struct = {}
        for label in instance_dict['labels']:
            if label['category'] in BDD_CLASSES:
                obj_struct['name'] = label['category']
                obj_struct['bbox'] = [float(label['box2d']['x1']),
                                      float(label['box2d']['y1']),
                                      float(label['box2d']['x2']),
                                      float(label['box2d']['y2'])]
                objects.append(obj_struct)

    return objects


def voc_ap(recall,
           precision,
           use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for threshold in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= threshold) == 0:
                threshold_precision = 0
            else:
                threshold_precision = np.max(precision[recall >= threshold])
            ap = ap + threshold_precision / 11.

    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def voc_eval(detection_path,
             annotation_file,
             class_name,
             cache_dir,
             output_txt,
             iou_threshold=0.5,
             use_07_metric=True):

    # create or get the cache_file
    if not osp.isdir(cache_dir):
        os.mkdir(cache_dir)
    cache_file = osp.join(cache_dir, 'annotations.pkl')

    # if cache_file does not exists
    if not osp.isfile(cache_file):
        targets = {}

        # per image, read annotations from XML file
        write_print(output_txt, 'Reading annotations')
        with open(annotation_file) as file:
            data = json.load(file)
            for instance in data:
                image_name = instance['name'].split('.')[0].strip()
                targets[image_name] = parse_annotation(instance)

        # save annotations to cache_file
        temp_string = 'Saving cached annotations to {:s}\n'.format(cache_file)
        write_print(output_txt, temp_string)
        with open(cache_file, 'wb') as f:
            pickle.dump(targets, f)

    # else if cache_file exists
    else:
        with open(cache_file, 'rb') as f:
            targets = pickle.load(f)

    image_names = [x for x in targets.keys()]

    class_targets = {}
    n_positive = 0

    # get targets for objects with class equal to class_name in image_name
    for image_name in image_names:
        target = [x for x in targets[image_name] if x['name'] == class_name]
        bbox = np.array([x['bbox'] for x in target])
        det = [False] * len(target)
        n_positive += len(target)
        class_targets[image_name] = {'bbox': bbox,
                                     'det': det}

    # read detections from class_name.txt
    detection_file = detection_path.format(class_name)
    with open(detection_file, 'r') as f:
        lines = f.readlines()

    # if there are detections
    if any(lines) == 1:

        # get ids, confidences, and bounding boxes
        values = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in values]
        confidences = np.array([float(x[1]) for x in values])
        bboxes = np.array([[float(z) for z in x[2:]] for x in values])

        # sort by confidence
        sorted_index = np.argsort(-confidences)
        bboxes = bboxes[sorted_index, :]
        image_ids = [image_ids[x] for x in sorted_index]

        num_detections = len(image_ids)
        tp = np.zeros(num_detections)
        fp = np.zeros(num_detections)

        # go through detections and mark TPs and FPs
        for i in range(num_detections):

            # get target bounding box
            image_target = class_targets[image_ids[i]]
            bbox_target = image_target['bbox'].astype(float)

            # get detected bounding box
            bbox = bboxes[i, :].astype(float)
            overlap_max = -np.inf

            if bbox_target.size > 0:

                # get the overlapping region
                # compute the area of intersection
                x_min = np.maximum(bbox_target[:, 0], bbox[0])
                y_min = np.maximum(bbox_target[:, 1], bbox[1])
                x_max = np.minimum(bbox_target[:, 2], bbox[2])
                y_max = np.minimum(bbox_target[:, 3], bbox[3])
                width = np.maximum(x_max - x_min, 0.)
                height = np.maximum(y_max - y_min, 0.)
                intersection = width * height

                # get the area of the gt and the detection
                # compute the union
                area_bbox = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                area_bbox_target = ((bbox_target[:, 2] - bbox_target[:, 0])
                                    * (bbox_target[:, 3] - bbox_target[:, 1]))
                union = area_bbox + area_bbox_target - intersection

                # compute the iou
                iou = intersection / union
                overlap_max = np.max(iou)
                j_max = np.argmax(iou)

            # if the maximum overlap is over the overlap threshold
            if overlap_max > iou_threshold:
                # if it is not yet detected, count as a true positive
                if not image_target['det'][j_max]:
                    tp[i] = 1.
                    image_target['det'][j_max] = 1
                # else, count as a false positive
                else:
                    fp[i] = 1.

            # else, count as a false positive
            else:
                fp[i] = 1.

        # compute precision and recall
        # avoid divide by zero if the first detection matches a difficult gt
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / float(n_positive)
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        ap = voc_ap(recall=recall,
                    precision=precision,
                    use_07_metric=use_07_metric)

    else:
        recall = -1.
        precision = -1.
        ap = -1.

    return recall, precision, ap


def do_python_eval(results_path,
                   dataset,
                   output_txt,
                   mode,
                   iou_threshold,
                   use_07_metric):

    # annotation cache directory
    cache_dir = osp.join(results_path,
                         'annotations_cache')

    # path to JSON annotation file
    annotation_file = dataset.annotation_path

    # The PASCAL VOC metric changed in 2010
    write_print(output_txt, '\nVOC07 metric? '
                + ('Yes\n' if use_07_metric else 'No\n'))

    # for each class, compute the recall, precision, and ap
    aps = []
    for class_name in BDD_CLASSES:
        detection_path = osp.join(results_path, class_name + '.txt')
        recall, precision, ap = voc_eval(detection_path=detection_path,
                                         annotation_file=annotation_file,
                                         class_name=class_name,
                                         cache_dir=cache_dir,
                                         output_txt=output_txt,
                                         iou_threshold=iou_threshold,
                                         use_07_metric=use_07_metric)
        aps += [ap]

        write_print(output_txt, 'AP for {} = {:.4f}'.format(class_name, ap))

        pickle_file = osp.join(results_path, class_name + '_pr.pkl')
        with open(pickle_file, 'wb') as f:
            pickle.dump({'rec': recall, 'prec': precision, 'ap': ap}, f)

    write_print(output_txt, 'Mean AP = {:.4f}'.format(np.mean(aps)))

    return aps, np.mean(aps)
