# -*- coding: utf-8 -*- 
"""
@project: 201904_dual_shot
@file: augment.py
@author: danna.li
@time: 2019-06-03 11:33
@description: 
"""
from __future__ import division
import torch
from torchvision import transforms
import cv2
import numpy as np
from numpy import random
import math

from PIL import Image

Image.MAX_IMAGE_PIXELS = 100000000000


def intersect(box_a, box_b):
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class ConvertFromInts(object):
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToAbsoluteCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels


class ToPercentCoords(object):
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels


class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        # print (image.shape[0]/self.size)
        image = cv2.resize(image, (self.size,
                                   self.size))
        return image, boxes, labels


class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, boxes, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, boxes, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, boxes, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, boxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        return image, boxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return image, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if random.randint(5):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)


        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels


class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes


class SwapChannels(object):
    """Transforms a tensorized image by swapping the channels in the order
     specified in the swap tuple.
    Args:
        swaps (int triple): final order of channels
            eg: (2, 1, 0)
    """

    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, image):
        """
        Args:
            image (Tensor): image tensor to be transformed
        Return:
            a tensor with channels swapped according to swap
        """
        # if torch.is_tensor(image):
        #     image = image.data.cpu().numpy()
        # else:
        #     image = np.array(image)
        image = image[:, :, self.swaps]
        return image


class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)

        return self.rand_light_noise(im, boxes, labels)


class RandomSSDCrop(object):
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.5,.7,.9
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        """
        :param image: 3-d array,channel last
        :param boxes: 2-d array,(num_gt,(x1,y1,x2,y2)
        :param labels: 1-d array(num_gt)
        :return:
        """
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                  rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                  rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels


class RandomBaiduCrop(object):
    def __init__(self, size):
        self.maxSize = 12000  # max size
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        """
        :param image: 3-d array,channel last
        :param boxes: 2-d array,(num_gt,(x1,y1,x2,y2)
        :param labels: 1-d array(num_gt)
        :return:
        """
        # resize original image and transfer the gt accordingly
        height, width, _ = image.shape
        box_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
        # print(boxes,box_area)
        rand_idx = random.randint(len(box_area))
        side_len = box_area[rand_idx] ** 0.5

        anchors = np.array([16, 32, 64, 128, 256, 512])
        distances = abs(anchors - side_len)
        anchor_idx = np.argmin(distances)
        target_anchor = random.choice(anchors[0:min(anchor_idx + 1, 5) + 1])

        ratio = float(target_anchor) / side_len
        # print('ratio:', ratio)
        ratio = ratio * (2 ** random.uniform(-1, 1))
        if int(height * ratio * width * ratio) > self.maxSize * self.maxSize:
            ratio = (self.maxSize * self.maxSize / (height * width)) ** 0.5
        # print('ratio:', ratio)

        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = random.choice(interp_methods)
        image = cv2.resize(image, None, None, fx=ratio, fy=ratio, interpolation=interp_method)
        boxes = boxes * ratio

        # randomly select 50 crop box which covers the selected gt
        height, width, _ = image.shape
        sample_boxes = []
        gt_x1, gt_y1, gt_x2, gt_y2 = boxes[rand_idx, :]
        crop_w = crop_h = self.size

        # randomly select a crop box
        if crop_w < max(height, width):
            crop_x1 = random.uniform(gt_x2 - crop_w, gt_x1)
            crop_y1 = random.uniform(gt_y2 - crop_h, gt_y1)
        else:
            crop_x1 = random.uniform(width - crop_w, 0)
            crop_y1 = random.uniform(height - crop_h, 0)
        crop_x1 = math.floor(crop_x1)
        crop_y1 = math.floor(crop_y1)
        choice_box = np.array([int(crop_x1), int(crop_y1), int(crop_x1 + crop_w), int(crop_y1 + crop_h)])

        # perform crop, keep gts with centers lying inside the cropped box
        pil_img = Image.fromarray(image.astype(np.uint8))
        current_image = np.array(pil_img.crop([i for i in choice_box]))

        c_xs = (boxes[:, 0] + boxes[:, 2]) * 0.5
        c_ys = (boxes[:, 1] + boxes[:, 3]) * 0.5
        m1 = (choice_box[0] < c_xs) * (c_xs < choice_box[2])
        m2 = (choice_box[1] < c_ys) * (c_ys < choice_box[3])
        mask = m1 * m2
        current_boxes = boxes[mask, :].copy()
        current_labels = labels[mask]

        current_boxes[:, :2] = np.maximum(current_boxes[:, :2], choice_box[:2])  # make sure gt is inside the crop
        current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], choice_box[2:])
        current_boxes[:, :2] -= choice_box[:2]
        current_boxes[:, 2:] -= choice_box[:2]
        return current_image, current_boxes, current_labels


class Augmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        # follow pyramid box augment
        if False:
            self.augment = Compose([
                ConvertFromInts(),
                # ToAbsoluteCoords(),
                PhotometricDistort(),
                # Expand(self.mean),
                RandomBaiduCrop(self.size),
                RandomMirror(),
                # ToPercentCoords(),
                # Resize(self.size),
                SubtractMeans(self.mean)
            ])
        # follow ssd augment
        else:
            self.augment = Compose([
                ConvertFromInts(),
                # ToAbsoluteCoords(),
                PhotometricDistort(),
                Expand(self.mean),
                RandomSSDCrop(),
                RandomMirror(),
                ToPercentCoords(),
                Resize(self.size),
                SubtractMeans(self.mean),
                ToAbsoluteCoords()
            ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


def aug_test():
    import skimage
    from skimage import io
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    def load_image(image_path):
        """加载图像
        :param image_path: 图像路径
        :return: [h,w,3] numpy数组
        """
        image = io.imread(image_path)
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image[..., :3]

    def plot_anchor(img_array, anchor_list):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(img_array.astype(int))
        for a in anchor_list:
            x1, y1, x2, y2 = [int(i) for i in a]
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.show()

    path = 'C:\\Users\\lidan\\OneDrive\\A_markdown\\public\\face\\code\\201904_dual_shot\\img\\000372.jpg'
    img = load_image(path)
    print('original image shape:', img.shape)
    size = 640
    labels = np.array([0, 0])
    boxes = np.array([[78., 22., 253., 304.], [269., 19., 360., 296.]])

    to_aug = Augmentation(size)
    img, boxes, labels = to_aug(img, boxes, labels)
    print(img.shape, np.sum(img), np.max(img), np.min(img), boxes, labels)
    plot_anchor(img, boxes)


if __name__ == '__main__':
    aug_test()
