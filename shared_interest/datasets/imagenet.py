"""Dataset for ImageNet with annotations."""

import os
import xml.etree.ElementTree as ET
import torch
from torchvision.datasets import ImageFolder


class ImageNet(ImageFolder):
    """Extends ImageFolder dataset to include ground truth annotations."""

    def __init__(self, image_path, ground_truth_path, image_transform=None,
                 ground_truth_transform=None):
        """
        Extends the parent class with annotation information.

        Additional Args:
        image_path: the path to the ImageNet images. This folder must be
            formatted in ImageFolder style (i.e. label/imagename.jpeg)
        ground_truth_path: the path to the ImageNet annotations. This folder
            must be formated in ImageFolder style (i.e., label/imagename.xml).
        image_transform: a pytorch transform to apply to the images or None.
            Defaults to None.
        ground_truth_transform: a pytorch transform to apply to the ground
            truth annotations or None. Defaults to None.

        """
        super().__init__(image_path, transform=image_transform)
        self.ground_truth_transform = ground_truth_transform
        self.ground_truth_path = ground_truth_path

    def __getitem__(self, index):
        """Returns the image, ground_truth mask, and label of the image."""
        image, _ = super().__getitem__(index)
        image_path, _ = self.imgs[index]
        image_name = image_path.strip().split('/')[-1].split('.')[0]
        label = image_path.strip().split('/')[-2]

        ground_truth_file = os.path.join(self.ground_truth_path, label, '%s.xml' %image_name)
        ground_truth = self._create_ground_truth(ground_truth_file)
        if self.ground_truth_transform is not None:
            ground_truth = self.ground_truth_transform(ground_truth).squeeze(0)

        return image, ground_truth, int(label)

    def _create_ground_truth(self, ground_truth_file):
        """Creates a binary groudn truth mask based on the ImageNet annotations."""
        annotation = self._parse_xml(ground_truth_file)
        height, width = int(annotation['height']), int(annotation['width'])
        ground_truth = torch.zeros((height, width))
        for coordinate in annotation['coordinates']:
            y_min, y_max = int(coordinate['ymin']), int(coordinate['ymax'])
            x_min, x_max = int(coordinate['xmin']), int(coordinate['xmax'])
            ground_truth[y_min:y_max, x_min:x_max] = 1
        return ground_truth

    def _parse_xml(self, ground_truth_file):
        """Parse ImageNet annotation XML file."""
        if not os.path.isfile(ground_truth_file):
            raise IOError('No annotation data for %s.' %(ground_truth_file))
        tree = ET.parse(ground_truth_file)
        root = tree.getroot()
        bboxes = [obj.find('bndbox') for obj in root.findall('object')]
        coords = [{'xmin': int(bbox.find('xmin').text),
                   'ymin': int(bbox.find('ymin').text),
                   'xmax': int(bbox.find('xmax').text),
                   'ymax': int(bbox.find('ymax').text), } for bbox in bboxes]
        height = root.find('size').find('height').text
        width = root.find('size').find('width').text
        return {'coordinates': coords, 'height': height, 'width': width}
    