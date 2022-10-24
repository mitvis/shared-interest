# ImageNet Download Util
This scripts in this folder reorganize ImageNet images and annotations into the Pytorch Dataset ImageFolder file structure.

Shared Interest is built on Pytorch. Pytorch has a class of functions called Datasets that handle loading data from disk. To load images and their bounding boxes, Shared Interest extends the Pytorch [ImageFolder Dataset](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html). ImageFolder requires images to be in a specific file structure. In order to use the data, the scripts in this folder reorganize the images and annotations, so the file structure is compatable with Pytorch ImageFolder.

**For ImageNet downloaded from the [ImageNet website](https://www.image-net.org/index.php)** use [`imagefolder_structure_from_imagenet.ipynb`](https://github.com/mitvis/shared-interest/blob/main/imagenet_download_util/imagefolder_structure_from_imagenet.ipynb).

**For ImageNet downloaded from [Kaggle](https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description)** use [`imagefolder_structure_from_kaggle.ipynb`](https://github.com/mitvis/shared-interest/blob/main/imagenet_download_util/imagefolder_structure_from_kaggle.ipynb).

After running one of these scripts, your `imagenet` directory will be formatted as follows:
```
imagenet
|
|---val/
|   |---images/
|       |---0000/
|           |---ILSVRC2012_val_<imageid0>.jpeg
|           |---ILSVRC2012_val_<imageid1>.jpeg
|           |...
|       |...
|       |---0999/
|   |---annotations/
|       |---0000/
|           |---ILSVRC2012_val_<imageid0>.xml
|           |---ILSVRC2012_val_<imageid1>.xml
|           |...
|       |...
|       |---0999/

```
