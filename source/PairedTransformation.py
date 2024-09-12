"""
Inherit the transformation classes and make it able to transform more than 1 image at one time
1. PairedRandomCrop from RandomCrop
2. PairedRandomHorizontalFlip from RandomHorizontalFlip

this is useful for data-label transformation
"""

from torchvision.transforms import RandomCrop, RandomHorizontalFlip
from PIL import Image
import random




class PairedRandomCrop(RandomCrop):
    def __init__(self, *args, **kwargs):
        super(PairedRandomCrop, self).__init__(*args, **kwargs)

    def __call__(self, images):
        """
        Args:
            images (PIL Image): a pair of Image (tuple: image + target) to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        i, j, h, w = self.get_params(images[0], self.size)

        return (images[0].crop((j, i, j + w, i + h)),
                images[1].crop((j, i, j + w, i + h)))


class PairedRandomHorizontalFlip(RandomHorizontalFlip):
    """Horizontally flip the given PIL Image randomly with a probability of 0.5."""

    def __init__(self, *args, **kwargs):
        super(RandomHorizontalFlip, self).__init__(*args, **kwargs)

    def __call__(self, images):
        """
        Args:
            images (PIL Image): a pair of Image (tuple: image + target) to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < 0.5:
            return (images[0].transpose(Image.FLIP_LEFT_RIGHT),
                    images[1].transpose(Image.FLIP_LEFT_RIGHT))
        return images