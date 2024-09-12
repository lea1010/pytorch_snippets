# from torchvision import transforms
from albumentations import *
from albumentations.pytorch import ToTensorV2

data_transforms = {}
import cv2

data_transforms["polarMap_228x228_resizeCrop_224"] = {'train': Compose([
    RandomResizedCrop(224, 224, scale=(0.7, 1.0), ratio=(0.9, 1.1), interpolation=cv2.INTER_LINEAR, p=1),
    # do not change aspect ratio
    # CenterCrop((224, 224)),
    # VerticalFlip(0.5),
    # HorizontalFlip(0.5),
    ShiftScaleRotate(shift_limit=(-0.10, 0.10), scale_limit=0.1, rotate_limit=(-10, 10), p=0.8),
    # HueSaturationValue(hue_shift_limit=(-10, 10), sat_shift_limit=50, val_shift_limit=0, p=0.8),
    CoarseDropout(max_holes=40, max_height=30, max_width=30, min_height=16,min_width=16,min_holes=8,fill_value=0, always_apply=False, p=0.9),
    # CoarseDropout(max_holes=20, max_height=30, max_width=30, min_height=16,min_width=16,min_holes=8,fill_value=0, always_apply=False, p=0.9),
    # Normalize([0.485, 0.456, 0.406],
    #           [0.229, 0.224, 0.225]),
    # Normalize([0.5, 0.5, 0.5],
    #           [0.5, 0.5, 0.5]),
    Normalize([0, 0, 0],
              [1, 1, 1]),
    ToTensorV2()
], additional_targets={'stress': 'image', 'rest': 'image', 'reserve': 'image'}),
    # this normalization below is relevant to the pretrained models in PyTorch model zoo , trained on ImageNet
    # https://pytorch.org/docs/stable/torchvision/models.html
    #         ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    'val': Compose([Resize(224, 224),
                    # Normalize([0.485, 0.456, 0.406],
                    #           [0.229, 0.224, 0.225]),
                    # Normalize([0.5, 0.5, 0.5],
                    #           [0.5, 0.5, 0.5]),
                    Normalize([0, 0, 0],
                              [1, 1, 1]),
                    ToTensorV2()
                    ], additional_targets={'stress': 'image', 'rest': 'image', 'reserve': 'image'})
}

data_transforms["polarMap_228x228_crop_224"] = {'train': Compose([
    CenterCrop(224, 224, p=1.0),
    # do not change aspect ratio
    # CenterCrop((224, 224)),
    # VerticalFlip(),
    # HorizontalFlip(),
    ShiftScaleRotate(shift_limit=(-0.15, 0.15), scale_limit=0.1, rotate_limit=20, p=0.5),
    # HueSaturationValue(hue_shift_limit=(0, 0), sat_shift_limit=50, val_shift_limit=0, p=0.8),
    CoarseDropout(max_holes=28, max_height=8, max_width=8,min_width=4,min_height=4, fill_value=0, always_apply=False, p=0.9),
    # Normalize([0.485, 0.456, 0.406],
    #           [0.229, 0.224, 0.225]),
    Normalize([0.5, 0.5, 0.5],
              [0.5, 0.5, 0.5]),
    ToTensorV2()
], additional_targets={'stress': 'image', 'rest': 'image', 'reserve': 'image'}),
    # this normalization below is relevant to the pretrained models in PyTorch model zoo , trained on ImageNet
    # https://pytorch.org/docs/stable/torchvision/models.html
    #         ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    'val': Compose([CenterCrop(224, 224, p=1.0),
                    # Normalize([0.485, 0.456, 0.406],
                    #           [0.229, 0.224, 0.225]),
                    Normalize([0.5, 0.5, 0.5],
                              [0.5, 0.5, 0.5]),
                    ToTensorV2()
                    ], additional_targets={'stress': 'image', 'rest': 'image', 'reserve': 'image'})
}

# if __name__ == "__main__":
# import numpy as np
#
# image = np.ones((228, 228, 3), dtype=np.uint8)
# mask = np.ones((228, 228), dtype=np.uint8)
# whatever_data = "my name"
# augmentation = data_transforms["polarMap_228x228_resizeCrop_224"]['val']
# print(augmentation)
# data = {"image": image, "mask": mask, "whatever_data": whatever_data, "additional": "hello"}
# augmented = augmentation(**data)
# image, mask, whatever_data, additional = augmented["image"], augmented["mask"], augmented["whatever_data"], \
#                                          augmented["additional"]
#
# print(image, image.shape)
# print(mask, mask.shape)
# print(whatever_data, len(whatever_data))
# print(additional, len(additional))
