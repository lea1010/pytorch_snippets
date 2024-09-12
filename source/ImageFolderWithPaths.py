from torchvision import datasets


# class Image_folder_filename(datasets.ImageFolder):
#
#     def __getitem__(self, index):
#         """
#          Args:
#              index (int): Index
#
#          Returns:
#              tuple: (image, target) where target is class_index of the target class.
#          """
#         path, target = self.imgs[index]
#         img = self.loader(path)
#         if self.transform is not None:
#             img = self.transform(img)
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#
#         print(path)
#         return img, target, path
#
#     def __len__(self):
#         return len(self.imgs)


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
