from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch


class CustomedDataset(Dataset):
    """
         Dataset reimplement to read label from text file
        Path to image folder
        Path to a tsv file containing filename of all images in that image folder
        Path to a tsv file containing labels with first column ID linked to images and second columns label
        len(labels)>= len(img_file_name)  OK
        len(labels)< len(img_file_name) NO!
        The img file list controls what will be included in the end
        this is easier to manage a data with one individual potentially having more than one image?
    """

    def __init__(self, img_folder_path, used_img_list, labels_file_path, transform, targetType="long"):
        # img file names
        self.label_path = labels_file_path
        self.img_list_path = used_img_list
        self.img_folder_path = img_folder_path
        # create a dictionary to store all labels
        self.all_labels = {}
        # self.id_list=None # is this desired as dict doesn't preserve order?
        self.ids = None
        self.image_lst = []
        self.transform = transform
        self.class_cnt = None
        self.targetType = targetType  # change based on which loss function in use
        with open(self.label_path, "r") as labelFile:
            for line in labelFile:
                line_vec = line.strip().split("\t")
                vec_len = len(line_vec)
                label_vec = np.zeros(vec_len - 1, dtype=float)
                for i in range(1, vec_len):
                    try:
                        np.put(label_vec, i - 1, float(line_vec[i]))
                    except:
                        print("labels not in float/integer, exit")
                        return
                # key:first col=ID (string) item:second+ col=label (string->float (for tensor transformation later)
                self.all_labels[line_vec[0]] = label_vec
        self.ids = list(self.all_labels.keys())
        # create a list to store the filenames
        with open(self.img_list_path, "r") as imgName:
            for line in imgName:
                id_img_pair = line.strip().split("\t")
                self.image_lst.append(id_img_pair[1])

        # a dict with class for every class: count , TODO how to handle multiple phenotype?
        uniques = set(tuple(arr) for arr in self.all_labels.values() if arr.size > 0)
        # print(uniques)
        # note that this ignore the multiple images and count from individuals
        self.class_cnt = {x: list(e[0] for e in self.all_labels.values()).count(x) for x in range(len(uniques))}

        # print(self.class_cnt)

    def __getitem__(self, index):
        """Process image and label
        img - label not 1:1 mapping but m:n  m>n one can have more than 1 images
        hence take image instead of label"""
        # id=self.image_lst[index]
        # print(index)
        # filename looks like this : 1018340.1.3.12.2.1107.5.2.18.41754.2015013116504949595372298.dcm.png
        img_name = self.image_lst[index]
        # print(img_name)
        img = Image.open(
            self.img_folder_path + '/' + img_name)  # PIL Image file ### why os.path.join() does not add '/' in this case?
        img = img.convert(
            'RGB')  # For safety, but maybe works the same without the step? this forces 1 channel image to 3 channels

        """Get the label using the img name"""
        id = img_name.split(".")[0]  # type(id) =string
        # print(id)
        target = self.all_labels[id]
        if self.targetType == "long":

            target = torch.squeeze(torch.as_tensor(target, dtype=torch.long))
        elif self.targetType == "double":
            target = torch.squeeze(torch.as_tensor(target, dtype=torch.double))
        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.image_lst)  # the total number of images in the index file
