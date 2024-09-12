from pytorch.CustomDataset_labelFromFile import CustomedDataset,Image,torch
import numpy as np
from pytorch.transform_config_albumentation import *

class CustomDataset_PolarMap_reserve(CustomedDataset):
    def __init__(self, img_folder_path, used_img_list, labels_file_path, transform, targetType,
                 pytorch_transform=False):
        # img file names
        super().__init__(img_folder_path, used_img_list, labels_file_path, transform, targetType)
        # switch for which transform library
        self.pytorch_transform = pytorch_transform

        # use a dict to store images instead ID : [img_stress,img_rest,img_reserve]
        self.id_imgs_dict = {}
        # create a dict to store the filenames
        with open(self.img_list_path, "r") as imgName:
            for line in imgName:
                id_img_pair = line.strip().split("\t")
                # combine the img prefix and id to create a unique id
                # this is because one ID has 2 sets of polar maps :[
                # hardcoded knowing images from the same polar map set have same same prefix followed by "_"
                comp_key = id_img_pair[0] + "_" + id_img_pair[1].split("_")[0]
                if comp_key not in list(self.id_imgs_dict.keys()):  # new id-img pair

                    self.id_imgs_dict[comp_key] = [id_img_pair[1]]
                else:
                    self.id_imgs_dict[comp_key].append(id_img_pair[1])  # append to exisiting list
        # # custom check if 3 images are present:
        # id_img_issue = {k: v for k, v in self.id_imgs_dict.items() if len(v) != 3}
        # for k, v in id_img_issue.items():
        #     print("Check sample:{} with wrong number of images- {}".format(k, v))
        #     print("Only keep IDs with 3 images")
        # self.id_imgs_dict = {k: v for k, v in self.id_imgs_dict.items() if len(v) == 3}

        # a dict with class for every class: count
        # filter out labels without images
        # key:first col=ID (string) item:second+ col=label (string->float (for tensor transformation later)

        # id_w_imgs=[ i.split("_")[0] for i in self.id_imgs_dict.keys()]
        # update the key in self.all_labels from id to the composite key
        temp_dict = {}
        id_set = set()
        for k in self.id_imgs_dict.keys():
            id = k.split("_")[0]
            id_set.add(k)
            temp_dict[k] = self.all_labels[id]
        self.all_labels = temp_dict
        # replace id to id-img_prefix
        self.ids = sorted(list(id_set))
        # print(self.ids)
        # keep those with non missing label
        uniques = set(tuple(arr) for arr in self.all_labels.values() if arr.size > 0)
        # print(uniques)
        # this is used for adjusting weight in loss computation in the presence of class imbalance
        self.class_cnt = {x: list(e[0] for e in self.all_labels.values()).count(x) for x in range(len(uniques))}

    def __getitem__(self, index):
        """Process image and label
        img - label not 1:1 mapping but m:n  m>n one can have more than 1 images
        hence take image instead of label
        Since transform accepts  PIL Image or numpy.ndarray when transform to tensor.
        """
        # ANN0101.OT.0.1.2017.09.25.14.24.23.916.30773060_MBFstress.png/
        # ANN0101.OT.0.1.2017.09.25.14.24.23.916.30773060_MBFrest.png/
        # ANN0101.OT.0.1.2017.09.25.14.24.23.916.30773060_MBFreserve.png
        # override the __getitem__ method. this is the method dataloader calls
        cur_id = self.ids[index]
        img_paths = sorted(self.id_imgs_dict[cur_id], reverse=True)
        img_reserve = Image.open( self.img_folder_path + '/' + img_paths[0])
            # self.img_folder_path + '/' + img_paths[0]), Image.open(
            # self.img_folder_path + '/' + img_paths[1]), Image.open(
            # self.img_folder_path + '/' + img_paths[2])
        # print(img_name)
        # img_stress = img_stress.convert('RGB')

        # shape: height x width x channel

        target = self.all_labels[cur_id]
        if self.targetType == "long":
            # For crossEntropyLoss
            target = torch.squeeze(torch.as_tensor(target, dtype=torch.long))
        elif self.targetType == "double":
            # For MSE loss
            target = torch.squeeze(torch.as_tensor(target, dtype=torch.double))
        if self.transform:
            if self.pytorch_transform:
                img_reserve =  self.transform(img_reserve)  #tensor
            else:  # albumentation
                # albumentation take np array
                img_reserve =  np.array(img_reserve)
                # print("shape before transform",img_stress.shape,img_rest.shape,img_reserve.shape)
                # print("****************self.transform**********",self.transform)
                data = {"image":img_reserve,"stress": img_reserve, "rest": img_reserve, "reserve": img_reserve}

                augmented = self.transform(**data)
                img_reserve = augmented["reserve"]  #tensor
                # print("shape after transform",img_stress.shape,img_rest.shape,img_reserve.shape)
                # img_9ch = np.concatenate((img_stress, img_rest, img_reserve),axis=0)
                # print("img_9ch.shape",img_9ch.shape)

        else:
            img_reserve = torch.from_numpy(img_reserve) #tensor

            # img_9ch = np.concatenate((img_stress, img_rest, img_reserve), axis=-1)

        return img_reserve, target

    def __len__(self):  # each ID would have 3x so return the number of unique id-img pair instead
        id_imgs = list(self.id_imgs_dict.keys())
        return len(id_imgs)


class CustomDataset_PolarMap_reserve_multipleOut(CustomDataset_PolarMap_reserve):
    def __init__(self, img_folder_path, used_img_list, labels_file_path, transform,
                 pytorch_transform=False):
        # img file names
        super().__init__(img_folder_path, used_img_list, labels_file_path, transform, targetType="long")

        # for adjusting the weights in cross entropy loss , hence take the PCI at last index of the vector
        uniques = set(arr[-1] for arr in self.all_labels.values() if arr.size > 0)
        # print(uniques)
        # this is used for adjusting weight in loss computation in the presence of class imbalance
        self.class_cnt = {x: list(e[-1] for e in self.all_labels.values()).count(x) for x in range(len(uniques))}

        # print(self.class_cnt)

    def __getitem__(self, index):
        """Process image and label
        img - label not 1:1 mapping but m:n  m>n one can have more than 1 images
        hence take image instead of label
        Since transform accepts  PIL Image or numpy.ndarray when transform to tensor.
        """

        # ANN0101.OT.0.1.2017.09.25.14.24.23.916.30773060_MBFreserve.png
        # override the __getitem__ method. this is the method dataloader calls
        cur_id = self.ids[index]
        img_paths = sorted(self.id_imgs_dict[cur_id], reverse=True)
        img_reserve = Image.open( self.img_folder_path + '/' + img_paths[0])

        # print(img_name)
        # img_stress = img_stress.convert('RGB')

        # shape: height x width x channel

        target = self.all_labels[cur_id]
        target = torch.FloatTensor(target)
        # if self.targetType == "long":
            # For crossEntropyLoss
            # target = torch.squeeze(torch.as_tensor(target, dtype=torch.long))
        # elif self.targetType == "double":
        #     For MSE loss
            # target = torch.squeeze(torch.as_tensor(target, dtype=torch.double))
        if self.transform:
            if self.pytorch_transform:
                img_reserve =  self.transform(img_reserve)  #tensor
            else:  # albumentation
                # albumentation take np array
                img_reserve =  np.array(img_reserve)
                # print("shape before transform",img_stress.shape,img_rest.shape,img_reserve.shape)
                # print("****************self.transform**********",self.transform)
                data = {"image":img_reserve,"stress": img_reserve, "rest": img_reserve, "reserve": img_reserve}

                augmented = self.transform(**data)
                img_reserve = augmented["reserve"]  #tensor
                # print("shape after transform",img_stress.shape,img_rest.shape,img_reserve.shape)
                # img_9ch = np.concatenate((img_stress, img_rest, img_reserve),axis=0)
                # print("img_9ch.shape",img_9ch.shape)

        else:
            img_reserve = torch.from_numpy(img_reserve) #tensor

            # img_9ch = np.concatenate((img_stress, img_rest, img_reserve), axis=-1)

        return img_reserve, target

    def __len__(self):  # each ID would have 3x so return the number of unique id-img pair instead
        id_imgs = list(self.id_imgs_dict.keys())
        return len(id_imgs)












# if __name__ == "__main__":
#     from transform_config import *
#     data_dir="/home/mw/Analyses/polar_map/pngs/crops"
#     transformation_config="9ChPolarMap_228x228_resize_224"
#     label_file_path_prefix="/home/mw/Analyses/polar_map/MACE/AbstractESC2020deeplearning_idHarmonized.tsv_Events_MACE"
#     labels_file = "{}.{}".format(label_file_path_prefix, 'train')
#     cv_fname_prefix="/home/mw/Analyses/polar_map/MACE/all_id_polarMapCrop_paths.txt_Events_MACE.train.CV.1"
#     # take the the subsampled img_list
#     img_files = {x: "{}.{}".format(cv_fname_prefix, x) for x in ['train', 'val']}
#
#     image_datasets = {x: CustomDataset_stackedPolarMap(data_dir, img_files[x], labels_file,
#                                              data_transforms[transformation_config][x],targetType="long")
#                           for x in ['train', 'val']}
#     print(len(image_datasets['train']))
#     # print((image_datasets['val']))
