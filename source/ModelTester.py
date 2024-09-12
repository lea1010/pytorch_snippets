import torch
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import argparse
import matplotlib.pyplot as plt
import datetime
from pytorch.ImageFolderWithPaths import ImageFolderWithPaths
from pytorch.visualize_model import imdisplay_save, visualize_model
# from bootstrap import bootstrap
from pytorch.set_random_seed import set_random_seed
# import copy
# import sys
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import os
from scipy.stats.stats import pearsonr


def test_classifier_model(dataset_name, model, device, test_loader):
    model.eval()
    total = 0
    correct = 0
    y_true = []
    y_score = []
    filepaths = []
    with torch.no_grad():  # replaced volatile=True in old version
        for data, labels in test_loader:
            # for data, labels, paths in test_loader:
            data, labels = data.to(device), labels.to(device)  # replaced .cuda() in old version
            try:
                outputs, aux = model(data)
            except:
                outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().data)
            y_score.extend(predicted.cpu().data)
            # filepaths.extend(paths)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    print('\n Accuracy of the model on {}: {}/{} ({:.1f}%)\n'.format(
        dataset_name, correct, total,
        100. * correct / total))
    # return y_true, y_score, filepaths
    return y_true, y_score


def test_classifier_model_multiOut(dataset_name, model, device, test_loader):
    model.eval()
    total = 0
    correct = 0
    y_true = []
    y_score = []
    filepaths = []
    with torch.no_grad():  # replaced volatile=True in old version
        for data, labels in test_loader:
            # for data, labels, paths in test_loader:
            data, labels = data.to(device), labels.to(device)  # replaced .cuda() in old version
            try:
                outputs, aux = model(data)
            except:
                outputs = model(data)
            # _, predicted = torch.max(outputs, 1)
            _, predicted = torch.max(outputs[:, 0:2], 1)

            total += labels.size(0)
            correct += (predicted ==  labels.data[:, 17]).sum().item()

            y_true.extend(labels.data[:, 17].cpu())
            y_score.extend(predicted.cpu().data)
            # filepaths.extend(paths)

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    print('\n Accuracy of the model on {}: {}/{} ({:.1f}%)\n'.format(
        dataset_name, correct, total,
        100. * correct / total))
    # return y_true, y_score, filepaths
    return y_true, y_score





def test_regression_model(dataset_name, model, device, test_loader):
    model.eval()
    # total = 0
    # correct = 0
    y_true = []
    y_score = []
    # filepaths = []
    with torch.no_grad():  # replaced volatile=True in old version
        for data, labels in test_loader:
            # for data, labels, paths in test_loader:
            data, labels = data.to(device), labels.to(device)  # replaced .cuda() in old version

            try:
                outputs, aux = model(data)
            except:
                outputs = model(data)
            outputs = torch.squeeze(outputs)
            # outputs = outputs.double()

            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().data)
            y_score.extend(outputs.cpu().data)
            # filepaths.extend(paths)

    corr = pearsonr(y_true, y_score)
    print('\n Pearson correlation between prediction and true labels on {}: {:.1f} (p={})\n'.format(
        dataset_name, corr[0],corr[1]))

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # return y_true, y_score, filepaths
    return y_true, y_score


# def test_classifier_inceptionv3(dataset_name, model, device, test_loader):
#     model.eval()
#     total = 0
#     correct = 0
#     y_true = []
#     y_score = []
#     filepaths = []
#     with torch.no_grad():  # replaced volatile=True in old version
#         for data, labels in test_loader:
#             # for data, labels, paths in test_loader:
#             data, labels = data.to(device), labels.to(device)  # replaced .cuda() in old version
#             try:
#                 outputs, aux = model(data)
#             except:
#                 outputs = model(data)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#             y_true.extend(labels.cpu().data)
#             y_score.extend(predicted.cpu().data)
#             # filepaths.extend(paths)
#
#     y_true = np.asarray(y_true)
#     y_score = np.asarray(y_score)
#
#     print('\n Accuracy of the inception_v3 on {}: {}/{} ({:.1f}%)\n'.format(
#         dataset_name, correct, total,
#         100. * correct / total))
#     # return y_true, y_score, filepaths
#     return y_true, y_score
#
#
# def test_regression_inceptionv3(dataset_name, model, device, test_loader):
#     model.eval()
#     # total = 0
#     # correct = 0
#     y_true = []
#     y_score = []
#     # filepaths = []
#     with torch.no_grad():  # replaced volatile=True in old version
#         for data, labels in test_loader:
#             # for data, labels, paths in test_loader:
#             data, labels = data.to(device), labels.to(device)  # replaced .cuda() in old version
#
#             try:
#                 outputs, aux = model(data)
#             except:
#                 outputs = model(data)
#             outputs = torch.squeeze(outputs)
#             # outputs = outputs.double()
#
#             # total += labels.size(0)
#             # correct += (predicted == labels).sum().item()
#
#             y_true.extend(labels.cpu().data)
#             y_score.extend(outputs.cpu().data)
#             # filepaths.extend(paths)
#
#     corr = pearsonr(y_true, y_score)
#     print('\n Pearson correlation between prediction and true labels on {}: {:.1f} (p={})\n'.format(
#         dataset_name, corr[0],corr[1]))
#
#     y_true = np.asarray(y_true)
#     y_score = np.asarray(y_score)
#
#     # return y_true, y_score, filepaths
#     return y_true, y_score

#
# if __name__ == "__main__":
#     # construct the argument parse and parse the arguments
#     ap = argparse.ArgumentParser()
#     ap.add_argument("-m", "--model", required=True,
#                     help="Path to model file")
#     ap.add_argument("-i", "--img", required=True,
#                     help="folder to test set images")
#     # ap.add_argument("-p", "--padding", required=True, default=False,
#     #                 help="boolean value whether to pad (True) or crop (False) the image to square")
#     ap.add_argument("-b", "--bts", required=False, default=80,
#                     help="batch size")
#     ap.add_argument("-n", "--name", required=False, default="norm",
#                     help="name of the dataset")
#     # ap.add_argument("-c", "--crop", required=False, default=1536,
#     #                 help="size after center crop")
#     # ap.add_argument("-ps", "--padSize", required=False, default=192,
#     #                 help="padding size")
#     ap.add_argument("-tr", "--transformation", required=False, default="c",
#                     help="crop or pad the images to sqaure")
#     ap.add_argument("-dim", "--transformation_dim", required=False, default=1536,
#                     help="the size to crop or pad the images to sqaure")
#     ap.add_argument("-s", "--seed", required=False, default=155,
#                     help="random seed")
#
#     args = vars(ap.parse_args())
#
#     start_time = time.time()
#     # pad=args['padding']
#     bat_size = int(args['bts'])
#     name = args['name']
#     # crop_size=int(args['crop'])
#     # pad_size=int(args['padSize'])
#     tr = args['transformation']
#
#     tr_d = list(args['transformation_dim'].split(','))
#     for i, pad in enumerate(tr_d):
#         tr_d[i] = int(pad)
#     tr_d_tuple = tuple(tr_d)
#
#     try:
#         model_path = args['model']
#     except:
#
#         print("model not found")
#
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     try:
#         test_data_dir = args['img']
#     except:
#         test_data_dir = 'data/test'
#
#     custom_seed = int(args['seed'])
#
#
#     # process test set
#
#     if tr == "p":
#         data_transforms = {
#             'train': transforms.Compose([
#                 transforms.Pad(tr_d_tuple),
#                 transforms.RandomResizedCrop(299),
#                 transforms.RandomVerticalFlip(),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomAffine(90, translate=(0.1, 0.1)),
#                 transforms.ToTensor()
#                 #         ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ]),
#             'val': transforms.Compose([transforms.Pad(tr_d_tuple), transforms.Resize((299, 299)),
#                                        transforms.ToTensor()
#                                        #         ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                                        ])}
#     else:
#         data_transforms = {
#             'train': transforms.Compose([
#                 transforms.RandomResizedCrop(416),
#                 transforms.CenterCrop(tr_d_tuple),
#                 # transforms.RandomResizedCrop(299),
#                 transforms.RandomVerticalFlip(),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomAffine(90, translate=(0.1, 0.1)),
#                 transforms.ToTensor()
#                 #         ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#             ]),
#             'val': transforms.Compose([transforms.CenterCrop(tr_d_tuple), transforms.Resize((299, 299)),
#                                        transforms.ToTensor()
#                                        #         ,transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#                                        ])
#         }
#
#     # test_dataset = datasets.ImageFolder(test_data_dir, data_transforms['val'])
#     # use the reimplemented class
#     test_dataset = ImageFolderWithPaths(test_data_dir, data_transforms['val'])
#
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bat_size,
#                                               shuffle=True, num_workers=4)
#
#     model_ft = models.inception_v3(pretrained=False)
#     # class_names = ['female', 'male'] # 0,1
#     class_names = test_dataset.classes
#
#     num_ftrs = model_ft.fc.in_features
#     # model_ft.fc = nn.Linear(num_ftrs, 2)
#     model_ft.fc = nn.Linear(num_ftrs, len(class_names))
#     # model_ft.load_state_dict(torch.load(model_path))
#     model_ft = torch.load(model_path)
#     model_ft = model_ft.to(device)
#     print('load model from {}'.format(model_path))
#
#     # try:
#     print('run script...')
#     y_true, y_score, filepaths = test_model('test set performance', model_ft, device, test_loader)
#     print('bootstrap...')
#     bootstrap(y_true, y_score, filepath=name, randomseed=custom_seed)
#     with open('{}_true_label.txt'.format(name), 'w+') as tt:
#         for label in y_true:
#             print(label, file=tt)
#     with open('{}_predicted_label.txt'.format(name), 'w+') as pp:
#         for prediction in y_score:
#             print(prediction, file=pp)
#     with open('{}_file_paths.txt'.format(name), 'w+') as fp:
#         for path in filepaths:
#             print(path, file=fp)
#
#     # except:
#
#     # print('debug test() or bootstrap()')
#     # pass
#     visualize_model(model_ft, device=device, data_loader=test_loader, plotname=name,
#                     class_names=class_names, num_images=8)
#     plt.close()
#     run_time = time.time() - start_time
#
#     print("Script run time: {:.0f}m {:.0f}s\n".format(run_time // 60, run_time % 60))
