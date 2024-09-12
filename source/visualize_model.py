import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import datetime


def imdisplay_save(inp, plotprefix=None, title=None, save=True):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    #     mean = np.array([0.485, 0.456, 0.406])
    #     std = np.array([0.229, 0.224, 0.225])
    #     inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(12, 3))
    plt.imshow(inp)
    plt.axis("off")

    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    if save:
        if plotprefix == None:
            st = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
            plotprefix = st
        plt.savefig('{}.png'.format(plotprefix), bbox_inches='tight')
    plt.close()    # close the figure window


def visualize_model(model, data_loader, device, plotname, class_names, num_images=8,regress=False,multi=False):
    model.eval()
    # fig = plt.figure()

    with torch.no_grad():
        try:

            for i, (inputs, labels, path) in enumerate(data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                #             outputs = model(inputs)
                #             outputs,aux = model(inputs)
                try:
                    outputs, aux = model(inputs)
                except:
                    outputs = model(inputs)

                if regress ==True:
                    outputs = torch.squeeze(outputs)
                else:
                    if multi ==True:
                        _, predicted = torch.max(outputs[:, 0:2], 1) # the first 2 output neurons are classification
                    else:
                        _, preds = torch.max(outputs, 1)

                # preds = outputs.float()

                inputs, classes = inputs[0:num_images], labels[0:num_images]
                inputs = inputs.cpu()
                out = torchvision.utils.make_grid(inputs)
                if multi ==True:
                    labels=labels.data[:, 17]
                    pairs =list( zip(predicted[0:num_images].data.cpu().numpy(), labels[0:num_images].data.cpu().numpy()))
                    print (pairs)
                else:
                    pairs = list(zip(preds[0:num_images].data.cpu().numpy(), labels[0:num_images].data.cpu().numpy()))
                if regress==True:
                    title = ['{}({})'.format(pair[0],pair[1]) for pair in pairs]
                else:
                    title = ['{}({})'.format(class_names[pair[0]], class_names[pair[1]]) for pair in pairs]

                imdisplay_save(out, title="Prediction(Truth):{}".format(title), plotprefix=plotname, save=True)
                plt.close()

                break
        except:
            for i, (inputs, labels) in enumerate(data_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                #             outputs = model(inputs)
                #             outputs,aux = model(inputs)
                try:
                    outputs, aux = model(inputs)
                except:
                    outputs = model(inputs)

                if regress ==True:
                    outputs = torch.squeeze(outputs)
                else:
                    if multi ==True:
                        _, predicted = torch.max(outputs[:, 0:2], 1) # the first 2 output neurons are classification
                    else:
                        _, preds = torch.max(outputs, 1)

                # preds = outputs.float()

                inputs, classes = inputs[0:num_images], labels[0:num_images]
                inputs = inputs.cpu()
                out = torchvision.utils.make_grid(inputs)
                if multi ==True:
                    labels=labels.data[:, 17]
                    pairs = list(zip(predicted[0:num_images].data.cpu().numpy(), labels[0:num_images].data.cpu().numpy()))
                else:
                    pairs =list( zip(preds[0:num_images].data.cpu().numpy(), labels[0:num_images].data.cpu().numpy()))

                if regress==True:
                    title = ['{}({})'.format(pair[0],pair[1]) for pair in pairs]
                else:
                    title = ['{}({})'.format(class_names[int(pair[0])], class_names[int(pair[1])]) for pair in pairs]
                print(title)
                imdisplay_save(out, title="Prediction(Truth):{}".format(title), plotprefix=plotname, save=True)
                plt.close()
                break
        return
