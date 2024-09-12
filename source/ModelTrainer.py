# from test_inception3_sex_classifier_outlabel import test, ImageFolderWithPaths, bootstrap
# import pickle
# from torch.optim import lr_scheduler
import datetime

import copy
import time

import matplotlib.pyplot as plt
# from torch.optim import lr_scheduler
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from pytorch.assessment_metrics import *

def simple_plot_metric(plot_name,num_epochs,y1,y2=None,clr_y1='b',clr_y2='r',title="By epoch"):
        plt.figure()
        plt.plot(np.arange(1, num_epochs + 1), y1, clr_y1)
        if y2 is not None:
            plt.plot(np.arange(1, num_epochs + 1), y2, clr_y2)  # visualize loss change
        plt.title(title)
        loss_plot_name = plot_name
        plt.savefig(loss_plot_name, bbox_inches='tight')


class ModelTrainer():
    def __init__(self,data_iterator, model,criterion,optimizer,out_prefix,device,batch_size,metrics,learning_scheduler=None):
        self.scheduler = learning_scheduler
        self.outpath=out_prefix
        self.dataloader=data_iterator
        self.bat_size=batch_size
        self.device=device
        self.model=model
        self.metrics_stat = None
        self.metrics_func= None
        self.criterion = criterion
        self.optimizer=optimizer
        self.learning_scheduler=learning_scheduler
        # self.best_model =  copy.deepcopy(model.state_dict())
        for key, value in metrics.items():
            self.metrics_stat[key]=np.array([])
            # https://stackoverflow.com/questions/28415595/how-to-store-function-in-class-attribute
            self.metrics_func[key]= staticmethod(value)
        # always have loss in stat
        self.metrics_stat["loss"] =np.array([])
    def forward(self, inputs, is_train, topk=True):
        # zero the parameter gradients
        # forward
        with torch.set_grad_enabled(is_train):
            if topk:
                outputs=self.model(inputs)
                _, preds = torch.max(outputs, 1)
                return outputs,preds
            else:
                return self.model(inputs),0

    def backward(self,outputs,labels):
        loss = self.criterion(outputs, labels)
        # backward
        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
        return loss
    def compute_metric(self,y_tr,y_sc):
        # loop over the selected metric functions
        for m_name, m_func in self.metrics_func.items():
            self.metrics_stat[m_name]=np.append(self.metrics_stat[m_name],m_func(y_tr,y_sc))


    def runEpochs(self,num_epochs,topk=True):
        since = time.time()

        print('------------Time: {} ---------------'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')))

        for epoch in range(num_epochs):
            y_tru = np.array([])
            y_scr = np.array([])

            print('Epoch {}/{} \n {}'.format(epoch, num_epochs - 1,'-' * 30))
            self.model.train()  # Set model to training mode
            running_loss = 0.0
            num_iter = 0
            # Iterate over data.
            for inputs, labels in self.dataloader:
                num_iter += 1
                inputs,labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()

                outputs,preds=self.forward(inputs, is_train=True,topk=topk)
                loss=self.backward(outputs,labels)


                y_tru = np.append(y_tru, labels.data.cpu().numpy())
                if topk:  #classfier taking top out of k classes as the prediction
                    y_scr = np.append(y_scr, preds.data.cpu().numpy())
                else:
                    y_scr = np.append(y_scr, outputs.data.cpu().numpy())

                # note loss.item() to get the python number instead of loss.data[0] which now gives a tensor!
                running_loss += loss.item() * inputs.size(0)
            epoch_loss = running_loss / (num_iter * self.bat_size)
            time_elapsed = time.time() - since
            print("Time elapsed: {:.0f}m {:.0f}s {} Loss: {:.4f} \n".format(time_elapsed // 60, time_elapsed % 60,"Training", epoch_loss))
            # attach loss to metric

            self.metrics_stat["loss"]=np.append(self.metrics_stat["loss"],epoch_loss)
            # compute the metrics with the y_tru and y_scr
            self.compute_metric(y_tru,y_scr)


        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def save_model(self):
        model_save_path=self.outpath+".pth"
        torch.save(self.model, model_save_path)
        print("Model saved to {}.".format(model_save_path))

    def get_model(self):
        return self.model()

    def get_metric_stats(self):
        return self.metrics_stat


class ModelTrainerWithValidation(ModelTrainer):
    def __init__(self,data_iterators, model,criterion,optimizer,out_prefix,device,batch_size,metrics,learning_scheduler=None):
        super().__init__(self,data_iterators, model,criterion,optimizer,out_prefix,device,batch_size,metrics,learning_scheduler)
        self.best_model_wts = copy.deepcopy(model.state_dict())
        # stats for both training and validation
        for key, value in metrics.items():
            self.metrics_stat[key]={x: np.array([]) for x in ['train', 'val']}
        self.metrics_stat["loss"] ={x: np.array([]) for x in ['train', 'val']}
    # update the compute metric for the change

    def compute_metric(self,y_tr,y_sc,phase):
        # loop over the selected metric functions
        for m_name, m_func in self.metrics_func.items():
            self.metrics_stat[m_name][phase]=np.append(self.metrics_stat[m_name][phase],m_func(y_tr,y_sc))

    def runEpochs(self,num_epochs,topk=True):
        since = time.time()

        print('------------Time: {} ---------------'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')))

        for epoch in range(num_epochs):
            y_tru_dict = {x: np.array([]) for x in ['train', 'val']}
            y_scr_dict = {x: np.array([]) for x in ['train', 'val']}


            print('Epoch {}/{} \n {}'.format(epoch, num_epochs - 1, '-' * 30))
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode important to network with batchnorm layer            running_loss = 0.0
                for inputs, labels in self.dataloader[phase]:
                    running_loss=0.0
                    num_iter = 0

                    # Iterate over data.
                    for inputs, labels in self.dataloader:
                        num_iter += 1
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        self.optimizer.zero_grad()
                        if phase == "train":
                            outputs, preds = self.forward(inputs, is_train=True,topk=topk)
                            loss = self.backward(outputs, labels)
                        else:
                            outputs,preds =self.forward (inputs, is_train=False,topk=topk)
                        # attach the true labels for metric computation later
                        y_tru_dict[phase] = np.append(y_tru_dict[phase], labels.data.cpu().numpy())
                        # attach the prediction
                        if topk:  # classfier taking top out of k classes as the prediction
                            y_scr_dict[phase] = np.append(y_scr_dict[phase], preds.data.cpu().numpy())
                        else:
                            y_scr_dict[phase] = np.append(y_scr_dict[phase], outputs.data.cpu().numpy())

                        # note loss.item() to get the python number instead of loss.data[0] which now gives a tensor!
                        running_loss += loss.item() * inputs.size(0)
                    epoch_loss = running_loss / (num_iter * self.bat_size)
                    time_elapsed = time.time() - since
                    print(
                        "Time elapsed: {:.0f}m {:.0f}s {} Loss: {:.4f} \n".format(time_elapsed // 60, time_elapsed % 60, "Training",
                                                                                  epoch_loss))
                    # attach loss to metric
                    self.metrics_stat["loss"] = np.append(self.metrics_stat["loss"], epoch_loss)
                    # compute the metrics with the y_tru and y_scr
                    self.compute_metric(y_tru, y_scr)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))





def train_classifier(dataloaders, model, criterion, optimizer, model_save_path, device, bat_size,
                     num_epochs=25, lrSch=False,
                     scheduler=None, save_model=True, stat_out=False, plot=True):
    """training along with validation """

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    acc_vec_train = []  # average training loss of each epoch
    acc_vec_val = []
    auc_vec_val=[]

    loss_vec_train = []  # average training loss of each epoch
    loss_vec_val = []  # average test loss of each epoch
    iter_loss_vec_train, iter_acc_vec_train = [], []
    iter_loss_vec_val, iter_acc_vec_val = [], []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 30)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode important to network with batchnorm layer
                # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/2
            y_tru = np.array([])
            y_scr = np.array([])
            running_loss = 0.0
            running_corrects = 0
            num_iter = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                num_iter += 1
                inputs = inputs.to(device)
                labels = labels.to(device)
                if phase == 'val':   #for AUC calculation
                    y_tru=np.append(y_tru,labels.data.cpu().numpy())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'val':
                        y_scr=np.append(y_scr,preds.data.cpu().numpy())


                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if lrSch is True and scheduler is not None:
                            scheduler.step()

                # statistics
                if num_iter % 10 == 0:
                    print("Cross Entropy Loss: {:.4f} * 1e-03  [{}/{} ({:.0f}%)]".format
                          (loss.item() * 1000, num_iter * bat_size, len(dataloaders[phase]) * bat_size,
                           num_iter / len(dataloaders[phase]) * 100))

                # note loss.item() to get the python number instead of loss.data[0] which now gives a tensor!
                running_loss += loss.item() * inputs.size(0)
                #                 print(running_loss)
                running_corrects += torch.sum(preds == labels.data)
                #                 print("torch sum ",torch.sum(preds == labels.data))
                if phase == 'train':
                    iter_loss_vec_train.append(loss.item())
                    iter_acc_vec_train.append(torch.sum(preds == labels.data) / inputs.size(0))

                if phase == 'val':
                    iter_loss_vec_val.append(loss.item())
                    iter_acc_vec_val.append(torch.sum(preds == labels.data) / inputs.size(0))

            epoch_loss = running_loss / (num_iter * bat_size)
            epoch_acc = running_corrects.double() / (num_iter * bat_size)

            if phase =='val':
                epoch_auc=roc_auc_score(y_tru,y_scr)
                auc_vec_val.append(epoch_auc)
            else:
                epoch_auc=-99
            print('{} Loss: {:.4f} Acc: {:.4f} AUC:{:.3f}'.format(
                phase, epoch_loss, epoch_acc,epoch_auc))
            # print duration for training one epoch
            time_elapsed = time.time() - since
            print("Time elapsed: {:.0f}m {:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60))

            if phase == 'train':
                loss_vec_train.append(epoch_loss)
                acc_vec_train.append(epoch_acc)

            if phase == 'val':
                loss_vec_val.append(epoch_loss)
                acc_vec_val.append(epoch_acc)

            # deep copy the model if the new model performs better by some margin
            if phase == 'val' and epoch >= 2 and epoch_acc - 0.005 >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # torch.save(model.state_dict(), model_save_path)
                print('Epoch: {} current best: {} model copied '.format(epoch, best_acc))

            if phase == 'val' and len(acc_vec_val) > 5:
                past_avg_val_acc = float(sum(acc_vec_val[-5:]) / len(acc_vec_val[-5:]))
                if (epoch_acc - past_avg_val_acc) < 0.005:
                    print("Epoch {}: val acc has not improved by 0.5% compared with average of last 5 epochs".format(
                        epoch))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    if save_model:
        torch.save(model, model_save_path)
        print("model saved to {}".format(model_save_path))

    try:
        plot_prefix = model_save_path[:-4]
        stat_file_name = plot_prefix + '_stat.txt'
        out_stat = np.array((loss_vec_train, loss_vec_val, acc_vec_train, acc_vec_val,auc_vec_val))
        np.savetxt(stat_file_name, out_stat, delimiter='\t')
    except:
        print("Stats not saved")
        pass
    if plot:
        try:
            plot_prefix = model_save_path[:-4]

            loss_plot_name = plot_prefix + '_loss.png'
            simple_plot_metric(loss_plot_name,num_epochs,y1=loss_vec_train,y2=loss_vec_val,title='Average loss of each epoch')

            acc_plot_name = plot_prefix + '_acc.png'
            simple_plot_metric(acc_plot_name,num_epochs,y1=acc_vec_train,y2=acc_vec_val,title='Average accuracy of each epoch')

            auc_plot_name = plot_prefix + '_valAuc.png'
            simple_plot_metric(auc_plot_name,num_epochs,y1=auc_vec_val,title='AUC of each epoch in validation')
        except:
            print("Figures not generated")
            pass
    if stat_out:
        out_stat_it = list((iter_loss_vec_train, iter_loss_vec_val, iter_acc_vec_train, iter_acc_vec_val))

        return model, out_stat.tolist(), out_stat_it
    else:
        return model


def train_classifier_no_val(dataloader, model, criterion, optimizer, model_save_path, device, bat_size,
                            num_epochs=25, lrSch=False,
                            scheduler=None, save_model=True, stat_out=False, plot=True):
    """training without validation """
    is_train = True
    since = time.time()
    acc_vec_train = []  # average training loss of each epoch
    loss_vec_train = []  # average training loss of each epoch
    print("training:{} {},{}".format(criterion, optimizer, device))
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 30)
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        num_iter = 0
        # Iterate over data.
        for inputs, labels in dataloader:
            num_iter += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            # backward
            loss.backward()
            optimizer.step()
            if lrSch is True and scheduler is not None:
                scheduler.step()

            # statistics
            if num_iter % 10 == 0:
                print("Cross Entropy Loss: {:.4f} * 1e-03  [{}/{} ({:.0f}%)]".format
                      (loss.item() * 1000, num_iter * bat_size, len(dataloader) * bat_size,
                       num_iter / len(dataloader) * 100))

            # note loss.item() to get the python number instead of loss.data[0] which now gives a tensor!
            running_loss += loss.item() * inputs.size(0)
            #                 print(running_loss)
            running_corrects += torch.sum(preds == labels.data)
        #                 print("torch sum ",torch.sum(preds == labels.data))

        epoch_loss = running_loss / (num_iter * bat_size)
        epoch_acc = running_corrects.double() / (num_iter * bat_size)

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            "Training", epoch_loss, epoch_acc))
        # print duration for training one epoch
        time_elapsed = time.time() - since
        print("Time elapsed: {:.0f}m {:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60))
        loss_vec_train.append(epoch_loss)
        acc_vec_train.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    if save_model:
        torch.save(model, model_save_path)
        print("model saved to {}".format(model_save_path))

    try:
        plot_prefix = model_save_path[:-4]
        stat_file_name = plot_prefix + '_stat.txt'
        out_stat = np.array((loss_vec_train, acc_vec_train))
        np.savetxt(stat_file_name, out_stat, delimiter='\t')
    except:
        print("Stats not saved")
        pass
    if plot:
        try:
            plot_prefix = model_save_path[:-4]


            loss_plot_name = plot_prefix + '_loss.png'
            simple_plot_metric(loss_plot_name, num_epochs, y1=loss_vec_train,
                               title='Average loss of each epoch')

            acc_plot_name = plot_prefix + '_acc.png'
            simple_plot_metric(acc_plot_name, num_epochs, y1=acc_vec_train,
                               title='Average accuracy of each epoch')

        except:
            print("Figures not generated")
            pass
    if stat_out:
        return model, out_stat.tolist()
    else:
        return model


def train_no_val_multiLoss(dataloader, model, criterions, optimizer, model_save_path, device, bat_size,
                           num_epochs=25, lrSch=False,
                           scheduler=None, save_model=True, stat_out=False, plot=True):
    """training without validation """
    is_train = True
    since = time.time()
    auc_vec_train=[]

    acc_vec_train = []  # average training loss of each epoch
    loss_vec_train = []  # average training loss of each epoch
    classification_loss = criterions[0]
    regression_loss = criterions[1]
    print("training:{} {},{}".format(criterions, optimizer, device))
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 30)
        model.train()  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0
        num_iter = 0
        y_tru = np.array([])
        y_scr = np.array([])
        # Iterate over data.
        for inputs, labels in dataloader:
            num_iter += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            outputs = outputs.double()
            # print("model output tensor type", outputs.type())
            _, preds = torch.max(outputs[:, 0:2], 1)
            # Cross entropy loss for PCI
            # cls_loss = classification_loss(outputs[:, 0:2], labels[:, 3].long())
            cls_loss = classification_loss(outputs[:, 0:2], labels[:, 17].long())

            # MSE loss for the numerical readouts
            # 3 regions
            # rgn_loss_lad=regression_loss(outputs[:,2],labels[:,0].double())
            # rgn_loss_lcx=regression_loss(outputs[:,3],labels[:,1].double())
            # rgn_loss_rca=regression_loss(outputs[:,4],labels[:,2].double())
            # region_loss=rgn_loss_lad/3 +rgn_loss_lcx/3 + rgn_loss_rca/3
            # 16 segments
            segment_loss=torch.zeros(1)
            # for i in range(17):
            #     print(labels[0, :])
            #     print(outputs[:, i + 2])
            # segment_loss+=regression_loss(outputs[:,i+2],labels[:,i].double())
            segment_loss = (regression_loss(outputs[:, 2:], labels[:, :17].double()))  #sum of 17 segments
            # loss = criterion(outputs, labels)
            p = 0.8
            q = 1.0 - p
            # loss=((p*cls_loss.double()) + q*(rgn_loss_lad/3 +rgn_loss_lcx/3 + rgn_loss_rca/3)).double()
            loss = ((p * cls_loss.double()) + q * (segment_loss)).double()

            # backward
            loss.backward()
            optimizer.step()

            # for AUC calculation
            y_tru = np.append(y_tru, labels[:, 17].data.cpu().numpy())
            y_scr = np.append(y_scr, preds.data.cpu().numpy())

            if lrSch is True and scheduler is not None:
                scheduler.step()

            # statistics
            if num_iter % 10 == 0:
                print("Classification loss (weight:{:.2f})/ Regression loss ({:.2f}): {:.3f} / {:.3f}".format(p, q,
                                                                                                      cls_loss.item(),
                                                                                                      segment_loss.item()))
                print("Total loss: {:.4f} * 1e-03  [{}/{} ({:.0f}%)]".format
                      (loss.item() * 1000, num_iter * bat_size, len(dataloader) * bat_size,
                       num_iter / len(dataloader) * 100))

            # note loss.item() to get the python number instead of loss.data[0] which now gives a tensor!
            running_loss += loss.item() * inputs.size(0)
            #                 print(running_loss)
            # ***************************************************************************************************
            # running_corrects += torch.sum(preds == labels.data)
            running_corrects += torch.sum(preds == labels.data[:, 17])
            # ****************************************************
        #                 print("torch sum ",torch.sum(preds == labels.data))

        epoch_loss = running_loss / (num_iter * bat_size)
        epoch_acc = running_corrects.double() / (num_iter * bat_size)

        epoch_auc = roc_auc_score(y_tru, y_scr)
        auc_vec_train.append(epoch_auc)


        print('{} Loss: {:.4f} Acc: {:.4f} AUC:{:.3f}'.format(
            "Training", epoch_loss, epoch_acc,epoch_auc))
        # print duration for training one epoch
        time_elapsed = time.time() - since
        print("Time elapsed: {:.0f}m {:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60))
        loss_vec_train.append(epoch_loss)
        acc_vec_train.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    if save_model:
        torch.save(model, model_save_path)
        print("model saved to {}".format(model_save_path))

    try:
        plot_prefix = model_save_path[:-4]
        stat_file_name = plot_prefix + '_stat.txt'
        out_stat = np.array((loss_vec_train, acc_vec_train,auc_vec_train))
        np.savetxt(stat_file_name, out_stat, delimiter='\t')
    except:
        print("Stats not saved")
        pass
    if plot:
        try:
            plot_prefix = model_save_path[:-4]
            plt.figure()
            plt.plot(np.arange(1, num_epochs + 1), loss_vec_train, 'b')
            # plt.plot(np.arange(1, num_epochs + 1), loss_vec_val, 'r')  # visualize loss change
            plt.title('Average loss of each epoch')
            loss_plot_name = plot_prefix + '_loss.png'
            plt.savefig(loss_plot_name, bbox_inches='tight')
            plt.figure()
            plt.plot(np.arange(1, num_epochs + 1), acc_vec_train, 'b')
            # plt.plot(np.arange(1, num_epochs + 1), acc_vec_val, 'r')
            plt.title('Average accuracy of each epoch')
            acc_plot_name = plot_prefix + '_acc.png'
            plt.savefig(acc_plot_name, bbox_inches='tight')
            plt.plot(np.arange(1, num_epochs + 1), auc_vec_train, 'crimson')
            plt.title('AUC of each epoch in training set')
            acc_plot_name = plot_prefix + '_trainAuc.png'
            plt.savefig(acc_plot_name, bbox_inches='tight')
        except:
            print("Figures not generated")
            pass
    if stat_out:
        return model, out_stat.tolist()
    else:
        return model




def train_multiLoss(dataloaders, model, criterions, optimizer, model_save_path, device, bat_size,
                    num_epochs=25, lrSch=False,
                    scheduler=None, save_model=True, stat_out=False, plot=True):
    """training along with validation """

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    acc_vec_train = []  # average training loss of each epoch
    acc_vec_val = []
    loss_vec_train = []  # average training loss of each epoch
    loss_vec_val = []  # average test loss of each epoch
    auc_vec_val=[]
    iter_loss_vec_train, iter_acc_vec_train = [], []
    iter_loss_vec_val, iter_acc_vec_val = [], []
    classification_loss = criterions[0]
    regression_loss = criterions[1]
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 30)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode important to network with batchnorm layer
                # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/2
            y_tru = np.array([])
            y_scr = np.array([])
            running_loss = 0.0
            running_corrects = 0
            num_iter = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                num_iter += 1
                if phase == 'val':   #for AUC calculation
                    y_tru=np.append(y_tru,labels[:, 17].data.cpu().numpy())

                inputs = inputs.to(device)
                labels = labels.to(device)



                # print(labels[0,:],labels.shape)
                # for i in range(17):
                #     print(labels[0, i], labels.shape)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = outputs.double()
                    # print("model output tensor type", outputs.type())
                    _, preds = torch.max(outputs[:, 0:2], 1)
                    if phase == 'val':
                        y_scr=np.append(y_scr,preds.data.cpu().numpy())
                    # Cross entropy loss for PCI
                    # cls_loss = classification_loss(outputs[:, 0:2], labels[:, 3].long())
                    cls_loss = classification_loss(outputs[:, 0:2], labels[:, 17].long())



                    # MSE loss for the numerical readouts
                    # 3 regions
                    # rgn_loss_lad=regression_loss(outputs[:,2],labels[:,0].double())
                    # rgn_loss_lcx=regression_loss(outputs[:,3],labels[:,1].double())
                    # rgn_loss_rca=regression_loss(outputs[:,4],labels[:,2].double())

                    # 16 segments
                    # segment_loss=torch.zeros(1)
                    # for i in range(17):
                    #     print(labels[0, :])
                    #     print(outputs[:, i + 2])
                     # segment_loss+=regression_loss(outputs[:,i+2],labels[:,i].double())
                    segment_loss=(regression_loss(outputs[:,2:],labels[:,:17].double()))  #sum of 17 segments
                    # loss = criterion(outputs, labels)
                    p = 0.8
                    q = 1.0-p
                    # loss=((p*cls_loss.double()) + q*(rgn_loss_lad/3 +rgn_loss_lcx/3 + rgn_loss_rca/3)).double()
                    loss = ((p * cls_loss.double()) + q * (segment_loss)).double()
                    # print(cls_loss.type(),(rgn_loss_rca/3).type())
                    # print(loss.type())
                    # loss=loss
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if lrSch is True and scheduler is not None:
                            scheduler.step()

                # statistics
                if num_iter % 5 == 0:
                    print("Classification loss (weight:{:.2f})/ Regression loss ({:.2f}): {:.3f} / {:.3f}".format(p, q,
                                                                                                          cls_loss.item(),
                                                                                                          segment_loss.item()))

                    print("Total loss: {:.4f} * 1e-03  [{}/{} ({:.0f}%)]".format
                          (loss.item() * 1000, num_iter * bat_size, len(dataloaders[phase]) * bat_size,
                           num_iter / len(dataloaders[phase]) * 100))

                # note loss.item() to get the python number instead of loss.data[0] which now gives a tensor!
                running_loss += loss.item() * inputs.size(0)
                #                 print(running_loss)
                # ****************************************************************************************************
                # running_corrects += torch.sum(preds == labels.data[:, 3])
                running_corrects += torch.sum(preds == labels.data[:, 17])
                # ****************************************************************************************************

                #                 print("torch sum ",torch.sum(preds == labels.data))
                if phase == 'train':
                    iter_loss_vec_train.append(loss.item())
                    # ****************************************************************************************************
                    # iter_acc_vec_train.append(torch.sum(preds == labels.data[:, 3]) / inputs.size(0))
                    iter_acc_vec_train.append(torch.sum(preds == labels.data[:, 17]) / inputs.size(0))
                # ****************************************************************************************************

                if phase == 'val':
                    iter_loss_vec_val.append(loss.item())
                    # ****************************************************************************************************
                    # iter_acc_vec_val.append(torch.sum(preds == labels.data[:, 3]) / inputs.size(0))
                    iter_acc_vec_val.append(torch.sum(preds == labels.data[:, 17]) / inputs.size(0))
                    # ****************************************************************************************************

            epoch_loss = running_loss / (num_iter * bat_size)
            epoch_acc = running_corrects.double() / (num_iter * bat_size)
            if phase =='val':
                epoch_auc=roc_auc_score(y_tru,y_scr)
                auc_vec_val.append(epoch_auc)
            else:
                epoch_auc=-99
            print('{} Loss: {:.4f} Acc: {:.4f} AUC:{:.3f}'.format(
                phase, epoch_loss, epoch_acc,epoch_auc))
            # print duration for training one epoch
            time_elapsed = time.time() - since
            print("Time elapsed: {:.0f}m {:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60))

            if phase == 'train':
                loss_vec_train.append(epoch_loss)
                acc_vec_train.append(epoch_acc)

            if phase == 'val':
                loss_vec_val.append(epoch_loss)
                acc_vec_val.append(epoch_acc)

            # deep copy the model if the new model performs better by some margin
            if phase == 'val' and epoch >= 2 and epoch_acc - 0.005 >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # torch.save(model.state_dict(), model_save_path)
                print('Epoch: {} current best: {} model copied '.format(epoch, best_acc))

            if phase == 'val' and len(acc_vec_val) > 5:
                past_avg_val_acc = float(sum(acc_vec_val[-5:]) / len(acc_vec_val[-5:]))
                if (epoch_acc - past_avg_val_acc) < 0.005:
                    print("Epoch {}: val acc has not improved by 0.5% compared with average of last 5 epochs".format(
                        epoch))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    if save_model:
        torch.save(model, model_save_path)
        print("model saved to {}".format(model_save_path))

    try:
        plot_prefix = model_save_path[:-4]
        stat_file_name = plot_prefix + '_stat.txt'
        out_stat = np.array((loss_vec_train, loss_vec_val, acc_vec_train, acc_vec_val,auc_vec_val))
        np.savetxt(stat_file_name, out_stat, delimiter='\t')
        stat_file_name = plot_prefix + '_stat_iter.txt'
        # out_stat_it = np.array((iter_loss_vec_train,iter_loss_vec_val,iter_acc_vec_train,iter_acc_vec_val))
        # np.savetxt(stat_file_name, out_stat_it, delimiter='\t')
        with open(stat_file_name, "w+") as iter_o:
            print("{}".format(("\t".join(iter_loss_vec_val))), file=iter_o)
        with open(stat_file_name, "a") as iter_o:
            print("{}".format(("\t".join(iter_acc_vec_train))), file=iter_o)
        with open(stat_file_name, "a") as iter_o:
            print("{}".format(("\t".join(iter_acc_vec_train))), file=iter_o)
        with open(stat_file_name, "a") as iter_o:
            print("{}".format(("\t".join(iter_acc_vec_val))), file=iter_o)



    except:
        print("Stats not saved")
        pass
    if plot:
        try:
            plot_prefix = model_save_path[:-4]
            plt.figure()
            plt.plot(np.arange(1, num_epochs + 1), loss_vec_train, 'b')
            plt.plot(np.arange(1, num_epochs + 1), loss_vec_val, 'r')  # visualize loss change
            plt.title('Average loss of each epoch')
            loss_plot_name = plot_prefix + '_loss.png'
            plt.savefig(loss_plot_name, bbox_inches='tight')
            plt.figure()
            plt.plot(np.arange(1, num_epochs + 1), acc_vec_train, 'b')
            plt.plot(np.arange(1, num_epochs + 1), acc_vec_val, 'r')
            plt.title('Average accuracy of each epoch')
            acc_plot_name = plot_prefix + '_acc.png'
            plt.savefig(acc_plot_name, bbox_inches='tight')
            plt.figure()
            plt.plot(np.arange(1, num_epochs + 1), auc_vec_val, 'crimson')
            plt.title('AUC of each epoch in validation')
            acc_plot_name = plot_prefix + '_valAuc.png'
            plt.savefig(acc_plot_name, bbox_inches='tight')
        except:
            print("Figures not generated")
            pass
    if stat_out:
        out_stat_it = list((iter_loss_vec_train, iter_loss_vec_val, iter_acc_vec_train, iter_acc_vec_val))

        return model, out_stat.tolist(), out_stat_it
    else:
        return model



# def train_multiRegLoss(dataloaders, model, criterions, optimizer, model_save_path, device, bat_size,
#                     num_epochs=25, lrSch=False,
#                     scheduler=None, save_model=True, stat_out=False, plot=True):
#     """training along with validation """
#
#     since = time.time()
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#     rmse_vec_train = []  # average training loss of each epoch
#     rmse_vec_val = []
#     loss_vec_train = []  # average training loss of each epoch
#     loss_vec_val = []  # average test loss of each epoch
#     adjusted_r_vec_val=[]
#     iter_loss_vec_train, iter_acc_vec_train = [], []
#     iter_loss_vec_val, iter_acc_vec_val = [], []
#     regression_loss1 = criterions[0]
#     regression_loss2 = criterions[1]
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 30)
#
#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()  # Set model to evaluate mode important to network with batchnorm layer
#                 # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/2
#             y_tru = np.array([])
#             y_scr = np.array([])
#             running_loss = 0.0
#             running_corrects = 0
#             num_iter = 0
#             # Iterate over data.
#             for inputs, labels in dataloaders[phase]:
#                 num_iter += 1
#                 if phase == 'val':   #for AUC calculation
#                     y_tru=np.append(y_tru,labels[:, 17].data.cpu().numpy())
#
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#
#
#                 # print(labels[0,:],labels.shape)
#                 # for i in range(17):
#                 #     print(labels[0, i], labels.shape)
#
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#
#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     outputs = outputs.double()
#                     # print("model output tensor type", outputs.type())
#                     _, preds = torch.max(outputs[:, 0:2], 1)
#                     if phase == 'val':
#                         y_scr=np.append(y_scr,preds.data.cpu().numpy())
#                     # Cross entropy loss for PCI
#                     # cls_loss = classification_loss(outputs[:, 0:2], labels[:, 3].long())
#                     # cls_loss = classification_loss(outputs[:, 0:2], labels[:, 17].long())
#
#
#
#                     # MSE loss for the numerical readouts
#                     # 3 regions
#                     # rgn_loss_lad=regression_loss(outputs[:,2],labels[:,0].double())
#                     # rgn_loss_lcx=regression_loss(outputs[:,3],labels[:,1].double())
#                     # rgn_loss_rca=regression_loss(outputs[:,4],labels[:,2].double())
#
#                     # 16 segments
#                     # segment_loss=torch.zeros(1)
#                     # for i in range(17):
#                     #     print(labels[0, :])
#                     #     print(outputs[:, i + 2])
#                      # segment_loss+=regression_loss(outputs[:,i+2],labels[:,i].double())
#                     segment_loss=(regression_loss2(outputs[:,2:],labels[:,:17].double()))/17  #sum of 17 segments
#                     target_loss=torch.zeros(1)
#                     # loss = criterion(outputs, labels)
#                     p = 0.2
#                     q = 1.0-p
#                     # loss=((p*cls_loss.double()) + q*(rgn_loss_lad/3 +rgn_loss_lcx/3 + rgn_loss_rca/3)).double()
#                     loss = ((p * target_loss.double()) + q * (segment_loss)).double()
#                     # print(cls_loss.type(),(rgn_loss_rca/3).type())
#                     # print(loss.type())
#                     # loss=loss
#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#                         if lrSch is True and scheduler is not None:
#                             scheduler.step()
#
#                 # statistics
#                 if num_iter % 5 == 0:
#                     print("Classification loss (weight:{:.2f})/ Regression loss ({:.2f}): {:.3f} / {:.3f}".format(p, q,
#                                                                                                           target_loss.item(),
#                                                                                                           segment_loss.item()))
#
#                     print("Total loss: {:.4f} * 1e-03  [{}/{} ({:.0f}%)]".format
#                           (loss.item() * 1000, num_iter * bat_size, len(dataloaders[phase]) * bat_size,
#                            num_iter / len(dataloaders[phase]) * 100))
#
#                 # note loss.item() to get the python number instead of loss.data[0] which now gives a tensor!
#                 running_loss += loss.item() * inputs.size(0)
#                 #                 print(running_loss)
#                 # ****************************************************************************************************
#                 # running_corrects += torch.sum(preds == labels.data[:, 3])
#                 running_corrects += torch.sum(preds == labels.data[:, 17])
#                 # ****************************************************************************************************
#
#                 #                 print("torch sum ",torch.sum(preds == labels.data))
#                 if phase == 'train':
#                     iter_loss_vec_train.append(loss.item())
#                     # ****************************************************************************************************
#                     # iter_acc_vec_train.append(torch.sum(preds == labels.data[:, 3]) / inputs.size(0))
#                     iter_acc_vec_train.append(torch.sum(preds == labels.data[:, 17]) / inputs.size(0))
#                 # ****************************************************************************************************
#
#                 if phase == 'val':
#                     iter_loss_vec_val.append(loss.item())
#                     # ****************************************************************************************************
#                     # iter_acc_vec_val.append(torch.sum(preds == labels.data[:, 3]) / inputs.size(0))
#                     iter_acc_vec_val.append(torch.sum(preds == labels.data[:, 17]) / inputs.size(0))
#                     # ****************************************************************************************************
#
#             epoch_loss = running_loss / (num_iter * bat_size)
#             epoch_acc = running_corrects.double() / (num_iter * bat_size)
#             if phase =='val':
#                 epoch_auc=roc_auc_score(y_tru,y_scr)
#                 auc_vec_val.append(epoch_auc)
#             else:
#                 epoch_auc=-99
#             print('{} Loss: {:.4f} Acc: {:.4f} AUC:{:.3f}'.format(
#                 phase, epoch_loss, epoch_acc,epoch_auc))
#             # print duration for training one epoch
#             time_elapsed = time.time() - since
#             print("Time elapsed: {:.0f}m {:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60))
#
#             if phase == 'train':
#                 loss_vec_train.append(epoch_loss)
#                 acc_vec_train.append(epoch_acc)
#
#             if phase == 'val':
#                 loss_vec_val.append(epoch_loss)
#                 acc_vec_val.append(epoch_acc)
#
#             # deep copy the model if the new model performs better by some margin
#             if phase == 'val' and epoch >= 2 and epoch_acc - 0.005 >= best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
#                 # torch.save(model.state_dict(), model_save_path)
#                 print('Epoch: {} current best: {} model copied '.format(epoch, best_acc))
#
#             if phase == 'val' and len(acc_vec_val) > 5:
#                 past_avg_val_acc = float(sum(acc_vec_val[-5:]) / len(acc_vec_val[-5:]))
#                 if (epoch_acc - past_avg_val_acc) < 0.005:
#                     print("Epoch {}: val acc has not improved by 0.5% compared with average of last 5 epochs".format(
#                         epoch))

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    if save_model:
        torch.save(model, model_save_path)
        print("model saved to {}".format(model_save_path))

    try:
        plot_prefix = model_save_path[:-4]
        stat_file_name = plot_prefix + '_stat.txt'
        out_stat = np.array((loss_vec_train, loss_vec_val, acc_vec_train, acc_vec_val,auc_vec_val))
        np.savetxt(stat_file_name, out_stat, delimiter='\t')
        stat_file_name = plot_prefix + '_stat_iter.txt'
        # out_stat_it = np.array((iter_loss_vec_train,iter_loss_vec_val,iter_acc_vec_train,iter_acc_vec_val))
        # np.savetxt(stat_file_name, out_stat_it, delimiter='\t')
        with open(stat_file_name, "w+") as iter_o:
            print("{}".format(("\t".join(iter_loss_vec_val))), file=iter_o)
        with open(stat_file_name, "a") as iter_o:
            print("{}".format(("\t".join(iter_acc_vec_train))), file=iter_o)
        with open(stat_file_name, "a") as iter_o:
            print("{}".format(("\t".join(iter_acc_vec_train))), file=iter_o)
        with open(stat_file_name, "a") as iter_o:
            print("{}".format(("\t".join(iter_acc_vec_val))), file=iter_o)



    except:
        print("Stats not saved")
        pass
    if plot:
        try:
            plot_prefix = model_save_path[:-4]
            plt.figure()
            plt.plot(np.arange(1, num_epochs + 1), loss_vec_train, 'b')
            plt.plot(np.arange(1, num_epochs + 1), loss_vec_val, 'r')  # visualize loss change
            plt.title('Average loss of each epoch')
            loss_plot_name = plot_prefix + '_loss.png'
            plt.savefig(loss_plot_name, bbox_inches='tight')
            plt.figure()
            plt.plot(np.arange(1, num_epochs + 1), acc_vec_train, 'b')
            plt.plot(np.arange(1, num_epochs + 1), acc_vec_val, 'r')
            plt.title('Average accuracy of each epoch')
            acc_plot_name = plot_prefix + '_acc.png'
            plt.savefig(acc_plot_name, bbox_inches='tight')
            plt.figure()
            plt.plot(np.arange(1, num_epochs + 1), auc_vec_val, 'crimson')
            plt.title('AUC of each epoch in validation')
            acc_plot_name = plot_prefix + '_valAuc.png'
            plt.savefig(acc_plot_name, bbox_inches='tight')
        except:
            print("Figures not generated")
            pass
    if stat_out:
        out_stat_it = list((iter_loss_vec_train, iter_loss_vec_val, iter_acc_vec_train, iter_acc_vec_val))

        return model, out_stat.tolist(), out_stat_it
    else:
        return model


#
#
# def train_regressor(dataloaders, model, criterion, optimizer, model_save_path, device, bat_size,
#                                num_epochs=25, lrSch=False,
#                                scheduler=None, save_model=True, stat_out=False, plot=True):
#     """training along with validation """
#
#     since = time.time()
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_auc = 0.5
#     loss_vec_train = []  # average training loss of each epoch
#     loss_vec_val = []  # average test loss of each epoch
#     auc_vec_val=[]
#     iter_loss_vec_train= []
#     iter_loss_vec_val = []
#     regression_loss = criterion
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 30)
#
#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()  # Set model to evaluate mode important to network with batchnorm layer
#                 # https://discuss.pytorch.org/t/error-expected-more-than-1-value-per-channel-when-training/26274/2
#             y_tru = np.array([])
#             y_scr = np.array([])
#             running_loss = 0.0
#             running_corrects = 0
#             num_iter = 0
#             # Iterate over data.
#             for inputs, labels in dataloaders[phase]:
#                 num_iter += 1
#                 if phase == 'val':   #for AUC calculation
#                     y_tru=np.append(y_tru,labels[:, 17].data.cpu().numpy())
#
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#
#
#                 # print(labels[0,:],labels.shape)
#                 # for i in range(17):
#                 #     print(labels[0, i], labels.shape)
#
#                 # zero the parameter gradients
#                 optimizer.zero_grad()
#
#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     outputs = outputs.double()
#                     # print("model output tensor type", outputs.type())
#                     _, preds = torch.max(outputs[:, 0:2], 1)
#                     if phase == 'val':
#                         y_scr=np.append(y_scr,preds.data.cpu().numpy())
#                     # Cross entropy loss for PCI
#                     # cls_loss = classification_loss(outputs[:, 0:2], labels[:, 3].long())
#                     cls_loss = classification_loss(outputs[:, 0:2], labels[:, 17].long())
#
#
#
#                     # MSE loss for the numerical readouts
#                     # 3 regions
#                     # rgn_loss_lad=regression_loss(outputs[:,2],labels[:,0].double())
#                     # rgn_loss_lcx=regression_loss(outputs[:,3],labels[:,1].double())
#                     # rgn_loss_rca=regression_loss(outputs[:,4],labels[:,2].double())
#
#                     # 16 segments
#                     # segment_loss=torch.zeros(1)
#                     # for i in range(17):
#                     #     print(labels[0, :])
#                     #     print(outputs[:, i + 2])
#                      # segment_loss+=regression_loss(outputs[:,i+2],labels[:,i].double())
#                     segment_loss=(regression_loss(outputs[:,2:],labels[:,:17].double()))/17  #sum of 17 segments
#                     # loss = criterion(outputs, labels)
#                     p = 0.2
#                     q = 1.0-p
#                     # loss=((p*cls_loss.double()) + q*(rgn_loss_lad/3 +rgn_loss_lcx/3 + rgn_loss_rca/3)).double()
#                     loss = ((p * cls_loss.double()) + q * (segment_loss)).double()
#                     # print(cls_loss.type(),(rgn_loss_rca/3).type())
#                     # print(loss.type())
#                     # loss=loss
#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#                         if lrSch is True and scheduler is not None:
#                             scheduler.step()
#
#                 # statistics
#                 if num_iter % 5 == 0:
#                     print("Classification loss (weight:{:.2f})/ Regression loss ({:.2f}): {:.3f} / {:.3f}".format(p, q,
#                                                                                                           cls_loss.item(),
#                                                                                                           segment_loss.item()))
#
#                     print("Total loss: {:.4f} * 1e-03  [{}/{} ({:.0f}%)]".format
#                           (loss.item() * 1000, num_iter * bat_size, len(dataloaders[phase]) * bat_size,
#                            num_iter / len(dataloaders[phase]) * 100))
#
#                 # note loss.item() to get the python number instead of loss.data[0] which now gives a tensor!
#                 running_loss += loss.item() * inputs.size(0)
#                 #                 print(running_loss)
#                 # ****************************************************************************************************
#                 # running_corrects += torch.sum(preds == labels.data[:, 3])
#                 running_corrects += torch.sum(preds == labels.data[:, 17])
#                 # ****************************************************************************************************
#
#                 #                 print("torch sum ",torch.sum(preds == labels.data))
#                 if phase == 'train':
#                     iter_loss_vec_train.append(loss.item())
#                     # ****************************************************************************************************
#                     # iter_acc_vec_train.append(torch.sum(preds == labels.data[:, 3]) / inputs.size(0))
#                     iter_acc_vec_train.append(torch.sum(preds == labels.data[:, 17]) / inputs.size(0))
#                 # ****************************************************************************************************
#
#                 if phase == 'val':
#                     iter_loss_vec_val.append(loss.item())
#                     # ****************************************************************************************************
#                     # iter_acc_vec_val.append(torch.sum(preds == labels.data[:, 3]) / inputs.size(0))
#                     iter_acc_vec_val.append(torch.sum(preds == labels.data[:, 17]) / inputs.size(0))
#                     # ****************************************************************************************************
#
#             epoch_loss = running_loss / (num_iter * bat_size)
#             epoch_acc = running_corrects.double() / (num_iter * bat_size)
#             if phase =='val':
#                 epoch_auc=roc_auc_score(y_tru,y_scr)
#                 auc_vec_val.append(epoch_auc)
#             else:
#                 epoch_auc=-99
#             print('{} Loss: {:.4f} Acc: {:.4f} AUC:{:.3f}'.format(
#                 phase, epoch_loss, epoch_acc,epoch_auc))
#             # print duration for training one epoch
#             time_elapsed = time.time() - since
#             print("Time elapsed: {:.0f}m {:.0f}s\n".format(time_elapsed // 60, time_elapsed % 60))
#
#             if phase == 'train':
#                 loss_vec_train.append(epoch_loss)
#                 acc_vec_train.append(epoch_acc)
#
#             if phase == 'val':
#                 loss_vec_val.append(epoch_loss)
#                 acc_vec_val.append(epoch_acc)
#
#             # deep copy the model if the new model performs better by some margin
#             if phase == 'val' and epoch >= 2 and epoch_acc - 0.005 >= best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(model.state_dict())
#                 # torch.save(model.state_dict(), model_save_path)
#                 print('Epoch: {} current best: {} model copied '.format(epoch, best_acc))
#
#             if phase == 'val' and len(acc_vec_val) > 5:
#                 past_avg_val_acc = float(sum(acc_vec_val[-5:]) / len(acc_vec_val[-5:]))
#                 if (epoch_acc - past_avg_val_acc) < 0.005:
#                     print("Epoch {}: val acc has not improved by 0.5% compared with average of last 5 epochs".format(
#                         epoch))
#
#         print()
#
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))
#
#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     if save_model:
#         torch.save(model, model_save_path)
#         print("model saved to {}".format(model_save_path))
#
#     try:
#         plot_prefix = model_save_path[:-4]
#         stat_file_name = plot_prefix + '_stat.txt'
#         out_stat = np.array((loss_vec_train, loss_vec_val, acc_vec_train, acc_vec_val,auc_vec_val))
#         np.savetxt(stat_file_name, out_stat, delimiter='\t')
#         stat_file_name = plot_prefix + '_stat_iter.txt'
#         # out_stat_it = np.array((iter_loss_vec_train,iter_loss_vec_val,iter_acc_vec_train,iter_acc_vec_val))
#         # np.savetxt(stat_file_name, out_stat_it, delimiter='\t')
#         with open(stat_file_name, "w+") as iter_o:
#             print("{}".format(("\t".join(iter_loss_vec_val))), file=iter_o)
#         with open(stat_file_name, "a") as iter_o:
#             print("{}".format(("\t".join(iter_acc_vec_train))), file=iter_o)
#         with open(stat_file_name, "a") as iter_o:
#             print("{}".format(("\t".join(iter_acc_vec_train))), file=iter_o)
#         with open(stat_file_name, "a") as iter_o:
#             print("{}".format(("\t".join(iter_acc_vec_val))), file=iter_o)
#
#
#
#     except:
#         print("Stats not saved")
#         pass
#     if plot:
#         try:
#             plot_prefix = model_save_path[:-4]
#             plt.figure()
#             plt.plot(np.arange(1, num_epochs + 1), loss_vec_train, 'b')
#             plt.plot(np.arange(1, num_epochs + 1), loss_vec_val, 'r')  # visualize loss change
#             plt.title('Average loss of each epoch')
#             loss_plot_name = plot_prefix + '_loss.png'
#             plt.savefig(loss_plot_name, bbox_inches='tight')
#             plt.figure()
#             plt.plot(np.arange(1, num_epochs + 1), acc_vec_train, 'b')
#             plt.plot(np.arange(1, num_epochs + 1), acc_vec_val, 'r')
#             plt.title('Average accuracy of each epoch')
#             acc_plot_name = plot_prefix + '_acc.png'
#             plt.savefig(acc_plot_name, bbox_inches='tight')
#             plt.figure()
#             plt.plot(np.arange(1, num_epochs + 1), auc_vec_val, 'crimson')
#             plt.title('AUC of each epoch in validation')
#             acc_plot_name = plot_prefix + '_valAuc.png'
#             plt.savefig(acc_plot_name, bbox_inches='tight')
#         except:
#             print("Figures not generated")
#             pass
#     if stat_out:
#         out_stat_it = list((iter_loss_vec_train, iter_loss_vec_val, iter_acc_vec_train, iter_acc_vec_val))
#
#         return model, out_stat.tolist(), out_stat_it
#     else:
#         return model
