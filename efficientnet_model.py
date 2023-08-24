import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
import glob, os, re
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import random
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

import  torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, confusion_matrix
from efficientnet_pytorch import EfficientNet

class MyEffientnet_b1_clinical(nn.Module):
    def __init__(self,out_features1, out_features2, out_features3, out_features4, out_features5, out_features6, 
                 out_features7, out_features8, model_name='efficientnet-b1',class_num=45,initfc_type='normal',gain=0.2):
        super(MyEffientnet_b1_clinical, self).__init__()
        
        
        self.clinical_fc1 = nn.Linear(3, out_features5) #(3, 32)
        self.clinical_fc2 = nn.Linear(out_features5, out_features6) #(32, 256)
        self.clinical_fc3 = nn.Linear(out_features6, out_features7) #(256, 1024)
        self.clinical_fc4 = nn.Linear(out_features7, out_features8) #(1024, 1280)

        
        
        model = EfficientNet.from_pretrained(model_name)
        self.model = model
        self.fc1 = nn.Linear(1280 + out_features8, out_features1) #1280
        self.fc2 = nn.Linear(out_features1, out_features2)
        self.fc3 = nn.Linear(out_features2, out_features3)
        self.fc4 = nn.Linear(out_features3, out_features4)
        self.fc5 = nn.Linear(out_features4, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.batchnorm = nn.BatchNorm1d(1280 + out_features8)
        self.batchnorm1 = nn.BatchNorm1d(out_features1)
        self.batchnorm2 = nn.BatchNorm1d(out_features2)
        self.batchnorm3 = nn.BatchNorm1d(out_features3)
        self.batchnorm4 = nn.BatchNorm1d(out_features4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        

        
        
        if hasattr(self.fc1, 'bias') and self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias.data, 0.0)
        if initfc_type == 'normal':
            nn.init.normal_(self.fc1.weight.data, 0.0, gain)
        elif initfc_type == 'xavier':
            nn.init.xavier_normal_(self.fc1.weight.data, gain=gain)
        elif initfc_type == 'kaiming':
            nn.init.kaiming_normal_(self.fc1.weight.data, a=0, mode='fan_in')
        elif initfc_type == 'orthogonal':
            nn.init.orthogonal_(self.fc1.weight.data, gain=gain)


    def forward(self,x, c):
        x = self.model.extract_features(x)
        x = x * torch.sigmoid(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        
        c = self.clinical_fc1(c)
        c = self.clinical_fc2(c)
        c = self.clinical_fc3(c)
        c = self.clinical_fc4(c)

        x = torch.cat((x, c), 1)
        
        x = self.batchnorm(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = F.softmax(x, dim=1)
        return x



def model_training(num_epoch, my_model, train_loader, val_loader, ex_test_loader, save_root):
    train_acc_check = np.array([])
    train_auc_check = np.array([])
    val_loss_check = np.array([])
    val_auc_check = np.array([])
    val_acc_check = np.array([])
    test_acc_check = np.array([])
    test_auc_check = np.array([])
    ex_test_acc_check = np.array([])
    ex_test_auc_check = np.array([])
    ex_val_auc_check = np.array([])
    ex_val_acc_check = np.array([])

    class_weight = torch.FloatTensor([0.6, 0.4]).cuda()
    criterion = nn.CrossEntropyLoss(class_weight)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=9e-5, weight_decay=1e-5)    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.99 ** epoch
    
    for epoch in range(num_epoch):
        epoch_loss_train = 0.0
        epoch_train_acc = 0.0
        predicted_train_output = np.array([])
        train_real = np.array([])
        train_probability = np.array([]).reshape(0, 2)
    
        my_model.train()
        for enu, (train_x_batch, train_y_batch, train_clinical_batch, p) in enumerate(tqdm(train_loader)):
            
            train_x = Variable(train_x_batch).cuda()
            train_y = Variable(train_y_batch).cuda()
            train_clinical = Variable(train_clinical_batch).cuda()
    
            optimizer.zero_grad()
    
            train_output = my_model(train_x, train_clinical)
            train_epoch_loss = criterion(train_output, torch.max(train_y, 1)[1])
    
            train_pred = np.argmax(train_output.detach().data.cpu().numpy(), axis = 1)
            train_true = np.argmax(train_y.detach().data.cpu().numpy(), axis = 1)
            predicted_train_output = np.append(predicted_train_output, train_pred)
            train_real = np.append(train_real, train_true)
            train_probability = np.append(train_probability, train_output.detach().data.cpu().numpy(), axis = 0)
    
            train_epoch_loss.backward()
            optimizer.step()
    
            epoch_loss_train += (train_epoch_loss.data.item() * len(train_x_batch))
    
    
        del train_x_batch, train_y_batch, train_output
        train_loss = epoch_loss_train / len(train_dataset)
        train_acc = len(np.where(predicted_train_output == train_real)[0]) / len(predicted_train_output)
        train_auc_score = roc_auc_score(train_real, train_probability[:, 1])
        
        train_acc_check = np.append(train_acc_check, train_acc)
        train_auc_check = np.append(train_auc_check, train_auc_score)
    
        with torch.no_grad():
            epoch_loss_val = 0.0
            epoch_acc_val = 0.0
            predicted_val_output = np.array([])
            val_real = np.array([])
            val_probability = np.array([]).reshape(0, 2)
    
            my_model.eval()
    
            for enu, (validation_x_batch, validation_y_batch, validation_clinical_batch, p) in enumerate(tqdm(val_loader)):
                validation_x = Variable(validation_x_batch).cuda()
                validation_y = Variable(validation_y_batch).cuda()
                vaidation_clinical = Variable(validation_clinical_batch).cuda()
    
                validation_output = my_model(validation_x, vaidation_clinical)
                validation_epoch_loss = criterion(validation_output, torch.max(validation_y, 1)[1])
    
                epoch_loss_val += (validation_epoch_loss.data.item() * len(validation_x_batch))
    
                pred_val = np.argmax(validation_output.data.cpu().numpy(), axis = 1)
                true_val = np.argmax(validation_y.data.cpu().numpy(), axis = 1)
                predicted_val_output = np.append(predicted_val_output, pred_val)
                val_real = np.append(val_real, true_val)
                val_probability = np.append(val_probability, validation_output.detach().data.cpu().numpy(), axis = 0)
    
    
            del validation_x_batch, validation_y_batch, validation_output
            val_loss = epoch_loss_val / len(val_dataset)
            val_acc = len(np.where(predicted_val_output == val_real)[0]) / len(predicted_val_output)
            val_auc_score = roc_auc_score(val_real, val_probability[:, 1])
            val_auc_check = np.append(val_auc_check, val_auc_score)
            val_loss_check = np.append(val_loss_check, val_loss)
            val_acc_check = np.append(val_acc_check, val_acc)
            
            
            
        with torch.no_grad():
            epoch_loss_test = 0.0
            epoch_acc_test = 0.0
            predicted_test_output = np.array([])
            test_real = np.array([])
            test_probability = np.array([]).reshape(0, 2)
    
            my_model.eval()
    
            for enu, (test_x_batch, test_y_batch, test_clinical_batch, p) in enumerate(tqdm(test_loader)):
                test_x = Variable(test_x_batch).cuda()
                test_y = Variable(test_y_batch).cuda()
                test_clinical = Variable(test_clinical_batch).cuda()
    
                test_output = my_model(test_x, test_clinical)
                test_epoch_loss = criterion(test_output, torch.max(test_y, 1)[1])
    
                epoch_loss_test += (test_epoch_loss.data.item() * len(test_x_batch))
    
                pred_test = np.argmax(test_output.data.cpu().numpy(), axis = 1)
                true_test = np.argmax(test_y.data.cpu().numpy(), axis = 1)
                predicted_test_output = np.append(predicted_test_output, pred_test)
                test_real = np.append(test_real, true_test)
                test_probability = np.append(test_probability, test_output.detach().data.cpu().numpy(), axis = 0)
    
    
            del test_x_batch, test_y_batch, test_output
            test_loss = epoch_loss_test / len(test_dataset)
            test_acc = len(np.where(predicted_test_output == test_real)[0]) / len(predicted_test_output)
            test_auc_score = roc_auc_score(test_real, test_probability[:, 1])
            test_auc_check = np.append(test_auc_check, test_auc_score)
            test_acc_check = np.append(test_acc_check, test_acc)
    
            
            
        with torch.no_grad():
            ex_epoch_loss_test = 0.0
            ex_epoch_acc_test = 0.0
            ex_predicted_test_output = np.array([])
            ex_test_real = np.array([])
            ex_test_probability = np.array([]).reshape(0, 2)
    
            my_model.eval()
    
            for enu, (ex_test_x_batch, ex_test_y_batch, ex_test_clinical_batch, p_number) in enumerate(tqdm(ex_test_loader)):
                ex_test_x = Variable(ex_test_x_batch).cuda()
                ex_test_y = Variable(ex_test_y_batch).cuda()
                ex_test_clinical = Variable(ex_test_clinical_batch).cuda()
    
                ex_test_output = my_model(ex_test_x, ex_test_clinical)
                ex_test_epoch_loss = criterion(ex_test_output, torch.max(ex_test_y, 1)[1])
    
                ex_epoch_loss_test += (ex_test_epoch_loss.data.item() * len(ex_test_x_batch))
    
                ex_pred_test = np.argmax(ex_test_output.data.cpu().numpy(), axis = 1)
                ex_true_test = np.argmax(ex_test_y.data.cpu().numpy(), axis = 1)
                ex_predicted_test_output = np.append(ex_predicted_test_output, ex_pred_test)
                ex_test_real = np.append(ex_test_real, ex_true_test)
                ex_test_probability = np.append(ex_test_probability, ex_test_output.detach().data.cpu().numpy(), axis = 0)
    
    
            del ex_test_x_batch, ex_test_y_batch, ex_test_output
            ex_test_loss = ex_epoch_loss_test / len(ex_test_dataset)
            ex_test_acc = len(np.where(ex_predicted_test_output == ex_test_real)[0]) / len(ex_predicted_test_output)
            ex_test_auc_score = roc_auc_score(ex_test_real, ex_test_probability[:, 1])
            ex_test_auc_check = np.append(ex_test_auc_check, ex_test_auc_score)
            ex_test_acc_check = np.append(ex_test_acc_check, ex_test_acc)
    
        if val_loss_check[epoch] == val_loss_check.min():
            test_value = test_auc_check[epoch]
    
        scheduler.step()
        
    if val_auc_check[epoch] == val_auc_check.max():
        torch.save(my_model.state_dict(), save_root + '/classification_checkpoint.pt')

        print('Epoch:[{}]/[{}]\t'
              'train auc: {:.2f} '
             'acc: {:.2f}\t '
              'val auc: {:.2f} '
             'acc: {:.2f}\n'
             'test auc: {:.2f} '
             'acc: {:.2f}\t'
               'ex_test auc: {:.2f} '
             'acc: {:.2f}\t'
              .format(epoch, num_epoch, train_auc_score, train_acc, 
                                          val_auc_score, val_acc, test_auc_score, test_acc, 
                      ex_test_auc_score, ex_test_acc))
    print('Final model saved at ' + save_root + '/classification_checkpoint.pt')
