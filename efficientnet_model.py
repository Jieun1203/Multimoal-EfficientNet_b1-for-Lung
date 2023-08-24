import torch.nn.functional as F
import  torchvision.models as models
import torch.nn as nn


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
