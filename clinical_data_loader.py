import os
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
from PIL import Image
import torch
import numpy as np

def read_data(fn, df):
    img = Image.open(fn)
    
    folders = fn.split('/')
    img_name = folders[-1]
    
    
    p_number = img_name.split('.')[0] 
    
    try:
        p_number = p_number.split('_')[0]
    except:
        pass

    p_number = int(p_number)
        
    p_info = df['number'] == p_number
    label = df[p_info]['EGFR']
    label = label.values[0]
    sex = df[p_info]['Sex']
    sex = sex.values[0]
    age = df[p_info]['Age_norm']
    age = age.values[0]
    smoke = df[p_info]['Smoking']
    smoke = smoke.values[0]
        
    return img, label, sex, age, smoke, p_number


class ImageDataset(Dataset):
    def __init__(self, root, imgs_name, clinical_root, mode, transform=None):
        super(ImageDataset, self).__init__()
        
        img_list = os.listdir(root)
        
        filenames = list()
        for imgs in img_list:
            if 'png' in imgs:
                im_number = imgs.split('.')[0]
                
                try:
                    im_number = im_number.split('_')[0]
                except:
                    pass
                
                im_number = int(im_number)
                if im_number in imgs_name:
                    filenames.append(os.path.join(root, imgs))
                    
        if mode == 'test':
            df = pd.read_excel(clinical_root, sheet_name = 'Test')

        else:
            df = pd.read_excel(clinical_root, sheet_name = 'Training')

        
        self.clinical_root = clinical_root
        self.filenames = filenames
        self.root = root
        self.transform = transform
        self.df = df
        
    def __len__(self):
        return len(self.filenames)
    
    
    def __getitem__(self, index):
        img, label, sex, age, smoke, p_number = read_data(self.filenames[index], self.df)
        
        if self.transform:
            img = self.transform(img)
        if label == 0:
            target = torch.tensor([1, 0])
        else:
            target = torch.tensor([0, 1])

        clinical_np = np.array([sex, age, smoke], dtype=np.float32)
        clinical = torch.from_numpy(clinical_np)        

            
        return img, target, clinical, p_number

class v_ImageDataset(Dataset):
    def __init__(self, root, clinical_root, mode, transform=None):
        super(ImageDataset, self).__init__()
        
        img_list = os.listdir(root)
        filenames = [i for i in img_list if 'png' in i]

                    
        if mode == 'val':
            df = pd.read_excel(clinical_root, sheet_name = 'val')

        else:
            df = pd.read_excel(clinical_root, sheet_name = 'Training')

        self.clinical_root = clinical_root
        self.filenames = filenames
        self.root = root
        self.transform = transform
        self.df = df
        
    def __len__(self):
        return len(self.filenames)
    
    
    def __getitem__(self, index):
        img, label, sex, age, smoke, p_number = read_data(os.path.join(self.root, self.filenames[index]), self.df)
        
        if self.transform:
            img = self.transform(img)
        if label == 0:
            target = torch.tensor([1, 0])
        else:
            target = torch.tensor([0, 1])

        clinical_np = np.array([sex, age, smoke], dtype=np.float32)
        clinical = torch.from_numpy(clinical_np)        

            
        return img, target, clinical, p_number

class t_ImageDataset(Dataset):
    def __init__(self, root, clinical_root, mode, transform=None):
        super(ImageDataset, self).__init__()
        
        img_list = os.listdir(root)
        filenames = [os.path.join(root, imgs) for imgs in img_list if 'png' in imgs]

                    
        if mode == 'test':
            df = pd.read_excel(clinical_root, sheet_name = 'Test')

        else:
            df = pd.read_excel(clinical_root, sheet_name = 'Training')

        
        self.clinical_root = clinical_root
        self.filenames = filenames
        self.root = root
        self.transform = transform
        self.df = df
        
    def __len__(self):
        return len(self.filenames)
    
    
    def __getitem__(self, index):
        img, label, sex, age, smoke, p_number = read_data(self.filenames[index], self.df)
        
        if self.transform:
            img = self.transform(img)
        if label == 0:
            target = torch.tensor([1, 0])
        else:
            target = torch.tensor([0, 1])

        clinical_np = np.array([sex, age, smoke], dtype=np.float32)
        clinical = torch.from_numpy(clinical_np)        

            
        return img, target, clinical, p_number
