import random
import numpy as np

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from utils.common import read_annotations
from data.transforms import MultiCropTransform, get_transforms

class ImageDataset(Dataset):
    def __init__(self, annotations, config, opt, balance=False):
        self.opt = opt
        self.config = config
        self.balance = balance
        self.class_num=config.class_num
        self.resize_size = config.resize_size
        self.second_resize_size = config.second_resize_size
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        if balance:
            self.data = [[x for x in annotations if x[1] == lab] for lab in [i for i in range(self.class_num)]]
        else:
            self.data = [annotations]

    def __len__(self):
        
        return max([len(subset) for subset in self.data])

    def __getitem__(self, index):
        
        if self.balance:
            labs = []
            imgs = []
            img_paths = []
            for i in range(self.class_num):
                safe_idx = index % len(self.data[i])
                img_path, lab = self.data[i][safe_idx]
                img = self.load_sample(img_path)
                labs.append(lab)
                imgs.append(img)
                img_paths.append(img_path)
                
            return torch.cat([imgs[i].unsqueeze(0) for i in range(self.class_num)]),\
                torch.tensor(labs, dtype=torch.long), img_paths
        else:
            img_path, lab = self.data[0][index]
            img = self.load_sample(img_path)
            lab = torch.tensor(lab, dtype=torch.long)
            
            return img, lab, img_path

    def load_sample(self, img_path):
        
        img = Image.open(img_path).convert('RGB')
        if img.size[0]!=img.size[1]:
            img = transforms.CenterCrop(size=self.config.crop_size)(img)
        if self.resize_size is not None:
            img = img.resize(self.resize_size)
        if self.second_resize_size is not None:
            img = img.resize(self.second_resize_size)
        
        img = self.norm_transform(img)    

        return img


class ImageMultiCropDataset(ImageDataset):
    def __init__(self, annotations, config, opt, balance=False):
        super(ImageMultiCropDataset, self).__init__(annotations, config, opt, balance)
        
        self.multi_size = config.multi_size
        crop_transforms = []
        for s in self.multi_size:
            RandomCrop = transforms.RandomCrop(size=s)
            crop_transforms.append(RandomCrop)
            self.multicroptransform = MultiCropTransform(crop_transforms)
            
    def __getitem__(self, index):
        
        if self.balance:
            labs = []
            imgs = []
            crops = []
            img_paths = []
            for i in range(self.class_num):
                safe_idx = index % len(self.data[i])
                img_path = self.data[i][safe_idx][0]
                img, crop = self.load_sample(img_path)
                lab = self.data[i][safe_idx][1]
                labs.append(lab)
                imgs.append(img)
                crops.append(crop)
                img_paths.append(img_path)
            crops = [torch.cat([crops[c][size].unsqueeze(0) for c in range(self.class_num)])
                for size in range(len(self.multi_size))]

            return torch.cat([imgs[i].unsqueeze(0) for i in range(self.class_num)]),\
                crops, torch.tensor(labs, dtype=torch.long), img_paths
        else:
            img_path, lab = self.data[0][index]
            lab = torch.tensor(lab, dtype=torch.long)
            img, crops = self.load_sample(img_path)

            return img, crops, lab, img_path

    def load_sample(self, img_path):
        img = Image.open(img_path).convert('RGB')
        if img.size[0]!=img.size[1]:
            img = transforms.CenterCrop(size=self.config.crop_size)(img)

        if self.resize_size is not None:
            img = img.resize(self.resize_size)
        if self.second_resize_size is not None:
            img = img.resize(self.second_resize_size)
            
        crops = self.multicroptransform(img)
        img = self.norm_transform(img)
        crops = [self.norm_transform(crop) for crop in crops]

        return img, crops

class ImageTransformationDataset(ImageDataset):
    def __init__(self, annotations, config, opt, balance=False):
        super(ImageTransformationDataset, self).__init__(annotations, config, opt, balance)
    
        self.data = annotations
        self.pretrain_transforms = get_transforms(config.crop_size)
        self.class_num = self.pretrain_transforms.class_num
        crop_transforms = []
        self.multi_size = config.multi_size
        for s in self.multi_size:
            RandomCrop = transforms.RandomCrop(size=s)
            crop_transforms.append(RandomCrop)
            self.multicroptransform = MultiCropTransform(crop_transforms)
            
    def __len__(self):

        return len(self.data)
    
    def __getitem__(self, index):
        
        img_path = self.data[index]
        img = Image.open(img_path).convert('RGB')
        img = transforms.RandomCrop(size=self.config.crop_size)(img)
        
        select_id=random.randint(0,self.class_num-1)
        pretrain_transform=self.pretrain_transforms.select_tranform(select_id)
        transformed = pretrain_transform(image=np.asarray(img))
        img = Image.fromarray(transformed["image"])

        if self.resize_size is not None:
            img = img.resize(self.resize_size)

        crops = self.multicroptransform(img)
        img = self.norm_transform(img)
        crops = [self.norm_transform(crop) for crop in crops]
        lab = torch.tensor(select_id, dtype=torch.long)
    
        return img, crops, lab, img_path    

class BaseData(object):
    def __init__(self, train_data_path, val_data_path, config, opt):

        train_set = ImageDataset(read_annotations(train_data_path), config, opt, balance=True)
        train_loader = DataLoader(
            dataset=train_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )
        
        val_set = ImageDataset(read_annotations(val_data_path), config, opt, balance=False)
        val_loader = DataLoader(
            dataset=val_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

        print('train: {}, val: {}'.format(len(train_set),len(val_set)))


class SupConData(object):
    def __init__(self, train_data_path, val_data_path, config, opt):
        
        train_set = ImageMultiCropDataset(read_annotations(train_data_path), config, opt, balance=True)
        train_loader = DataLoader(
            dataset=train_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )

        val_set = ImageMultiCropDataset(read_annotations(val_data_path), config, opt, balance=False)
        val_loader = DataLoader(
            dataset=val_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

        self.train_loader = train_loader
        self.val_loader = val_loader

        print('train: {}, val: {}'.format(len(train_set),len(val_set)))


class TranformData(object):
    def __init__(self, train_data_path, val_data_path, config, opt):

        
        train_set = ImageTransformationDataset(read_annotations(train_data_path), config, opt)
        train_loader = DataLoader(
            dataset=train_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=True,
            drop_last=True
        )
        self.train_loader = train_loader
        self.class_num = train_set.class_num

        val_set = ImageTransformationDataset(read_annotations(val_data_path), config, opt)
        val_loader = DataLoader(
            dataset=val_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )
        self.val_loader = val_loader
        
        print('train: {}, val: {}'.format(len(train_set),len(val_set)))
        

