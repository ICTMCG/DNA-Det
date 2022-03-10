import os
import argparse 
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split

from dataset.data_paths import celeba_closed_set, lsun_closed_set, celeba_cross_seed, lsun_cross_seed, \
                    celeba_cross_loss, lsun_cross_loss, celeba_cross_finetune, lsun_cross_finetune, \
                        celeba_cross_dataset, lsun_cross_dataset

def createdirs(save_path,save_dirs):
    save_paths= [os.path.join(save_path,save_dirs[i],'annotations') for i in range(3)]
    [os.makedirs(save_paths[i]) for i in range(3) if not os.path.exists(save_paths[i])]
    train_list, val_list,test_list = [os.path.join(save_paths[i], save_dirs[i]+'.txt') for i in range(3)]
    print('train_collection='+save_dirs[0])
    print('val_collection='+save_dirs[1])
    print('test_data_path='+test_list)
    return train_list, val_list,test_list

def shuffle_data(data,label):
    index = [i for i in range(len(data))]
    shuffle(index)
    data = [data[index[i]] for i in range(len(index))]
    label = [label[index[i]] for i in range(len(index))]
    return data,label

def closed_data_split(data_sequences,save_path,save_dirs,num=None):
    file_all=[]
    label_all=[]
    for index, sub_path in enumerate(data_sequences):
        file_names=[]
        for root, _, files in os.walk(sub_path):
            for file in files:
                if file.endswith('jpg') or file.endswith('png'):
                    file_name = os.path.join(root, file)
                    file_names.append(file_name)
        if num is not None:
            file_names=file_names[:num]
        labels = [index for i in range(len(file_names))]
        file_all+=file_names
        label_all+=labels
        print(sub_path,len(file_names))

    train_X, test_X, train_y, test_y = train_test_split(file_all, label_all, test_size=0.2, random_state=1,stratify=label_all)
    test_X, val_X, test_y, val_y = train_test_split(test_X, test_y, test_size=0.5, random_state=1,stratify=test_y)

    print("train dataset: ", len(train_X), len(train_y))
    for i in range(len(set(label_all))):
        print(i, sum(np.array(train_y)==i))

    print("val dataset: ", len(val_X), len(val_y))
    for i in range(len(set(label_all))):
        print(i, sum(np.array(val_y)==i))

    print("test dataset: ", len(test_X), len(test_y))
    for i in range(len(set(label_all))):
        print(i, sum(np.array(test_y)==i))

    train_list, val_list, test_list = createdirs(save_path, save_dirs)

    with open(train_list, "w") as f:
        for i in range(len(train_X)):
            f.write(train_X[i] + "\t" + str(train_y[i]) + "\n")
    with open(val_list, "w") as f:
        for i in range(len(val_X)):
            f.write(val_X[i] + "\t" + str(val_y[i]) + "\n")
    with open(test_list, "w") as f:
        for i in range(len(test_X)):
            f.write(test_X[i] + "\t" + str(test_y[i]) + "\n")
            

def cross_data_prepare(data_sequences,save_path,save_dirs,save_name,label,num=None):
    test_X=[]
    test_y=[]
    for index, sub_paths in enumerate(data_sequences):
        file_names = []
        if not isinstance(sub_paths, list):
            sub_paths = [sub_paths]
        for sub_path in sub_paths:
            for root, _, files in os.walk(sub_path):
                for file in files:
                    if file.endswith('jpg') or file.endswith('png'):
                        file_name = os.path.join(root, file)
                        file_names.append(file_name)
                if num is not None:
                    file_names=file_names[:num]        
            labels = [label[index] for i in range(len(file_names))]
            test_X+=file_names
            test_y+=labels
            print(sub_path, len(file_names))

    test_X, test_y=shuffle_data(test_X,test_y)
    print("test dataset: ", len(test_X), len(test_y))
    for i in set(test_y):
        print(i, sum(np.array(test_y) == i))

    test_list = os.path.join(save_path, save_dirs[-1], 'annotations', save_name + '.txt')
    print(test_list)
    with open(test_list, "w") as f:
        for i in range(len(test_X)):
            f.write(test_X[i] + "\t" + str(test_y[i]) + "\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='prepare dataset')
    parser.add_argument('--mode', type=str, default='lsun', help='celeba/lsun')
    parser.add_argument('--save_path', type=str, default='./dataset_generate')
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)    
    
    if args.mode == 'celeba':
        closed_set, cross_seed, cross_loss, cross_ft, cross_dataset = celeba_closed_set, celeba_cross_seed, celeba_cross_loss, celeba_cross_finetune, celeba_cross_dataset
    elif args.mode == 'lsun':
        closed_set, cross_seed, cross_loss, cross_ft, cross_dataset = lsun_closed_set, lsun_cross_seed, lsun_cross_loss, lsun_cross_finetune, lsun_cross_dataset
        
    save_dirs = ['{}_{}'.format(args.mode, split) for split in ['train', 'val', 'test']]
    print('------prepare closed set data-----')
    closed_data_split(closed_set,args.save_path, save_dirs, num=20000)
    print('------prepare cross-seed data-----')
    cross_data_prepare(cross_seed, args.save_path, save_dirs, save_name='{}_test_cross_seed'.format(args.mode), label=[1], num=2000)
    print('------prepare cross-loss data-----')
    cross_data_prepare(cross_loss, args.save_path, save_dirs, save_name='{}_test_cross_loss'.format(args.mode), label=[2,4], num=2000)
    print('------prepare cross-finetune data-----')
    cross_data_prepare(cross_ft, args.save_path, save_dirs, save_name='{}_test_cross_finetune'.format(args.mode), label=[1,2,3,4], num=2000)
    print('------prepare cross-dataset data-----')
    cross_data_prepare(cross_dataset, args.save_path, save_dirs, save_name='{}_test_cross_dataset'.format(args.mode), label=[0,1,2,3,4], num=2000)
