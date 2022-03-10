import os
import glob
import random
import argparse
import numpy as np
from sklearn.model_selection import train_test_split 

from dataset.data_paths import data_sequences

def createdirs(save_path,save_dirs):
    save_paths= [os.path.join(save_path,save_dirs[i],'annotations') for i in range(3)]
    [os.makedirs(save_paths[i]) for i in range(3) if not os.path.exists(save_paths[i])]
    train_list,val_list,test_list = [os.path.join(save_paths[i], save_dirs[i]+'.txt') for i in range(3)]
    test_cross_list = os.path.join(save_paths[2], save_dirs[2]+'_cross_dataset.txt')
    print('train_collection='+save_dirs[0])
    print('val_collection='+save_dirs[1])
    print('test_data_path='+test_list)
    print('test_cross_data_path='+test_cross_list)
    return train_list,val_list,test_list,test_cross_list

def dataset_split(data_sequences,save_path,save_dirs,selected_train_list=None,num=None):
    train_file_all=[]
    train_label_all=[]
    test_file_all=[]
    test_label_all=[]
    for index, data_sequence in enumerate(data_sequences):
        selected_train=selected_train_list[index]
        for selected in selected_train:
            train_path=data_sequence[selected]
            train_files=[]
            for root, _, files in os.walk(train_path):
                for file in files:
                    if file.endswith('jpg') or file.endswith('png'):
                        file_name = os.path.join(root, file)
                        train_files.append(file_name)
            if num is not None:
                train_files=train_files[:num]
            print('close', train_path, len(train_files))
            train_labels = [index for i in range(len(train_files))]
            train_file_all+=train_files
            train_label_all+=train_labels

        for idx, selected in enumerate(selected_train):
            del data_sequence[selected-idx]
            num_cross_test = num//10
            
        for test_path in data_sequence:
            test_files=[]
            for root, _, files in os.walk(test_path):
                for file in files:
                    if file.endswith('jpg') or file.endswith('png'):
                        file_name = os.path.join(root, file)
                        test_files.append(file_name)
            if num is not None:
                test_files = test_files[:num_cross_test]
            print('cross', test_path, len(test_files))
            test_labels = [index for i in range(len(test_files))]
            test_file_all+=test_files
            test_label_all+=test_labels

    train_X, valtest_X, train_y, valtest_y = train_test_split(train_file_all, train_label_all, test_size=0.2, random_state=1,stratify=train_label_all)
    val_X, test_X, val_y, test_y = train_test_split(valtest_X, valtest_y, test_size=0.5, random_state=1,stratify=valtest_y)
    test_cross_X=test_file_all
    test_cross_y=test_label_all

    print("train dataset: ", len(train_X), len(train_y))
    for i in range(len(set(train_label_all))):
        print(i, sum(np.array(train_y)==i))

    print("val dataset: ", len(val_X), len(val_y))
    for i in range(len(set(train_label_all))):
        print(i, sum(np.array(val_y)==i))

    print("test dataset: ", len(test_X), len(test_y))
    for i in range(len(set(train_label_all))):
        print(i, sum(np.array(test_y)==i))

    print("cross test dataset: ", len(test_cross_X), len(test_cross_y))
    for i in range(len(set(test_cross_y))):
        print(i, sum(np.array(test_cross_y)==i))

    train_list,val_list,test_list,test_cross_list = createdirs(save_path, save_dirs)

    with open(train_list, "w") as f:
        for i in range(len(train_X)):
            f.write(train_X[i] + "\t" + str(train_y[i]) + "\n")
    with open(val_list, "w") as f:
        for i in range(len(val_X)):
            f.write(val_X[i] + "\t" + str(val_y[i]) + "\n")
    with open(test_list, "w") as f:
        for i in range(len(test_X)):
            f.write(test_X[i] + "\t" + str(test_y[i]) + "\n")
    with open(test_cross_list, "w") as f:
        for i in range(len(test_cross_X)):
            f.write(test_cross_X[i] + "\t" + str(test_cross_y[i]) + "\n")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='prepare dataset')
    parser.add_argument('--mode', type=str, default='in_the_wild')
    parser.add_argument('--save_path', type=str, default='./dataset/')
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)    
    selected_train_list = [[0,2,3,22],[0],[0],[0],[0],[0,1],[0,1],[0,1],[0,1],[0],[0]]
    save_dirs = ['{}_{}'.format(args.mode, split) for split in ['train', 'val', 'test']]
    
    dataset_split(data_sequences,args.save_path,save_dirs,selected_train_list,num=5000)



