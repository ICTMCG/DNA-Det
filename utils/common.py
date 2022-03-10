import os
import random
import importlib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def load_config(config_path):
    module = importlib.import_module(config_path)
    return module.Config()

def read_annotations(data_path):
    with open(data_path,'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        elements = line.strip().split('\t')
        if len(elements) == 2:
            sample_path, label = elements
            label = int(label)
            data.append((sample_path, label))
        else:
            data.append(elements[0])
    return data

def get_train_paths(args):
    train_data_path = os.path.join(args.data_path, args.train_collection, "annotations", args.train_collection + ".txt")
    val_data_path = os.path.join(args.data_path, args.val_collection, "annotations", args.val_collection + ".txt")
    model_dir = os.path.join(args.data_path, args.train_collection, "models", args.val_collection, args.config_name,
                             "run_%s" % args.run_id)
    os.makedirs(model_dir,exist_ok=True)
    return [model_dir, train_data_path, val_data_path]

def evaluate_multiclass(gt_labels, pred_labels):
    acc = accuracy_score(gt_labels, pred_labels)
    f1 = f1_score(gt_labels, pred_labels, average='macro')
    recall = recall_score(gt_labels, pred_labels, average='macro')
    recall_per_class = recall_score(gt_labels, pred_labels, average=None)
    return {'recall_per_class':recall_per_class,'recall':recall,'f1':f1,'acc':acc}

