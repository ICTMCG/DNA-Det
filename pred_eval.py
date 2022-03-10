import os
import argparse

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np

from models.models import Simple_CNN
from data.dataset import ImageDataset
from utils.common import load_config,evaluate_multiclass,read_annotations,collate_fn
from utils.logger import Progbar

def parse_args():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--test_data_paths', nargs='+', type=str, required=True)
    parser.add_argument('--model_path', type=str, help='model_path', required=True)
    parser.add_argument('--config_name', type=str, help='model configuration file')
    parser.add_argument('--device', default='cuda:7', type=str, help='cuda:n or cpu')
    parser.add_argument('--overwrite', action='store_true', default=False, help='whether to overwrite existing result')
    args = parser.parse_args()
    return args

def predict_set(model, dataloader, device):
    model.eval()
    progbar = Progbar(len(dataloader), stateful_metrics=['run-type'])
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_img_batch, label_batch, img_path = batch 
            input_img = input_img_batch.reshape((-1, 3, input_img_batch.size(-2), input_img_batch.size(-1))).to(device)
            label = label_batch.reshape((-1)).to(device)

            prob, _ = model(input_img)

            if i == 0:
                probs = prob
                gt_labels = label
                img_paths = img_path
            else:
                probs = torch.cat([probs, prob], dim=0)
                gt_labels = torch.cat([gt_labels, label])
                img_paths += img_path
            progbar.add(1, values=[('run-type', 'test')])

    gt_labels = gt_labels.cpu().numpy()
    probs = probs.cpu().numpy()
    pred_labels = np.argmax(probs,axis=1)

    return gt_labels,pred_labels,probs,img_paths

if __name__ == '__main__':
    opt = parse_args()
    config = load_config('configs.{}'.format(opt.config_name))
    device = torch.device(opt.device)
    
    print('load model from %s', opt.model_path)
    model_path = opt.model_path
    model = torch.load(model_path, map_location='cpu')    
    netE = Simple_CNN(class_num=config.class_num)
    netE.load_state_dict(model)
    netE = netE.to(device)

    for test_data_path in opt.test_data_paths:
        pred_dir = os.path.join(os.path.splitext(test_data_path)[0], "pred")
        os.makedirs(pred_dir,exist_ok=True)
        res_file = os.path.join(pred_dir, 'result.txt')
        print('test_data_path: %s' % test_data_path)
        print('result file: %s'% res_file)

        if os.path.exists(res_file) and not opt.overwrite:
            result_lines = open(res_file).readlines()
            gt_lines = open(test_data_path).readlines()
            result_lines.sort()
            gt_lines.sort()
            gt_labels=[]
            pred_labels=[]
            for idx, line in enumerate(gt_lines):
                pred_labels.append(int(result_lines[idx].strip('\t').split()[1]))
                gt_labels.append(int(line.strip('\t').split()[1]))
        else:
            test_set = ImageDataset(read_annotations(test_data_path), config, opt)
            test_loader = DataLoader(
                dataset=test_set,
                num_workers=config.num_workers,
                batch_size=config.batch_size,
                pin_memory=True,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn
            )
            gt_labels, pred_labels, scores, img_paths = predict_set(netE, test_loader, device)
            with open(res_file, 'w') as fw:
                fw.write('\n'.join(['{}\t{}'.format(img_paths[i], pred_labels[i]) for i in range(len(img_paths))]))

        result = evaluate_multiclass(gt_labels, pred_labels)
        cm = confusion_matrix(gt_labels, pred_labels, [i for i in range(config.class_num)])
        print('result',result)
        print('confusion matrix',cm)