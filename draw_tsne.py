import os
import argparse
import numpy as np
from random import shuffle
from matplotlib import pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import ImageDataset
from models.models import Simple_CNN
from utils.common import load_config, read_annotations, collate_fn
from utils.logger import Progbar

def tsne_vis(features, labels, draw_dir, opt):
    
    embedding_path = os.path.join(draw_dir, '{}_embedding.npy'.format(opt.save_name))
    img_path = os.path.join(draw_dir,  '{}_tsne.png'.format(opt.save_name))
    print('tsne save path: %s' % img_path)
    
    if opt.do_fit or not os.path.exists(embedding_path):
        print(f">>> t-SNE fitting")
        tsne_model = TSNE(n_jobs=4, perplexity=opt.perplexity)
        embeddings = tsne_model.fit_transform(features)
        print(f"<<< fitting over")
        np.save(embedding_path, embeddings)        
    else:
        embeddings=np.load(embedding_path)

    index = [i for i in range(len(embeddings))]
    shuffle(index)
    embeddings = [embeddings[index[i]] for i in range(len(index))]
    labels = [labels[index[i]] for i in range(len(index))]
    embeddings = np.array(embeddings)
    print('embedding shape:', embeddings.shape)

    print(f">>> draw image")
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    plt.figure(figsize=(8,8))
    plt.rcParams['figure.dpi'] = 1000
    colors=['black','blue','red','lime','cyan']
    num_classes = len(set(labels))
    for i in range(num_classes):
        color = colors[i]
        class_index = [j for j,v in enumerate(labels) if v == i]
        plt.scatter(vis_x[class_index], vis_y[class_index], c=color)

    plt.xticks([])
    plt.yticks([])
    plt.legend(opt.legend)
    plt.show()
    plt.savefig(img_path)
    print(f"<<<save image")


def extract_feature(model, draw_loader, device):
    
    pool = nn.AdaptiveAvgPool2d(1)
    features = None
    model.eval()
    progbar = Progbar(len(draw_loader), stateful_metrics=['run-type'])
    with torch.no_grad():
        for _, batch in enumerate(draw_loader):
            input_img_batch, label_batch, _ = batch 
            input_img = input_img_batch.reshape((-1, 3, input_img_batch.size(-2), input_img_batch.size(-1))).to(device)
            label = label_batch.reshape((-1)).to(device)

            _, embedding = model(input_img)
            feature = pool(embedding)
            feature = feature.view(feature.shape[0], -1)

            if features is None:
                features=feature.cpu().numpy()
                gt_labels = label
            else:
                gt_labels = torch.cat([gt_labels, label])
                features=np.vstack((features, feature.cpu().numpy()))
                
            progbar.add(1, values=[('run-type', 'extract feature')])
    
    gt_labels = gt_labels.cpu().numpy()
    
    return features, gt_labels

def parse_args():
    parser = argparse.ArgumentParser(description='draw tsne')
    parser.add_argument('--draw_data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, help='model_path', required=True)
    parser.add_argument('--config_name', type=str, help='model configuration file')
    parser.add_argument('--device', default='cuda:1', type=str, help='cuda:n or cpu')
    parser.add_argument('--do_extract', action='store_true', default=False, help='whether to extract features')
    parser.add_argument('--do_fit', action='store_true', default=False, help='whether to fit tsne model')
    parser.add_argument('--save_name', default='cross_all', type=str)
    parser.add_argument('--legend', nargs='+', help='legend')
    parser.add_argument('--num', default=4000, type=int)
    parser.add_argument('--perplexity',default=15,type=int)
    args = parser.parse_args()
    return args


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
    
    draw_dir = os.path.join(os.path.splitext(opt.draw_data_path)[0], "tsne")
    os.makedirs(draw_dir, exist_ok=True)
    feature_path = os.path.join(draw_dir,  '{}_features.npy'.format(opt.save_name))
    label_path = os.path.join(draw_dir,  '{}_labels.npy'.format(opt.save_name))
    print('draw dir: %s' % draw_dir)
    
    annotations = read_annotations(opt.draw_data_path)
    annotations = annotations[:opt.num] if opt.num is not None else annotations
    draw_set = ImageDataset(annotations, config, opt)
    draw_loader = DataLoader(
        dataset=draw_set,
        num_workers=config.num_workers,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn
    )
    
    if opt.do_extract or not os.path.exists(feature_path):
        features, gt_labels = extract_feature(netE, draw_loader, device)
        np.save(feature_path, features)
        np.save(label_path, gt_labels)
    else:
        features = np.load(feature_path)
        gt_labels = np.load(label_path)
        
    print('labels:', gt_labels.shape, 'features:', features.shape)
    tsne_vis(features, gt_labels, draw_dir, opt)



