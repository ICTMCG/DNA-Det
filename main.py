import os
import sys
import json
import argparse 
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from trainer import Trainer
from data.dataset import SupConData, TranformData, ImageMultiCropDataset, ImageTransformationDataset
from models.models import Simple_CNN, SupConNet
from utils.loss import SupConLoss, AutomaticWeightedLoss
from utils.common import load_config, get_train_paths, setup_seed, read_annotations
from utils.logger import create_logger

def parse_args():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--data_path', type=str, help='path to datasets')
    parser.add_argument('--train_collection', type=str, help='training collection', required=True)
    parser.add_argument('--val_collection', type=str, help='validation collection', required=True)
    parser.add_argument('--test_data_path', type=str, help='test data path', default='')
    parser.add_argument('--config_name', type=str, help='model configuration file')
    parser.add_argument('--run_id', type=str, default='0', help='run_id')
    parser.add_argument('--resume', action='store_true', default=False, help='resume from an existing checkpoint')
    parser.add_argument('--pretrain', action='store_true', default=False, help='whether to pretrain on image transformations')
    parser.add_argument('--pretrain_model_path', type=str, help='pretrain_model_path', default=None)
    parser.add_argument('--device', default='cuda:0', type=str, help='cuda:n or cpu')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # load configs
    opt = parse_args()
    config = load_config('configs.{}'.format(opt.config_name))

    # setup random seed
    setup_seed(config.seed)
    
    # setup gpu device
    torch.cuda.set_device(int(opt.device.split(':')[1]))
    device = torch.device(opt.device)

    # setup data path
    model_dir, train_data_path, val_data_path = get_train_paths(opt)
    
    model_path = os.path.join(model_dir, "model.pth")
    writer = SummaryWriter(logdir=model_dir)
    logger = create_logger(model_dir)    
    logger.info('model dir: %s' % model_dir)

    # save configs
    options_file = os.path.join(model_dir, 'options.json')
    with open(options_file, 'w') as fp:
        json.dump(vars(opt), fp, indent=4)
    logger.info('options: %s',opt)

    # setup data
    if opt.pretrain:
        Data = TranformData(train_data_path, val_data_path, config, opt)
        class_num = Data.class_num
    else:
        Data = SupConData(train_data_path, val_data_path, config, opt)
        class_num = config.class_num
    train_loader, val_loader = Data.train_loader, Data.val_loader

    if os.path.exists(opt.test_data_path):
        if opt.pretrain:
            test_set = ImageTransformationDataset(read_annotations(val_data_path), config, opt)
        else:
            test_set = ImageMultiCropDataset(read_annotations(opt.test_data_path), config, opt, balance=False)
        test_loader = DataLoader(
            dataset=test_set,
            num_workers=config.num_workers,
            batch_size=config.batch_size,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
        )

    # setup network
    netE = Simple_CNN(class_num, opt.pretrain)
    
    # load from a pretrained or existing model
    if opt.pretrain_model_path is not None:
        if os.path.exists(opt.pretrain_model_path):
            logger.info('resume from pretrained model %s' % opt.pretrain_model_path)
            netE_dict = netE.state_dict()
            pretrained_dict = torch.load(opt.pretrain_model_path, map_location='cpu')
            pretrained_dict = {k:v for k,v in pretrained_dict.items() \
                            if k in netE_dict.keys() and pretrained_dict[k].shape==netE_dict[k].shape}
            netE_dict.update(pretrained_dict)
            netE.load_state_dict(netE_dict)
    elif opt.resume and os.path.exists(model_path):
        logger.info('resume from existing model %s' % model_path)
        pretrained_dict = torch.load(opt.model_path, map_location='cpu')
        netE.load_state_dict(pretrained_dict)

    # model to device
    model = SupConNet(netE)
    model = model.to(device)

    # setup optimizer, scheduler, criterion
    awl = AutomaticWeightedLoss(2)
    awl = awl.to(device)
    optimizer = torch.optim.Adam([
        {'params':model.parameters(), 'lr':config.init_lr_E},
        {'params': awl.parameters(), 'lr': 0.01}
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    criterionCE = nn.CrossEntropyLoss()
    criterionCon = SupConLoss(temperature=config.temperature)

    # setup trainer
    Trainer = Trainer(model, awl, criterionCE, criterionCon,
                            optimizer, scheduler, train_loader,
                            device, config, opt, writer, logger)
    
    # begin to train
    logger.info("begin to train!")
    s_time = time.time()
    best_perf = 0
    for epoch in range(config.max_epochs):
        Trainer.train_epoch(epoch)
        val_perf = Trainer.predict_set(val_loader, run_type='val')
        
        if val_perf >= best_perf: 
            best_perf = val_perf
            no_impr_counter = 0
            torch.save(netE.state_dict(), os.path.join(model_dir, "model.pth"))
        else:
            no_impr_counter += 1
                
        logger.info('epoch %d -> metric %s, val: %.4f, best: %.4f' % 
                    (epoch, config.metric, val_perf, best_perf))
        
        if no_impr_counter > config.early_stop_bar:
            logger.info('Early stop')
            break
        
        if (epoch+1) % config.save_interval == 0:
            if os.path.exists(opt.test_data_path):
                test_perf = Trainer.predict_set(test_loader, run_type='test')
                torch.save(netE.state_dict(), os.path.join(model_dir, "model_epoch_{}_val_{}_test_{}.pth". \
                                                       format(epoch+1, round(val_perf,4), round(test_perf,4))))
            else:
                torch.save(netE.state_dict(), os.path.join(model_dir, "model_epoch_{}_val_{}.pth". \
                                                       format(epoch+1, round(val_perf,4)))) 
    
    time_span = time.time() - s_time
    logger.info("training done in {} minutes".format(time_span / 60.0))          