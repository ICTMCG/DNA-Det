import time
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from utils.logger import Progbar, AverageMeter
from utils.common import evaluate_multiclass

class Trainer(): 
    def __init__(self, model, awl, criterionCE, criterionCon,
                optimizer, scheduler, train_loader,
                device, config, opt, writer, logger):

        self.model, self.awl = model, awl
        self.criterionCE, self.criterionCon = criterionCE, criterionCon 
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.opt = opt
        self.config = config
        self.device = device
        self.logger = logger
        self.writer = writer
        self.board_num = 0

    def train_epoch(self, epoch):
        progbar = Progbar(len(self.train_loader), stateful_metrics=['epoch'])
        batch_time = AverageMeter()
        end = time.time()
        self.model.train()
        for _, batch in enumerate(self.train_loader):
            self.board_num += 1
            _, crops_batch, label_batch, _ = batch
            crops = [crop_batch.reshape((-1, 3, crop_batch.size(-2), crop_batch.size(-1))).to(self.device) for crop_batch in crops_batch]
            labels = label_batch.reshape((-1)).to(self.device)
            self.optimizer.zero_grad()
            
            # predict on crops
            crops_result = [self.model(crop) for crop in crops]
            # classification probs on crops
            crops_prob = torch.cat([result[0] for result in crops_result], dim=0)
            # features on crops
            crops_feat = torch.cat([result[1].unsqueeze(1) for result in crops_result], dim=1)
            # labels for crops
            crops_label = torch.cat([labels]*len(self.config.multi_size), dim=0)
            # calculate classification loss
            loss_cls = self.criterionCE(crops_prob, crops_label)
            # calculate contrastive loss
            loss_contra = self.criterionCon(crops_feat, labels)
            # calculate total loss
            loss_total = self.awl(loss_cls, loss_contra)
            
            loss_total.backward()
            self.optimizer.step()
            self.scheduler.step()

            losses = {'loss_cls': loss_cls.item(), 'loss_contra': loss_contra.item()}
            for loss_key in losses.keys():
                self.writer.add_scalars(loss_key, {'loss_key': losses[loss_key]}, self.board_num)
            progbar.add(1, values=[('epoch', epoch)]+[(loss_key,losses[loss_key]) for loss_key in losses.keys()]+[('lr', self.scheduler.get_lr()[0])])

            batch_time.update(time.time() - end)
            end = time.time()


    def predict_set(self, dataloader, run_type='test'):
        self.model.eval()
        progbar = Progbar(len(dataloader), stateful_metrics=['run-type'])
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                input_img_batch, _, label_batch, _ = batch 
                input_img = input_img_batch.reshape((-1, 3, input_img_batch.size(-2), input_img_batch.size(-1))).to(self.device)
                label = label_batch.reshape((-1)).to(self.device)

                prob, _ = self.model(input_img)
                loss_cls = self.criterionCE(prob, label)

                if i == 0:
                    probs = prob
                    gt_labels = label
                else:
                    probs = torch.cat([probs, prob], dim=0)
                    gt_labels = torch.cat([gt_labels, label])
                progbar.add(1, values=[('run-type', run_type),('loss_cls',loss_cls.item())])

        gt_labels = gt_labels.cpu().numpy()
        probs = probs.cpu().numpy()
        pred_labels = np.argmax(probs,axis=1)

        results = evaluate_multiclass(gt_labels, pred_labels)
        CM = confusion_matrix(gt_labels, pred_labels)
        perf = round(results[self.config.metric], 4)
        self.logger.info('%s results: %s' % (run_type, str(results)))
        self.logger.info('%s confusion matrix: %s' % (run_type, str(CM)))

        return perf
    

