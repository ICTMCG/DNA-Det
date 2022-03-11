import os
import argparse
import numpy as np
import cv2

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
 
from data.dataset import ImageDataset
from models.models import Simple_CNN
from utils.logger import Progbar
from utils.common import read_annotations, collate_fn, load_config
from visualization.grad_cam import GradCAM, GradCamPlusPlus
from visualization.guided_back_propagation import GuidedBackPropagation

def get_last_conv_name(net):
    """
    get the name of last convolution layer
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    print('layer_name',layer_name)
    return layer_name

def gen_cam(image, mask):
    """
    generate CAM map
    :param image: [H,W,C], original image
    :param mask: [H,W], range: 0~1
    :return: tuple(cam,heatmap)
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  # gbr to rgb
    cam = heatmap + np.float32(image)
    return norm_image(cam), (heatmap * 255).astype(np.uint8)

def norm_image(image):
    """
    image normalization
    :param image: [H,W,C]
    :return:
    """
    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def gen_gb(grad):
    """
    guided back propagation
    :param grad: tensor,[3,H,W]
    :return:
    """
    grad = grad.data.cpu().numpy()
    gb = np.transpose(grad, (1, 2, 0))
    return gb

def save_image(pred, label, image_dicts, dir_name, prefix, draw_dir):
    save_dir=os.path.join(draw_dir, dir_name)
    os.makedirs(save_dir,exist_ok=True)
    for key, image in image_dicts.items():
        cv2.imwrite(os.path.join(save_dir,'{}_{}_{}_{}.png'.format(prefix, key, label, pred)), image)

def parse_args():
    parser = argparse.ArgumentParser(description='draw gradcam')
    parser.add_argument('--draw_data_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, help='model_path', required=True)
    parser.add_argument('--config_name', type=str, help='model configuration file')
    parser.add_argument('--device', default='cuda:1', type=str, help='cuda:n or cpu')
    parser.add_argument('--layer_name', type=str, default=None)    
    args = parser.parse_args()
    return args

def main():
    opt = parse_args()
    config = load_config('configs.{}'.format(opt.config_name))
    device = torch.device(opt.device)
    
    print('load model from %s', opt.model_path)
    model_path = opt.model_path
    model = torch.load(model_path, map_location='cpu')    
    netE = Simple_CNN(class_num=config.class_num)
    netE.load_state_dict(model)
    netE = netE.to(device)
    
    draw_set = ImageDataset(read_annotations(opt.draw_data_path), config, opt)
    draw_loader = DataLoader(
        dataset=draw_set,
        num_workers=config.num_workers,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn
    )

    if opt.layer_name is None:
        layer_name = get_last_conv_name(netE)
    else:
        layer_name = opt.layer_name
    
    draw_dir = os.path.join(os.path.splitext(opt.draw_data_path)[0], "gradcam", layer_name)
    os.makedirs(draw_dir, exist_ok=True)
    print('draw dir: %s' % draw_dir)
    
    grad_cam = GradCAM(netE, layer_name)
    grad_cam_plus_plus = GradCamPlusPlus(netE, layer_name)
    gbp = GuidedBackPropagation(netE)
    
    progbar = Progbar(len(draw_set))
    for _, batch in enumerate(draw_loader):
        inputs, labels, img_paths = batch 

        img_path = img_paths[0]
        img = inputs[0]
        label = labels[0]
        inputs = inputs.reshape((-1, 3, inputs.size(-2), inputs.size(-1)))
        inputs = inputs.to(device)

        img = ToPILImage()(img * 0.5 + 0.5)
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR) / 255.0
        image_dict = {}
        image_dict['inputs'] = img * 255
        
        inputs = inputs.requires_grad_(True)        
        # Grad-CAM
        mask, pred = grad_cam(inputs, None, config.resize_size)
        image_dict['cam'], image_dict['heatmap'] = gen_cam(img, mask)
        image_dict['mask'] = np.uint8(255 * mask)
        grad_cam.remove_handlers()

        """
        # Uncomment to draw heatmaps by other methods
        # Grad-CAM++
        mask_plus_plus = grad_cam_plus_plus(inputs, None, config.resize_size)
        image_dict['campp'], image_dict['heatmappp'] = gen_cam(img, mask_plus_plus)
        grad_cam_plus_plus.remove_handlers()

        # GuidedBackPropagation
        inputs.grad.zero_()  
        grad = gbp(inputs)
        gb = gen_gb(grad)
        image_dict['gb'] = norm_image(gb)

        # Guided Grad-CAM
        cam_gb = gb * mask[..., np.newaxis]
        image_dict['cam_gb'] = norm_image(cam_gb)
        """
        
        # Save results
        save_image(pred, label, image_dict, img_path.split('/')[-3], os.path.splitext(os.path.basename(img_path))[0], draw_dir)
        
        progbar.add(1)

if __name__ == '__main__':
    main() 
