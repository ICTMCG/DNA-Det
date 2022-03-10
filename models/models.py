import torch.nn as nn
import torch.nn.functional as F

class vgg_layer(nn.Module):
    def __init__(self, nin, nout):
        super(vgg_layer, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 3, 1, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2)
        )

    def forward(self, input):
        return self.main(input)

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nin, nout, 4, 2, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input):
        return self.main(input)

class Simple_CNN(nn.Module):
    def __init__(self, class_num, pretrain=False):
        super(Simple_CNN, self).__init__()
        nc = 3
        nf = 64
        self.main = nn.Sequential(
            dcgan_conv(nc, nf),
            vgg_layer(nf, nf),

            dcgan_conv(nf, nf * 2),
            vgg_layer(nf * 2, nf * 2),

            dcgan_conv(nf * 2, nf * 4),
            vgg_layer(nf * 4, nf * 4),

            dcgan_conv(nf * 4, nf * 8),
            vgg_layer(nf * 8, nf * 8),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classification_head = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(nf * 8, class_num, bias=True)
        )
        self.pretrain = pretrain

    def forward(self, input):
        embedding = self.main(input)
        feature = self.pool(embedding)
        feature = feature.view(feature.shape[0], -1)
        cls_out = self.classification_head(feature)
        if not self.pretrain:
            cls_out = F.softmax(cls_out)
        return cls_out, embedding

class SupConNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, backbone, head='mlp', dim_in=512, feat_dim=128):
        super(SupConNet, self).__init__()
        self.backbone=backbone
        if head=='linear':
            self.head=nn.Linear(dim_in, feat_dim)
        elif head=='mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )

    def forward(self, x):
        cls_out, embedding = self.backbone(x)
        feat = self.backbone.pool(embedding)
        feat = feat.view(feat.shape[0], -1)
        feat = F.normalize(self.head(feat), dim=1)
        return cls_out, feat












