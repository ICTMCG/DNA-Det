# For multiple cross setup evaluation
celeba_cross_dataset = ['./dataset/real/lsun/bedroom',
                        './dataset/GANs/ProGAN/multiseed_lsun/lsun_bedroom_200k_png_seed_v0',
                        './dataset/GANs/MMDGAN/lsun_bedroom_200k_png',
                        './dataset/GANs/SNGAN/lsun_bedroom_200k_png',
                        './dataset/GANs/InfoMax-GAN/lsun_bedroom_200k_png']

lsun_cross_dataset = ['./dataset/real/celeba/celeba_align_png_cropped',
                    './dataset/GANs/ProGAN/multiseed_celeba/celeba_align_png_cropped_seed_v0',
                    './dataset/GANs/MMDGAN/celeba_align_png_cropped',
                    './dataset/GANs/SNGAN/celeba_align_png_cropped',
                    './dataset/GANs/InfoMax-GAN/celeba_align_png_cropped']

celeba_cross_seed=[['./dataset/GANs/ProGAN/multiseed_celeba/celeba_align_png_cropped_seed_v1',
                   './dataset/GANs/ProGAN/multiseed_celeba/celeba_align_png_cropped_seed_v2',
                   './dataset/GANs/ProGAN/multiseed_celeba/celeba_align_png_cropped_seed_v3',
                   './dataset/GANs/ProGAN/multiseed_celeba/celeba_align_png_cropped_seed_v4',
                   './dataset/GANs/ProGAN/multiseed_celeba/celeba_align_png_cropped_seed_v5',
                   './dataset/GANs/ProGAN/multiseed_celeba/celeba_align_png_cropped_seed_v6',
                   './dataset/GANs/ProGAN/multiseed_celeba/celeba_align_png_cropped_seed_v7',
                   './dataset/GANs/ProGAN/multiseed_celeba/celeba_align_png_cropped_seed_v8',
                   './dataset/GANs/ProGAN/multiseed_celeba/celeba_align_png_cropped_seed_v9']]

lsun_cross_seed = [['./dataset/GANs/ProGAN/multiseed_lsun/lsun_bedroom_200k_png_seed_v1',
                   './dataset/GANs/ProGAN/multiseed_lsun/lsun_bedroom_200k_png_seed_v2',
                   './dataset/GANs/ProGAN/multiseed_lsun/lsun_bedroom_200k_png_seed_v3',
                   './dataset/GANs/ProGAN/multiseed_lsun/lsun_bedroom_200k_png_seed_v4',
                   './dataset/GANs/ProGAN/multiseed_lsun/lsun_bedroom_200k_png_seed_v5',
                   './dataset/GANs/ProGAN/multiseed_lsun/lsun_bedroom_200k_png_seed_v6',
                   './dataset/GANs/ProGAN/multiseed_lsun/lsun_bedroom_200k_png_seed_v7',
                   './dataset/GANs/ProGAN/multiseed_lsun/lsun_bedroom_200k_png_seed_v8',
                   './dataset/GANs/ProGAN/multiseed_lsun/lsun_bedroom_200k_png_seed_v9']]

celeba_cross_finetune=['./dataset/GANs/ProGAN/ft_old_crop',
                     './dataset/GANs/MMDGAN/ft_old_crop',
                     './dataset/GANs/SNGAN/ft_old_crop',
                     './dataset/GANs/InfoMax-GAN/ft_old_crop']

lsun_cross_finetune=['./dataset/GANs/ProGAN/ft_sofa_crop',
                     './dataset/GANs/MMDGAN/ft_sofa_crop',
                     './dataset/GANs/SNGAN/ft_sofa_crop',
                     './dataset/GANs/InfoMax-GAN/ft_sofa_crop']

celeba_cross_loss=['./dataset/GANs/CramerGAN/celeba_align_png_cropped',
                 './dataset/GANs/SSGAN/celeba_align_png_cropped']

lsun_cross_loss=['./dataset/GANs/CramerGAN/lsun_bedroom_200k_png',
                 './dataset/GANs/SSGAN/lsun_bedroom_200k_png']

celeba_closed_set = lsun_cross_dataset
lsun_closed_set = celeba_cross_dataset

# For in the wild evaluation
# 128 resolution
Real_128=['./dataset/real/celeba/celeba_align_png_cropped',
'./dataset/real/lsun/bedroom']
ProGAN_128=['./dataset/GANs/ProGAN/multiseed_celeba/celeba_align_png_cropped_seed_v0',
'./dataset/GANs/ProGAN/multiseed_lsun/lsun_bedroom_200k_png_seed_v0']
MMDGAN_128=['./dataset/GANs/MMDGAN/celeba_align_png_cropped',
'./dataset/GANs/MMDGAN/lsun_bedroom_200k_png']
SNGAN_128=['./dataset/GANs/SNGAN/celeba_align_png_cropped',
'./dataset/GANs/SNGAN/lsun_bedroom_200k_png']
InfoMaxGAN_128=['./dataset/GANs/InfoMax-GAN/celeba_align_png_cropped',
'./dataset/GANs/InfoMax-GAN/lsun_bedroom_200k_png']

Real_256=['./dataset/GANs/CNNDetect/train/cat/0_real',
'./dataset/GANs/CNNDetect/train/airplane/0_real',
'./dataset/GANs/CNNDetect/train/boat/0_real',
'./dataset/GANs/CNNDetect/train/horse/0_real',
'./dataset/GANs/CNNDetect/train/sofa/0_real',
'./dataset/GANs/CNNDetect/train/cow/0_real',
'./dataset/GANs/CNNDetect/train/dog/0_real',
'./dataset/GANs/CNNDetect/train/train/0_real',
'./dataset/GANs/CNNDetect/train/bicycle/0_real',
'./dataset/GANs/CNNDetect/train/bottle/0_real',
'./dataset/GANs/CNNDetect/train/diningtable/0_real',
'./dataset/GANs/CNNDetect/train/motorbike/0_real',
'./dataset/GANs/CNNDetect/train/sheep/0_real',
'./dataset/GANs/CNNDetect/train/tvmonitor/0_real',
'./dataset/GANs/CNNDetect/train/bird/0_real',
'./dataset/GANs/CNNDetect/train/bus/0_real',
'./dataset/GANs/CNNDetect/train/chair/0_real',
'./dataset/GANs/CNNDetect/train/person/0_real',
'./dataset/GANs/CNNDetect/train/pottedplant/0_real',
'./dataset/GANs/CNNDetect/train/car/0_real']

ProGAN_256= ['./dataset/GANs/CNNDetect/train/cat/1_fake',
'./dataset/GANs/CNNDetect/train/airplane/1_fake',
'./dataset/GANs/CNNDetect/train/boat/1_fake',
'./dataset/GANs/CNNDetect/train/horse/1_fake',
'./dataset/GANs/CNNDetect/train/sofa/1_fake',
'./dataset/GANs/CNNDetect/train/cow/1_fake',
'./dataset/GANs/CNNDetect/train/dog/1_fake',
'./dataset/GANs/CNNDetect/train/train/1_fake',
'./dataset/GANs/CNNDetect/train/bicycle/1_fake',
'./dataset/GANs/CNNDetect/train/bottle/1_fake',
'./dataset/GANs/CNNDetect/train/diningtable/1_fake',
'./dataset/GANs/CNNDetect/train/motorbike/1_fake',
'./dataset/GANs/CNNDetect/train/sheep/1_fake',
'./dataset/GANs/CNNDetect/train/tvmonitor/1_fake',
'./dataset/GANs/CNNDetect/train/bird/1_fake',
'./dataset/GANs/CNNDetect/train/bus/1_fake',
'./dataset/GANs/CNNDetect/train/chair/1_fake',
'./dataset/GANs/CNNDetect/train/person/1_fake',
'./dataset/GANs/CNNDetect/train/pottedplant/1_fake',
'./dataset/GANs/CNNDetect/train/car/1_fake']

# 256 resolution

StackGAN2_256=['./dataset/GANs/StackGAN2/cat',
'./dataset/GANs/StackGAN2/church',
'./dataset/GANs/StackGAN2/bird',
'./dataset/GANs/StackGAN2/bedroom',
'./dataset/GANs/StackGAN2/dog']

CycleGAN_256=['./dataset/GANs/cyclegan/winter',
'./dataset/GANs/cyclegan/orange',
'./dataset/GANs/cyclegan/apple',
'./dataset/GANs/cyclegan/horse',
'./dataset/GANs/cyclegan/summer',
'./dataset/GANs/cyclegan/zebra']

StyleGAN2_256=['./dataset/GANs/styleGAN2/cat',
'./dataset/GANs/styleGAN2/church',
'./dataset/GANs/styleGAN2/horse']

# 1024 resolution
Real_1024=['./dataset/real/Flickr-Faces-HQ_10K',
'./dataset/real/celebA-HQ_10K']

StyleGAN_1024=['./dataset/GANs/styleGAN/ffhq',
'./dataset/GANs/styleGAN/celebahq',
'./dataset/GANs/styleGAN/race_yellow',
'./dataset/GANs/styleGAN/seeprettyface_models',
'./dataset/GANs/styleGAN/seeprettyface_asian_stars',
'./dataset/GANs/styleGAN/age_kids',
'./dataset/GANs/styleGAN/glasses',
'./dataset/GANs/styleGAN/age_elders',
'./dataset/GANs/styleGAN/age_adults',
'./dataset/GANs/styleGAN/male',
'./dataset/GANs/styleGAN/female',
'./dataset/GANs/styleGAN/smile']

StyleGAN2_1024=['./dataset/GANs/styleGAN2/stylegan2-ffhq-config-f',
'./dataset/GANs/styleGAN2/generated_yellow-stylegan2',
'./dataset/GANs/styleGAN2/generated_wanghong-stylegan2',
'./dataset/GANs/styleGAN2/generated_asian_stars-stylegan2',
'./dataset/GANs/styleGAN2/generated_babies-stylegan2']

Real_all = Real_128 + Real_256 + Real_1024
data_sequences = [Real_all,ProGAN_128,MMDGAN_128,SNGAN_128,InfoMaxGAN_128,StackGAN2_256,ProGAN_256,CycleGAN_256,StyleGAN2_256,StyleGAN_1024,StyleGAN2_1024]

