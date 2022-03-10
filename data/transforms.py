import random
import albumentations as A

class get_transforms():
    random.seed(0)
    def __init__(self,img_size):
        self.augs = []
        
        qualitys=[i for i in range(30,100,3)]
        for quality in qualitys:
            self.augs.append(A.JpegCompression(quality_lower=quality, quality_upper=quality,always_apply=True))
        # print('qualitys: %s' % qualitys)
        
        double_qualitys=[]
        qualitys2 = [39, 99, 36, 93, 60, 75, 81, 39, 63, 69, 60, 87, 90, 48, 81, 75, 72, 78, 54, 33, 81, 30, 36, 99]
        for idx, quality1 in enumerate(qualitys):
            quality2 = qualitys2[idx]
            doublecom = A.Compose([A.JpegCompression(quality_lower=quality1, quality_upper=quality1,always_apply=True),
            A.JpegCompression(quality_lower=quality2, quality_upper=quality2,always_apply=True)])
            self.augs.append(doublecom)
            double_qualitys.append([quality1,quality2])
        # print('double_qualitys: %s' % double_qualitys)

        blur_kernels=[i for i in range(3,30,2)]
        # print('blur_kernels: %s' % blur_kernels)
        for blur_kernel in blur_kernels:
            self.augs.append(A.GaussianBlur(blur_limit=[blur_kernel,blur_kernel],always_apply=True))
        for blur_kernel in blur_kernels:
            self.augs.append(A.MedianBlur(blur_limit=[blur_kernel,blur_kernel],always_apply=True))

        resize_scales=[round(0.01*i,2) for i in range(15,95,5)]
        # print('resize_scales: %s' % resize_scales)
        for resize_type in range(5):
            for resize_scale in resize_scales:
                resize_op = A.Compose([A.Resize(int(resize_scale*img_size[0]),int(resize_scale*img_size[1]),interpolation=resize_type,always_apply=True),
            A.Resize(img_size[0],img_size[1],interpolation=resize_type,always_apply=True)])
                self.augs.append(resize_op)

        noise_limits=[i for i in range(10,50,3)]
        # print('noise_limits: %s' % noise_limits)
        for noise_limit in noise_limits:
            self.augs.append(A.GaussNoise(var_limit=[noise_limit,noise_limit],always_apply=True))
        
        # print('total transform num: %d' % len(self.augs))
        self.class_num = len(self.augs)

    def select_tranform(self,selected_T):
        selected_transform=self.augs[selected_T]
        return selected_transform

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform) -> None:
        self.transform=transform
    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class MultiCropTransform:
    """Create multi crops of the same image"""
    def __init__(self, transforms) -> None:
        self.transforms=transforms
    def __call__(self, x):
        return [T(x) for T in self.transforms]