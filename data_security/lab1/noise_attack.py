import numpy as np
import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fgsm_attack import Accuracy
from torchvision.models import resnet50, ResNet50_Weights


class AlbumentationsTransform:
    def __init__(self, albumentations_transform):
        self.albumentations_transform = albumentations_transform
    
    def __call__(self, img):
        img = np.array(img)
        augmented = self.albumentations_transform(image=img)['image']
        return augmented
    
    def __str__(self):
        return self.albumentations_transform.__str__()


if __name__ == '__main__':
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMG_SIZE = 224
    
    transforms_test = [
        transforms.Compose([AlbumentationsTransform(A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),]
        ))]),
        transforms.Compose([AlbumentationsTransform(A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Rotate(limit=40, p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(), ]
        ))]),
        transforms.Compose([AlbumentationsTransform(A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.ToGray(p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(), ]
        ))]),
        transforms.Compose([AlbumentationsTransform(A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(), ]
        ))]),
        transforms.Compose([AlbumentationsTransform(A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=20, p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(), ]
        ))]),
        transforms.Compose([AlbumentationsTransform(A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Blur(blur_limit=13, p=1.0),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(), ]
        ))]),
        transforms.Compose([AlbumentationsTransform(A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.RandomSunFlare(),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(), ]
        ))]),
    ]
    
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    model = model.to('cuda')
    
    for transform in transforms_test:
        acc = Accuracy()
        testset = torchvision.datasets.ImageFolder(
            root='./data/imagenet10/val',
            transform=transform
        )
        dataloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)
        
        with torch.no_grad():
            for data, target in tqdm.tqdm(dataloader):
                data = data.to('cuda')
                target = target.to('cuda')
                output = model(data)
                acc.update(output, target)
        
        print(f'Accuracy {acc.compute():.4f}, transforms: {transform}')