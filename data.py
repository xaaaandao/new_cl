import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from config import DataConfig

class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class DataManager:
    def __init__(self, config: DataConfig):
        self.cfg = config
        self.mean = config.mean
        self.std = config.std

    def get_transforms(self, mode: str = 'train'):
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        
        if mode == 'train':
            return transforms.Compose([
                transforms.Resize((self.cfg.resize_size, self.cfg.resize_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.cfg.resize_size, self.cfg.resize_size)),
                transforms.ToTensor(),
                normalize,
            ])

    def get_loader(self, dataset_path: str, is_contrastive: bool = True):
        transform = self.get_transforms(mode='train')
        
        if is_contrastive:
            transform = TwoCropTransform(transform)

        dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True
        )
        return loader