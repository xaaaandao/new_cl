import os
import json
import torch
import logging
from crop_random_module import RandomCropWithValidation
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from config import DataConfig, TrainConfig
from typing import Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)

class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class DataManager:
    def __init__(self, config: DataConfig, train_dir: str, train_config: TrainConfig):
        self.cfg = config
        self.train_dir = train_dir
        self.train_config = train_config
        
        self.mean, self.std = self._get_mean_std()
        self.cfg.mean = self.mean
        self.cfg.std = self.std
        
        logger.info(f"Normalização definida -> Mean: {self.mean}, Std: {self.std}")
    
    def _get_mean_std(self) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
        stats_file = os.path.join(self.train_dir, 'dataset_stats.json')
        
        if os.path.exists(stats_file):
            logger.info(f"Carregando estatísticas de dataset em cache: {stats_file}")
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            return tuple(stats['mean']), tuple(stats['std'])
        
        logger.info("Calculando média e desvio padrão do dataset de TREINO...")
        return self._compute_and_save_stats(stats_file)
    
    def _compute_and_save_stats(self, save_path: str):
        pre_transform = transforms.Compose([
            transforms.Resize((self.cfg.resize_size, self.cfg.resize_size)),
            transforms.ToTensor(),
        ])
        
        dataset = datasets.ImageFolder(root=self.train_dir, transform=pre_transform)
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)
        
        mean = 0.
        std = 0.
        nb_samples = 0.
        
        for data, _ in tqdm(loader, desc="Calculando Stats"):
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples
        
        mean_list = mean.tolist()
        std_list = std.tolist()
        
        with open(save_path, 'w') as f:
            json.dump({'mean': mean_list, 'std': std_list}, f)
            
        return tuple(mean_list), tuple(std_list)
    
    def get_transforms(self, mode: str = 'train'):
        normalize = transforms.Normalize(mean=self.mean, std=self.std)
        size = self.cfg.resize_size
        
        transform_list = []

        if self.train_config.crop_with_validation:
            transform_list.append(RandomCropWithValidation(size=size, min_info_ratio=self.train_config.crop_validation_info_ratio))
        else:
            transform_list.append(transforms.Resize((size, size)))

        # 2. Augmentations (Apenas no treino)
        if mode == 'train':
            transform_list.append(transforms.RandomHorizontalFlip())
            transform_list.append(transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8))
            transform_list.append(transforms.RandomGrayscale(p=0.2))

        # 3. Conversão e Normalização (Sempre acontece)
        transform_list.append(transforms.ToTensor())
        transform_list.append(normalize)

        # Retorna a composição da lista
        return transforms.Compose(transform_list)
    
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

    def get_loader(self, dataset_path: str, is_contrastive: bool = True, mode: str = 'train'):
        transform = self.get_transforms(mode='train')
        
        if is_contrastive:
            transform = TwoCropTransform(transform)

        base_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
        dataset = GenusSpeciesDataset(base_dataset)
        batch_size = self.cfg.batch_size
        
        if mode != 'train':
            batch_size = 1
            
        loader = DataLoader(
            dataset,
            shuffle=(mode == 'train'),
            batch_size=batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=(mode == 'train')
        )
        
        return loader

class GenusSpeciesDataset(torch.utils.data.Dataset):
    """
    Wrapper que recebe um ImageFolder e retorna (image, species_label, genus_label).
    Assume que o nome da pasta segue o formato 'Genus+species' (ex: Peperomia+alata).
    """
    def __init__(self, dataset):
        self.dataset = dataset

        # Mapeia species_idx -> genus_name
        species_names = dataset.classes  # Lista ordenada dos nomes das pastas
        self.idx_to_genus = [name.split('+')[0] for name in species_names]

        # Cria labels numéricos para gênero
        unique_genera = sorted(list(set(self.idx_to_genus)))
        self.genus_to_idx = {genus: idx for idx, genus in enumerate(unique_genera)}

        # Pré-computa o genus_label para cada species_label
        self.species_to_genus_idx = [
            self.genus_to_idx[g] for g in self.idx_to_genus
        ]

        logger.info(f"Dataset: {len(dataset.classes)} espécies, {len(unique_genera)} gêneros")
        logger.info(f"Gêneros encontrados: {unique_genera}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, species_label = self.dataset[index]
        genus_label = self.species_to_genus_idx[species_label]
        return image, species_label, genus_label