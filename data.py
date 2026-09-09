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


class GenusSpeciesImageFolder(datasets.ImageFolder):
    def __init__(self, root: str, transform=None, genus_separator: str = '+'):
        super().__init__(root=root, transform=transform)
        self.genus_separator = genus_separator

        # Extrai o gênero (primeiro token) de cada nome de classe/espécie
        self.species_to_genus_name = {
            cls_name: cls_name.split(genus_separator)[0]
            for cls_name in self.classes
        }

        # Vocabulário de gêneros únicos, ordenado para reprodutibilidade entre execuções
        unique_genus_names = sorted(set(self.species_to_genus_name.values()))
        self.genus_to_idx = {g: i for i, g in enumerate(unique_genus_names)}
        self.idx_to_genus_name = {i: g for g, i in self.genus_to_idx.items()}

        # Mapeamento direto idx_da_especie -> idx_do_genero (evita reprocessar strings a cada __getitem__)
        self.species_idx_to_genus_idx = {
            species_idx: self.genus_to_idx[self.species_to_genus_name[species_name]]
            for species_name, species_idx in self.class_to_idx.items()
        }

        self.num_species = len(self.classes)
        self.num_genus = len(unique_genus_names)

        logger.info(
            f"[GenusSpeciesImageFolder] {self.num_species} espécies mapeadas para "
            f"{self.num_genus} gêneros a partir de '{root}'."
        )

    def __getitem__(self, index: int):
        path, species_label = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        genus_label = self.species_idx_to_genus_idx[species_label]

        return sample, genus_label, species_label


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
            transform_list.append(
                RandomCropWithValidation(size=size, min_info_ratio=self.train_config.crop_validation_info_ratio))
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

        dataset = GenusSpeciesImageFolder(root=dataset_path, transform=transform)

        # Guarda referência para permitir consultar num_genus/num_species depois
        # (ex: para instanciar LinearClassifier de avaliação em cada nível)
        self.last_dataset = dataset

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