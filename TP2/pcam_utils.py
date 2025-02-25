import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models
import torchvision.transforms.v2 as T
from datasets import DatasetDict
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

__all__ = [
    "get_model_and_optimizer",
    "get_dataloaders",
    "train_epoch",
    "validate",
]


def get_model_and_optimizer(lr: float = 1e-3, pretrained: bool = False) -> tuple[nn.Module, Optimizer]:
    """Crée un modèle et son optimiseur.

    Args:
        lr (float, optional): Taux d'apprentissage. Par défaut, 1e-3.
        pretrained (bool, optionnel): Si True, charge les poids pré-entraînés sur ImageNet. Si False, initialise les
            poids aléatoirement. Par défaut, False.

    Returns:
        tuple[nn.Module, Optimizer]: le modèle et son optimiseur.
    """
    seed_everything(42)
    torch.cuda.empty_cache()
    backbone = torchvision.models.mobilenet_v3_large(weights="DEFAULT" if pretrained else None)
    backbone.classifier[3] = torch.nn.Linear(1280, 1)
    model = nn.Sequential(backbone, nn.Sigmoid(), nn.Flatten(0))
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return model, optimizer


def get_dataloaders(
    dataset: DatasetDict,
    *,
    batch_size: int,
    train_subset: float | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Crée trois dataloaders pour l'entraînement, la validation et le test.

    Args:
        dataset (DatasetDict): le dataset à utiliser.
        batch_size (int): le nombre d'échantillons par batch.
        train_subset (float, optionnel): Si spécifié, fraction du dataset d'entraînement à utiliser. Par défaut, None
            (ce qui correspond à utiliser tout le dataset d'entraînement).
        num_workers (int, optionnel): le nombre de processus à utiliser pour le chargement des données. Par défaut, le
            minimum entre 8 et le nombre de coeurs de la machine. Pour désactiver le chargement parallèle, mettre à 0.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: les dataloaders pour l'entraînement, la validation et le test.
    """
    if train_subset is not None:
        if not 0 < train_subset < 1:
            raise ValueError("train_subset must be a float in the range (0, 1)")

        dataset = deepcopy(dataset)
        dataset["train"] = dataset["train"].train_test_split(
            train_size=train_subset,
            seed=0,
            stratify_by_column="label",
            load_from_cache_file=False,
        )["train"]

    _transform = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
        ]
    )

    def transform(data: dict[str, list[torch.Tensor]]) -> dict[str, list[torch.Tensor]]:
        data["image"] = [_transform(image) for image in data["image"]]
        return data

    train_loader = DataLoader(dataset["train"].with_transform(transform), shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(dataset["validation"].with_transform(transform), batch_size=batch_size)
    test_loader = DataLoader(dataset["test"].with_transform(transform), batch_size=batch_size)

    return train_loader, val_loader, test_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
) -> list[float]:
    """Entraîne le modèle sur une époque.

    Args:
        model (nn.Module): le modèle à entraîner.
        loader (DataLoader): le dataloader pour l'entraînement.
        optimizer (Optimizer): l'optimiseur à utiliser.

    Returns:
        list[float]: la liste des valeurs de loss pour chaque batch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    losses = []
    for batch in tqdm(loader, leave=False, desc="Training"):
        images, targets = batch["image"].to(device), batch["label"].to(device).float()
        optimizer.zero_grad()
        probs = model(images)
        loss = F.binary_cross_entropy(probs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def validate(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[float, float, list[int]]:
    """Évalue le modèle sur l'ensemble de validation.

    Args:
        model (nn.Module): le modèle à évaluer.
        loader (DataLoader): le dataloader pour la validation.

    Returns:
        tuple[float, float, list[int]: la loss moyenne, l'accuracy, et les prédictions du modèle.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    sum_loss = 0
    num_correct = 0
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False, desc="Validation"):
            images, targets = batch["image"].to(device), batch["label"].to(device).float()
            probs = model(images)
            sum_loss += F.binary_cross_entropy(probs, targets, reduction="sum").item()
            preds_ = probs > 0.5
            num_correct += preds_.eq(targets.view_as(preds_)).sum().item()
            preds.extend(preds_.cpu().int().tolist())

    avg_loss = sum_loss / len(loader.dataset)
    accuracy = num_correct / len(loader.dataset)

    return avg_loss, accuracy, preds


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
