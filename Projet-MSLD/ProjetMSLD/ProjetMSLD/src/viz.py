import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

from .algo import MultiScaleLineDetector, roc
from .dataset import Sample


def show_diff(msld: MultiScaleLineDetector, sample: Sample, threshold: float, ax: plt.Axes = None) -> None:
    """Affiche la comparaison entre la prédiction de l'algorithme et les valeurs attendues (labels) selon le code
    couleur suivant:
       - Noir: le pixel est absent de la prédiction et du label
       - Rouge: le pixel n'est présent que dans la prédiction
       - Bleu: le pixel n'est présent que dans le label
       - Blanc: le pixel est présent dans la prédiction ET le label

    Args:
        msld (MultiScaleLineDetector): L'objet MSLD qu'on souhaite évaluer.
        sample (Sample): Échantillon de la base de données contenant les champs "image", "label" et "mask".
        threshold (float): Le seuil à appliquer à la réponse MSLD.
        ax (plt.Axes, optionnel): Un système d'axes dans lequel afficher la comparaison. Par défaut, None.
    """

    # Calcule la segmentation des vaisseaux
    pred = msld.segment_vessels(sample.image, threshold)

    # Applique le masque à la prédiction et au label
    pred = pred & sample.mask
    label = sample.label & sample.mask

    # Calcule chaque canal de l'image:
    # rouge: 1 seulement pred est vrai, 0 sinon
    # bleu: 1 seulement si label est vrai, 0 sinon
    # vert: 1 seulement si label et pred sont vrais (de sorte que la couleur globale soit blanche), 0 sinon
    red = pred.astype(float)
    blue = label.astype(float)
    green = (pred & label).astype(float)

    rgb = np.stack([red, green, blue], axis=2)

    if ax is None:
        # Si `ax` n'est pas spécifié en argument, on crée une nouvelle figure et un nouveau système d'axes.
        # On peut alors afficher la comparaison avec les méthodes de `ax`.
        fig, ax = plt.subplots(1, 1)
    ax.imshow(rgb)
    ax.set_axis_off()
    ax.set_title("Différences entre la segmentation prédite et attendue")


def plot_roc(msld: MultiScaleLineDetector, dataset: list[Sample], ax: plt.Axes = None, titre: str = None) -> float:
    """Affiche la courbe ROC et calcule l'AUC de l'algorithme pour un dataset donnée et sur la région d'intérêt indiquée
    par le champ mask.

    Args:
        msld (MultiScaleLineDetector): L'objet MSLD qu'on souhaite évaluer.
        dataset (list[Sample]): Base de données sur laquelle calculer l'AUC.
        ax (plt.Axes, optionnel): Un système d'axes dans lequel afficher la courbe ROC. Par défaut, None.

    Returns:
        roc_auc (float): Aire sous la courbe ROC.
    """
    if ax is None:
        # Si `ax` n'est pas spécifié en argument, on crée une nouvelle figure et un nouveau système d'axes.
        # On peut alors afficher la courbe ROC avec les méthodes de `ax`.
        fig, ax = plt.subplots(1, 1)
        

    # TODO: 2.3.Q2
    # Utilisez la fonction roc(dataset) déjà implémentée.

    fpr, tpr, _ = roc(msld, dataset)
    roc_auc = auc(fpr, tpr)

    ax.plot(fpr, tpr, label=f"AUC {titre} = {roc_auc}")
    ax.set_xlabel('Taux de faux positifs')
    ax.set_ylabel('Taux de vrais positifs')
    ax.set_title('Courbe ROC')
    ax.legend()
    ax.grid()

    return roc_auc
