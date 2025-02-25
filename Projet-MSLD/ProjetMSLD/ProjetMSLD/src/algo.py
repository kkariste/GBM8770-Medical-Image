import cv2
import numpy as np
from scipy.ndimage import convolve
from sklearn.metrics import roc_curve
from copy import deepcopy
from .dataset import Sample


class MultiScaleLineDetector:
    def __init__(self, W: int, L: list[int], num_orientations: int) -> None:
        """Constructeur qui initialise un objet de type MultiScaleLineDetector. Cette méthode est appelée par
        >>> msld = MultiScaleLineDetector(W=..., L=..., num_orientations=...)

        Args:
            W (int): Taille de la fenêtre (telle que définie dans l'article).
            L (list[int]): Une *liste d'entiers* avec les longueurs des lignes qui seront détectées par la MSLD.
            num_orientations (int): Nombre d'orientations des lignes à détecter.
        """
        self.W = W
        self.L = L
        self.num_orientations = num_orientations

        # TODO: 1.2.Q1
        self.avg_mask = np.ones((W,W))/(W*W)

        # TODO: 1.2.Q2
        # line_detectors_masks est un dictionnaire contenant les masques de détection de ligne pour toutes les
        # échelles contenues dans la liste L et pour un nombre d'orientation égal à num_orientations. Ainsi pour
        # toutes les valeurs de L, self.line_detectors_masks[l] est un array de forme (num_orientations, l, l).

        self.line_detectors_masks = {}
        for l in L:
            # On calcule le détecteur de ligne initial de taille `l` (les dimensions du masque sont `(l, l)`).
            line_detector = np.zeros((l,l))
            line_detector[l//2,:]= 1/l
            line_detector = line_detector / line_detector.sum()

            # On initialise la liste des num_orientations masques de taille lxl.
            line_detectors_masks = [line_detector]
            for angle in np.linspace(180/num_orientations, 180 * (1-(1/num_orientations)), 12, endpoint=False)[1:]:
                # On effectue `num_orientations - 1` rotations du masque `line_detector`.
                # Pour un angle donné, la rotation sera effectué par
                r = cv2.getRotationMatrix2D((l // 2, l // 2), angle, 1)
                rotated_mask = cv2.warpAffine(line_detector, r, (l, l))
                line_detectors_masks.append(rotated_mask / rotated_mask.sum())

            # On assemble les `num_orientations` masques ensemble:
            self.line_detectors_masks[l] = np.stack(line_detectors_masks, axis=0)

    def basic_line_detector(self, grey_lvl: np.ndarray, L: int) -> np.ndarray:
        """Applique l'algorithme Basic Line Detector sur la carte d'intensité grey_lvl avec des lignes de longueurs L.

        Args:
            grey_lvl (np.ndarray): Carte d'intensité 2D avec dtype float sur laquelle est appliqué le BLD.
            L (int): Longueur des lignes (on supposera que L est présent dans self.L et donc que
                self.line_detectors_masks[L] existe).

        Returns:
            R (np.ndarray): Carte de réponse 2D en float du Basic Line Detector.
        """
        # TODO: 1.2.Q3
        # Les masques de détections de lignes de longueur L initialisés dans le constructeur sont accessibles par:
        # self.line_detectors_masks[L]
        line_detector = self.line_detectors_masks[L]

        # On inverse les intensités du canal vert
        inverted_grey_lvl = 1.0 - grey_lvl

        # On fait la moyenne locale dans une fenêtre W X W
        #local_avg = cv2.filter2D(inverted_grey_lvl, -1, self.avg_mask)

        # On calcule les réponses pour chacune des orientations
        #responses = [cv2.filter2D(inverted_grey_lvl, -1, mask) for mask in line_detector]
        #responses = np.stack(responses, axis=0)  # (num_orientations, H, W)

        # On trouve la réponse maximale parmi toutes les orientations
        #max_response = np.max(responses, axis=0)  # (H, W)

        # On calcule la carte de réponse
        #R = max_response - local_avg

        # Effectuons une normalisation de R
        #R_mean = np.mean(R)
        #R_std = np.std(R)
        #R = (R - R_mean) / R_std
        Imax = None
        I = np.zeros(shape= (self.num_orientations, grey_lvl.shape[0], grey_lvl.shape[1]))
        for i in range(self.num_orientations):
            line = line_detector[i, :,:]
            I[i,:,:] =convolve(inverted_grey_lvl, line)

        Imax = np.max(I, axis = 0)
        Iavg = convolve(inverted_grey_lvl, self.avg_mask)
        R = Imax - Iavg 
        R = (R- np.mean(R))/ np.std(R) #Normalisation de R.
        return R

    def multi_scale_line_detector(self, image: np.ndarray) -> np.ndarray:
        """Applique l'algorithme de Multi-Scale Line Detector et combine les réponses des BLD pour obtenir la carte
        d'intensité de l'équation 4 de la section 3.3 Combination Method.

        Args:
            image (np.ndarray): Image RGB aux intensitées en float comprises entre 0 et 1 et de dimensions
                (hauteur, largeur, canal) (canal: R=0 G=1 B=2)

        Returns:
            Rcombined (np.ndarray): Carte d'intensité combinée.
        """
        # TODO: 1.3.Q1
        # Pour les hyperparamètres L et W utilisez les valeurs de self.L et self.W.
        Iigc = image[:,:, 1]
        #Rcombined = Iigc.copy()
        Rcombined = deepcopy(Iigc)
        for l in self.L :
            R = self.basic_line_detector(Iigc, l)
            Rcombined += R

        Rcombined = Rcombined / (len(self.L)+1)

        return Rcombined

    def segment_vessels(self, image: np.ndarray, threshold: float) -> np.ndarray:
        """
        Segmente les vaisseaux sur une image en utilisant la MSLD.

        Args:
            image (np.ndarray): Image RGB sur laquelle appliquer l'algorithme MSLD.
            threshold (float): Le seuil à appliquer à la réponse MSLD.

        Returns:
            vessels (np.ndarray): Carte binaire 2D de la segmentation des vaisseaux.
        """
        # TODO: 1.5.Q1
        # Utilisez self.multi_scale_line_detector(image) et threshold.

        image_multi_scale_line_detector = self.multi_scale_line_detector(image)
        vessels = image_multi_scale_line_detector > threshold
        return vessels


def learn_threshold(msld: MultiScaleLineDetector, dataset: list[Sample]) -> tuple[float, float]:
    """Apprend le seuil optimal pour obtenir la précision la plus élevée sur le dataset donné.

    Args:
        msld (MultiScaleLineDetector): L'objet MSLD qu'on souhaite évaluer.
        dataset (list[Sample]): Dataset sur lequel apprendre le seuil.

    Returns:
        threshold (float): Seuil proposant la meilleure précision
        accuracy (float): Valeur de la meilleure précision
    """
    fpr, tpr, thresholds = roc(msld, dataset)

    # TODO: 1.4.Q3
    # Utilisez np.argmax pour trouver l'indice du maximum d'un vecteur.

    print(f"fpr = {fpr} \n tpr = {tpr} \n thresholds = {thresholds}")

    # TODO: 1.4.Q3
    # Nombre total de pixels positifs et négatifs.
    P = sum(np.sum((d.label[d.mask])) for d in dataset)  # Pixels positifs ET dans le masque
    N = sum(np.sum((~d.label[d.mask])) for d in dataset) # Pixels négatifs ET dans le masque
    S = P + N
    print(f"P = {P} \n N = {N}")


    precisions = (tpr * P + (1-fpr) * N) / S

    # Utilisez np.argmax pour trouver l'indice du maximum d'un vecteur.
    best_index = np.argmax(precisions)

    # Le seuil optimal et la précision associée
    threshold = thresholds[best_index]
    accuracy = precisions[best_index]




    #acc = np.zeros(shape= (len(dataset), len(thresholds)))
    #for d, sample in enumerate (dataset) :
      #  mask = sample.mask
       # positif = np.sum((d.label[d.mask]))   # Pixels positifs ET dans le masque
        #negatif = np.sum((~d.label[d.mask]))  # Pixels négatifs ET dans le masque
        #sum_predic = positif + negatif
        #positif = np.sum(mask)
        #negatif = sum_predic - positif
        # Précision pour tous les seuil de l'échantillon
     #   tnr = 1 - np.array(fpr)
      #  acc[d, :] = (np.array(tpr) * positif + np.array(tnr) * negatif) / sum_predic

    # Calcul de la précision moyenne pour tous les seuils de tous les échantillons
  #  avg_acc = np.mean(acc, axis = 0)
    # Calcul de l'indice de la précision
   # idx_max = np.argmax(avg_acc)
    #threshold = thresholds[idx_max]
    #accuracy = avg_acc[idx_max]

    return threshold, accuracy


def naive_metrics(msld: MultiScaleLineDetector, dataset: list[Sample], threshold: float) -> tuple[float, np.ndarray]:
    """
    Évalue la précision et la matrice de confusion de l'algorithme sur
    un dataset donné et sur la région d'intérêt indiquée par le
    champs mask.

    Args:
        msld (MultiScaleLineDetector): L'objet MSLD qu'on souhaite évaluer.
        dataset (list[Sample]): Base de données sur laquelle calculer les métriques.

    Returns:
        accuracy (float): Précision.
        confusion_matrix (np.ndarray): Matrice de confusion 2 x 2 normalisée par le nombre de labels positifs et négatifs.
    """
    # TODO: 2.1.Q1
    
    y_true = []
    y_pred = []

    for d in dataset:
        # Pour chaque élément de dataset, on lit les champs "image", "label" et "mask".
        image, label, mask = d.image, d.label, d.mask

        # On calcule la prédiction du msld sur cette image.
        response = msld.segment_vessels(image, threshold)

        # On applique les masques à label et prediction pour qu'ils contiennent uniquement
        # la liste des pixels qui appartiennent au masque.
        label = label[mask]
        response = response[mask]

        # On ajoute les vecteurs label et prediction aux listes y_true et y_pred
        y_true.append(label)
        y_pred.append(response)

    # On concatène les vecteurs de la listes y_true pour obtenir un unique vecteur contenant
    # les labels associés à tous les pixels qui appartiennent au masque du dataset.
    y_true = np.concatenate(y_true)
    # Même chose pour y_pred.
    y_pred = np.concatenate(y_pred)

    p = np.sum(y_true)
    n = np.sum(~y_true)
    tp = np.sum(y_true & y_pred)
    tn = np.sum((~y_true) & (~y_pred))
    fp = np.sum((~y_true) & y_pred)
    fn = np.sum(y_true & (~y_pred))

    # Calcul de la précision et la matrice de confusion
    accuracy = (tp + tn)/ (tp + tn + fp + fn)
    confusion_matrix = np.array([[tp / p, fn / p], [fp / n, tn / n]])

    return accuracy, confusion_matrix


def roc(msld: MultiScaleLineDetector, dataset: list[Sample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calcule la courbe ROC de l'algorithme MSLD sur un dataset donné et sur la région d'intérêt indiquée par le
    champ "mask".

    Args:
        msld (MultiScaleLineDetector): L'objet MSLD qu'on souhaite évaluer.
        dataset (list[Sample]): Base de données sur laquelle calculer la courbe ROC.

    Returns:
        fpr (np.ndarray): Vecteur float des taux de faux positifs.
        tpr (np.ndarray): Vecteur float des taux de vrais positifs.
        thresholds (np.ndarray): Vecteur float des seuils associés à ces taux.
    """

    y_true = []
    y_pred = []

    for d in dataset:
        # Pour chaque élément de dataset, on lit les champs "image", "label" et "mask".
        image, label, mask = d.image, d.label, d.mask

        # On calcule la prédiction du msld sur cette image.
        response = msld.multi_scale_line_detector(image)

        # On applique les masques à label et prediction pour qu'ils contiennent uniquement
        # la liste des pixels qui appartiennent au masque.
        label = label[mask]
        response = response[mask]

        # On ajoute les vecteurs label et prediction aux listes y_true et y_pred
        y_true.append(label)
        y_pred.append(response)

    # On concatène les vecteurs de la listes y_true pour obtenir un unique vecteur contenant
    # les labels associés à tous les pixels qui appartiennent au masque du dataset.
    y_true = np.concatenate(y_true)
    # Même chose pour y_pred.
    y_pred = np.concatenate(y_pred)

    # On calcule le taux de vrai positif et de faux positif du dataset pour chaque seuil possible.
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    return fpr, tpr, thresholds


def dice(msld: MultiScaleLineDetector, dataset: list[Sample], threshold: float) -> float:
    """Évalue l'indice Sørensen-Dice de l'algorithme sur un dataset donné et sur la région d'intérêt indiquée par le
    champ "mask".

    Args:
        msld (MultiScaleLineDetector): L'objet MSLD qu'on souhaite évaluer.
        dataset (list[Sample]): Base de données sur laquelle calculer l'indice Dice.

    Returns:
        dice_index (float): Indice de Sørensen-Dice.
    """
    # TODO: 2.2.Q2
    # Vous pouvez utiliser la fonction `_dice` fournie tout en bas de ce fichier.

    y_true = []
    y_pred = []

    for d in dataset:
        # Pour chaque élément de dataset, on lit les champs "image", "label" et "mask".
        image, label, mask = d.image, d.label, d.mask

        # On calcule la prédiction du msld sur cette image.
        response = msld.segment_vessels(image, threshold)

        # On applique les masques à label et prediction pour qu'ils contiennent uniquement
        # la liste des pixels qui appartiennent au masque.
        label = label[mask]
        response = response[mask]

        # On ajoute les vecteurs label et prediction aux listes y_true et y_pred
        y_true.append(label)
        y_pred.append(response)

    # On concatène les vecteurs de la listes y_true pour obtenir un unique vecteur contenant
    # les labels associés à tous les pixels qui appartiennent au masque du dataset.
    y_true = np.concatenate(y_true)
    # Même chose pour y_pred.
    y_pred = np.concatenate(y_pred)

    
    dice_index = _dice(y_true, y_pred)

    return dice_index


def _dice(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Calcule l'indice de Sørensen-Dice entre les prédictions et la vraie segmentation. Les deux arrays doivent avoir
    la même forme.

    Args:
        targets (np.ndarray): Les vraies valeurs de segmentation.
        predictions (np.ndarray): Prédiction de la segmentation.

    Returns:
        dice_index (float): Indice de Sørensen-Dice.
    """
    dice_index = 2 * np.sum(targets * predictions) / (targets.sum() + predictions.sum())
    return dice_index
