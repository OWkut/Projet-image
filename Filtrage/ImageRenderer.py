# ============================================================
# 🎨 IMAGE RENDERER - Traitement et Filtrage d'Images en Python
# ============================================================
# 📌 Description :
# `ImageRenderer` est une classe Python permettant de :
#   ✅ Charger des images depuis un dossier
#   ✅ Générer des images aléatoires en niveaux de gris ou binaires
#   ✅ Appliquer des **filtres morphologiques et transformations**
#   ✅ Afficher une image unique ou une grille d'images
#   ✅ Gérer une liste de filtres et leur application en séquence
#
# 🛠️ Fonctionnalités principales :
#   - 💾 **Chargement** et conversion d'images (PNG, JPG, BMP, TIFF)
#   - 🎨 **Filtres disponibles** :
#       - `Négatif`, `Sobel`, `Prewitt`, `Laplacian`, `Flou`, `Affinage`, `Dilatation`, `Érosion`, `Seuillage`, etc.
#   - 🔄 **Application de filtres en chaîne**
#   - 📊 **Affichage d’images individuelles ou multiples**
#   - 🛠️ **Ajout facile de nouveaux filtres**
#

# ============================================================
# 🔥 Auteur : GOAREGUER Mael
# 📂 Version : 1.0.3
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
from PIL import Image
import Filtre as flt


class ImageRenderer:
    def __init__(self, images=None, hauteur=512, largeur=512):
        """
        Classe permettant le traitement d'image et l'application de filtre
        @param images [np.array]: Liste d'image
        @param hauteur [int]: hauteur des images
        @param largeur [int]: largeur des images
        """
        self.largeur = largeur
        self.hauteur = hauteur
        self.images = images
        self.image_names = {}
        self.cmap_color = "gray"
        self.filtres = {}
        self.filtres_actif = []
        # Ici j'ajoute les filtre
        self.add_filter("negatif", flt.negatif)
        self.add_filter("sobel_vertical", flt.sobel_vertical)
        self.add_filter("sobel_horizontal", flt.sobel_horizontal)
        self.add_filter("sobel_magnitude", flt.sobel_magnitude)
        self.add_filter("prewitt_vertical", flt.prewitt_vertical)
        self.add_filter("prewitt_horizontal", flt.prewitt_horizontal)
        self.add_filter("prewitt_magnitude", flt.prewitt_maginitude)
        self.add_filter("laplacian", flt.laplacian)
        self.add_filter("gaussian_flou", flt.gaussian_flou)
        self.add_filter("flou_moyen", flt.flou_moyen)
        self.add_filter("affinage", flt.affinage)
        self.add_filter("relief", flt.relief)
        self.add_filter("seuillage", flt.seuillage)
        self.add_filter("dilatation", flt.dilatation)
        self.add_filter("erosion", flt.erosion)

    def add_filter(self, nom, fct):
        assert callable(fct), "ERREUR: La fonction du filtre doit être appellable"
        self.filtres[nom] = fct

    def call_filter(self, nom, *args, **kwargs):
        assert nom in self.filtres, f"ERREUR : Filtre '{nom}' non trouvé."

        self.filtres_actif.append(nom)

        if isinstance(self.images, list):
            self.images = np.array(self.images)

        print(f"\n🛠️ Application du filtre '{nom}' sur {len(self.images)} images...\n")

        # Barre de progression
        self.images = np.array(
            [
                self.filtres[nom](img, *args, **kwargs)
                for img in tqdm(self.images, desc=f"📸 Filtre '{nom}' en cours")
            ]
        )

        print(f"\n✅ Filtre '{nom}' appliqué avec succès !")

    def call_filters(self, filtres):
        """
        Applique plusieurs filtres en séquence sur les images

        @param filtres [list] : Liste de tuples (nom_filtre, args, kwargs)
        """

        for nom, args, kwargs in filtres:
            self.call_filter(nom, *args, **kwargs)

    def generate_images(self, taille, mode="binaire"):
        assert mode in ["binaire", "gris"], "ERREUR: Le mode renseigner est inconnu"

        if mode == "binaire":
            self.images = np.random.choice(
                [0, 1], size=(taille, self.hauteur, self.largeur)
            )
        elif mode == "gris":
            self.images = np.random.randint(
                0, 256, size=(taille, self.hauteur, self.largeur), dtype=np.uint8
            )

    def load_images(self, path, formats=(".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
        path = os.path.join(os.path.dirname(__file__), path)
        assert os.path.exists(path), f"ERREUR : Le dossier '{path}' n'existe pas."

        images = []
        fichiers = [f for f in os.listdir(path) if f.lower().endswith(formats)]

        assert len(fichiers) > 0, f"ERREUR : Aucune image trouvée dans '{path}'."

        print(f"\n📂 Chargement de {len(fichiers)} images depuis '{path}'...\n")

        for fichier in tqdm(fichiers, desc="🔄 Chargement des images"):
            image_path = os.path.join(path, fichier)
            image = Image.open(image_path).convert("L")  # Convertir en niveaux de gris
            image = image.resize((self.largeur, self.hauteur))
            image = np.array(image, dtype=np.uint8)
            images.append(image)
            self.image_names[len(images) - 1] = fichier

        self.images = np.array(images)

        print(f"\n✅ {len(self.images)} images chargées avec succès !")

    def afficherImage(self, image, axe, cmap, titre, vmin=0, vmax=1):
        axe.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        axe.set_title(titre)
        axe.axis("off")

    def renderImage(self, index):
        assert (
            self.images is not None and len(self.images) > 0
        ), "ERREUR : La liste d'images est vide."

        if isinstance(self.images, list):
            self.images = np.array(self.images)  # Conversion list->np.array

        assert (
            self.images.ndim == 3
        ), "ERREUR : La liste doit être de dimensions (nb_images, hauteur, largeur)."
        assert isinstance(
            self.images, np.ndarray
        ), "ERREUR : La liste doit être de type np.array"
        assert (
            len(self.images) < index or index >= 0
        ), "ERREUR: l'index est en dehors de la liste d'image"

        vmin_value = 0
        vmax_value = (
            1 if self.images[index].max() == 1 else 255
        )  # On adapte celon binaire/gris

        plt.imshow(
            self.images[index], cmap=self.cmap_color, vmin=vmin_value, vmax=vmax_value
        )
        plt.title("Image")
        plt.axis("off")
        plt.show()

    def renderImages(self):
        assert (
            self.images is not None and len(self.images) > 0
        ), "ERREUR : La liste d'images est vide."

        if isinstance(self.images, list):
            self.images = np.array(self.images)  # Conversion list->np.array

        assert (
            self.images.ndim == 3
        ), "ERREUR : La liste doit être de dimensions (nb_images, hauteur, largeur)."
        assert isinstance(
            self.images, np.ndarray
        ), "ERREUR : La liste doit être de type np.array"

        taille = len(self.images)
        nbcols = min(taille, 5)
        nbrows = (taille + nbcols - 1) // nbcols

        vmin_value = 0
        vmax_value = (
            1 if self.images.max() == 1 else 255
        )  # On adapte celon binaire/gris

        fig, axes = plt.subplots(nbrows, nbcols, figsize=(10, 6))

        # Dans le cas ou on à qu'un image
        if nbrows == 1:
            axes = np.array(axes).reshape(1, -1)

        for index, image in enumerate(self.images):
            # Donne pour row le nombre de fois que l'on peu diviser et dans col le reste
            row, col = divmod(index, nbcols)
            self.afficherImage(
                image,
                axes[row, col],
                self.cmap_color,
                f"Image n°{index}",
                vmin_value,
                vmax_value,
            )

        plt.tight_layout()
        plt.show()

    def getImage(self):
        return self.images

    def setImages(self, images):
        assert isinstance(
            images, np.ndarray
        ), "ERREUR: la liste d'image doit être du type np.array"

        self.images = images

    def getFiltres(self):
        return self.filtres

    def normalize_contrast(self):
        """
        Améliore le contraste des images en normalisant les valeurs de pixels (Min-Max).
        """
        assert (
            self.images is not None and len(self.images) > 0
        ), "ERREUR : Aucune image à traiter."

        if isinstance(self.images, list):
            self.images = np.array(self.images)

        assert (
            self.images.ndim == 3
        ), "ERREUR : Les images doivent être de dimensions (nb_images, hauteur, largeur)."

        self.images = np.array(
            [
                cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                for img in tqdm(self.images, desc="⚡ Normalisation en cours")
            ],
            dtype=np.uint8,
        )

    def __str__(self) -> str:
        nb_images = len(self.images)
        nb_filtres = len(self.filtres)

        filtres_str = (
            "Aucun filtre ajouté" if nb_filtres == 0 else ", ".join(self.filtres.keys())
        )

        filtres_actif_str = (
            "Aucun filtre appliqué"
            if len(self.filtres_actif) == 0
            else ", ".join(self.filtres_actif)
        )

        return (
            "🖼️ ImageRenderer\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📏 => Taille des images : {self.largeur} x {self.hauteur} pixels\n"
            f"📸 => Nombre d'images   : {nb_images}\n"
            f"🎨 => Filtres disponibles : {nb_filtres} filtres\n"
            f"🛠️ => Liste des filtres  : {filtres_str}\n"
            f"🔥 => Filtre appliquer : {filtres_actif_str}\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )


# J'ai mis ca pour tester quand on execute ce code en mode scrip et pas en mode module

kernel = np.ones((3, 3), dtype=np.uint8)

if __name__ == "__main__":
    imageRenderer = ImageRenderer()
    imageRenderer.load_images("Ressources")
    imageRenderer.call_filter("sobel_magnitude")
    imageRenderer.renderImage(0)
    print(imageRenderer)
