import numpy as np
import matplotlib.pyplot as plt
from numpy.ma import array

# ========================================================================== #
#                       TRANSFORMATION MORPHOLOGIQUE                         #
# ========================================================================== #


def erosion(image, kernel):
    """
    Applique l'érosion sur une image en niveaux de gris avec un élément structurant (kernel).
    """
    assert len(kernel.shape) == 2, "ERREUR : Le Kernel doit être de dimension 2"
    assert len(image.shape) == 2, "ERREUR : L'image doit être de dimension 2"
    assert isinstance(image, np.ndarray), "ERREUR : L'image doit être un tableau numpy"
    assert isinstance(
        kernel, np.ndarray
    ), "ERREUR : Le kernel doit être un tableau numpy"

    k_hauteur, k_largeur = kernel.shape
    i_hauteur, i_largeur = image.shape
    pad_h, pad_l = k_hauteur // 2, k_largeur // 2

    value = 1 if image.max() == 1 else 255

    image_padded = np.pad(
        image, ((pad_h, pad_h), (pad_l, pad_l)), mode="constant", constant_values=value
    )
    image_eroder = image.copy()

    for i in range(i_hauteur):
        for j in range(i_largeur):
            region = image_padded[i : i + k_hauteur, j : j + k_largeur]
            image_eroder[i, j] = np.min(region[kernel == 1])

    return image_eroder


def dilatation(image, kernel):
    """
    Applique la dilatation sur une image en niveaux de gris avec un élément structurant (kernel).
    """
    assert len(kernel.shape) == 2, "ERREUR : Le Kernel doit être de dimension 2"
    assert len(image.shape) == 2, "ERREUR : L'image doit être de dimension 2"
    assert isinstance(image, np.ndarray), "ERREUR : L'image doit être un tableau numpy"
    assert isinstance(
        kernel, np.ndarray
    ), "ERREUR : Le kernel doit être un tableau numpy"

    k_hauteur, k_largeur = kernel.shape
    i_hauteur, i_largeur = image.shape
    pad_h, pad_l = k_hauteur // 2, k_largeur // 2

    image_padded = np.pad(
        image, ((pad_h, pad_h), (pad_l, pad_l)), mode="constant", constant_values=0
    )
    image_dilate = image.copy()

    for i in range(i_hauteur):
        for j in range(i_largeur):
            region = image_padded[i : i + k_hauteur, j : j + k_largeur]
            image_dilate[i, j] = np.max(region[kernel == 1])

    return image_dilate


# Filtre


# ========================================================================== #
#                            DETECTION DES CONTOURS                          #
# ========================================================================== #
def sobel_vertical(image):
    """Applique le filtre de Sobel Vertical"""
    height, width = image.shape

    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    sobel_y = np.zeros((height, width))

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = image[i - 1 : i + 2, j - 1 : j + 2]
            sobel_y[i, j] = np.sum(Gy * region)

    return sobel_y


def sobel_horizontal(image):
    """Applique le filtre de Sobel Horizontal"""
    height, width = image.shape

    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    sobel_x = np.zeros((height, width))

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = image[i - 1 : i + 2, j - 1 : j + 2]
            sobel_x[i, j] = np.sum(Gx * region)

    return sobel_x


def sobel_magnitude(image):
    """Calcul la magnitude du gradient à partir des filtres de Sobel."""
    sobel_x = sobel_horizontal(image)
    sobel_y = sobel_vertical(image)
    sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    return (sobel_mag / np.max(sobel_mag)) * 255


def prewitt_horizontal(image):
    """Applique le filtre de Prewitt horizontal"""
    Gx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    height, width = image.shape
    prewitt_x = np.zeros((height, width))

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = image[i - 1 : i + 2, j - 1 : j + 2]
            prewitt_x[i, j] = np.sum(Gx * region)

    return np.abs(prewitt_x)


def prewitt_vertical(image):
    """Applique le filtre de Prewitt sur l'axe vertical"""
    width, height = image.shape

    Gy = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    prewitt_y = np.zeros((height, width))

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = image[i - 1 : i + 2, j - 1 : j + 2]
            prewitt_y[i, j] = np.sum(Gy * region)

    return np.abs(prewitt_y)

def prewitt_maginitude(image):
    """Calcule la magnitude du gradient avec prewitt"""
    prewitt_x = prewitt_horizontal(image)
    prewitt_y = prewitt_vertical(image)
    prewitt_mag = np.sqrt(prewitt_x ** 2 + prewitt_y ** 2)
    return (prewitt_mag / np.max(prewitt_mag)) * 255

def laplacian(image):
    """Applique le filtre Laplacien pour détecter les bords"""
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0])

    height, width = image.shape
    laplacian_img = np.np.zeros((height, width))
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = image[i-1 : i+2, j-1 : j+2]
            laplacian_img[i, j] = np.sum(kernel * region)

    return np.clip(np.abs(laplacian_img), 0, 255).astype(np.uint8)

# ========================================================================== #
#                              FILTRAGE SPACIAL                              #
# ========================================================================== #


def gaussian_flou(image, kernel_size=3):
    """Applique un flou Gaussien avec un noyau carré de taille kernel_size."""
    kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16

    height, width = image.shape
    blurred = np.zeros((height, width))

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = image[i-1 : i+2, j-1 : j+2]
            blurred[i, j] = np.sum(kernel * region)

    return blurred.astype(np.uint8)

def flou_moyen(image, kernel_size=3):
    """Applique un flou moyen avec un noyau carré de taille kernel_size"""
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

    height, width = image.shape
    blurred = np.zeros((height, width))

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            region = image[i-1 : i+2, j-1 : j+2]
            blurred[i, j] = np.sum(kernel * region)

    return blurred.astype(np.uint8)

# ========================================================================== #
#             TRANSFORMATION EN ECHELLE DE GRIS ET SEUILLAGE                 #
# ========================================================================== #
def negatif(image):
    """Inverse les couleurs d'une image."""
    return 255 - image


# thresold ca veut dire seuil en anglais
def thresold(image, seuil=128):
    """Convertit l'image en noire et blanc celon un seuil"""
    return np.where(image > seuil, 255, 0).astype(np.uint8)


if __name__ == "__main__":
    image_test = (
        np.array(
            [
                [0, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=np.uint8,
        )
        * 255
    )  # Convertir en niveaux de gris (0-255)

    # Élément structurant 3x3 (carré)
    kernel_test = np.ones((3, 3), dtype=np.uint8)

    # Appliquer les filtres
    image_erosion = erosion(image_test, kernel_test)
    image_dilatation = dilatation(image_test, kernel_test)
    image_negatif = negatif(image_test)

    # Affichage des résultats
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    axes[0].imshow(image_test, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Image originale")
    axes[1].imshow(image_erosion, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Érosion")
    axes[2].imshow(image_dilatation, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Dilatation")
    axes[3].imshow(image_negatif, cmap="gray", vmin=0, vmax=255)
    axes[3].set_title("Négatif")
    plt.show()
