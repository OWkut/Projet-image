import numpy as np
import matplotlib.pyplot as plt


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


def negatif(image):
    """Inverse les couleurs d'une image."""
    return 255 - image


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
