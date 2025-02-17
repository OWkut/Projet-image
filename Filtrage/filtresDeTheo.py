import numpy as np
import cv2
from PIL import Image
from skimage import feature, util
import matplotlib.pyplot as plt


def filtre_moyenne(image):
    image_gray = image.convert("L")
    image_array = np.array(image_gray)

    kernel = np.ones((5, 5), np.float32) / 25
    image_filtered = cv2.filter2D(image_array, -1, kernel)

    return image_filtered


def filtre_moyenneur_non_uniforme(image):
    image_gray = image.convert("L")
    image_array = np.array(image_gray)

    # Créer un filtre moyenneur non uniforme
    fltrm = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    # Appliquer ce filtre avec cv2.filter2D
    image_filtered_non_uniform = cv2.filter2D(image_array, -1, fltrm)

    return image_filtered_non_uniform


def filtre_gaussien(image, sigma=2):
    image_gray = image.convert("L")
    image_array = np.array(image_gray)

    # Appliquer un filtre gaussien avec cv2.GaussianBlur
    image_filtered_gaussian = cv2.GaussianBlur(image_array, (5, 5), sigmaX=sigma)

    return image_filtered_gaussian


def filtre_laplacien(image):
    image_gray = image.convert("L")
    image_array = np.array(image_gray)

    # Créer un filtre Laplacien en utilisant cv2.Laplacian
    image_filtered_laplacian = cv2.Laplacian(image_array, cv2.CV_64F)

    # Convertir en type uint8 pour l'affichage
    image_filtered_laplacian = cv2.convertScaleAbs(image_filtered_laplacian)

    return image_filtered_laplacian


def filtre_sobel(image):
    image_gray = image.convert("L")
    image_array = np.array(image_gray)

    # Appliquer un filtre de Sobel pour détecter les contours
    image_sobel_x = cv2.Sobel(image_array, cv2.CV_64F, 1, 0, ksize=5)
    image_sobel_y = cv2.Sobel(image_array, cv2.CV_64F, 0, 1, ksize=5)

    # Combiner les résultats
    image_sobel = np.sqrt(image_sobel_x**2 + image_sobel_y**2)

    return image_sobel


def filtre_prewitt(image):
    image_gray = image.convert("L")
    image_array = np.array(image_gray)

    # Filtres de Prewitt horizontaux et verticaux
    prewitt_h = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    prewitt_v = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    contour_prewitt_h = cv2.filter2D(image_array, -1, prewitt_h)
    contour_prewitt_v = cv2.filter2D(image_array, -1, prewitt_v)

    # Combiner les résultats
    contour_prewitt = np.sqrt(contour_prewitt_h**2 + contour_prewitt_v**2)

    return contour_prewitt


def filtre_canny(image):
    image_gray = image.convert("L")
    image_array = np.array(image_gray)

    # Appliquer le filtre de Canny
    contour_canny = feature.canny(image_array / 255.0)  # Normalisation entre 0 et 1

    return contour_canny


def filtre_robert(image):
    image_gray = image.convert("L")
    image_array = np.array(image_gray)

    A = np.array([[1, 0], [0, -1]])
    B = np.array([[0, 1], [-1, 0]])

    contour_robert_A = cv2.filter2D(image_array, -1, A)
    contour_robert_B = cv2.filter2D(image_array, -1, B)

    return np.sqrt(contour_robert_A**2 + contour_robert_B**2)


def ajouter_bruit(image, var=0.01):
    image_gray = image.convert("L")
    image_array = np.array(image_gray)

    # Ajouter un bruit gaussien
    image_noisy = util.random_noise(image_array, mode="gaussian", var=var) * 255
    image_noisy = image_noisy.astype(np.uint8)

    return image_noisy
