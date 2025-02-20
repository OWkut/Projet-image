import os
import json
import shutil
import random


# Définition des dossiers
DATASET_DIR = "data"
TACO_YOLO_DIR = "TACO-YOLO"
IMG_TRAIN_DIR = os.path.join(TACO_YOLO_DIR, "images/train")
IMG_VAL_DIR = os.path.join(TACO_YOLO_DIR, "images/val")
IMG_TEST_DIR = os.path.join(TACO_YOLO_DIR, "images/test")

# Créer les dossiers s'ils n'existent pas
for folder in [IMG_TRAIN_DIR, IMG_VAL_DIR, IMG_TEST_DIR]:
    os.makedirs(folder, exist_ok=True)

# Charger les annotations
with open(os.path.join(DATASET_DIR, "annotations.json"), "r") as f:
    annotations = json.load(f)

# Liste des images
images = annotations["images"]

# Mélanger les images pour un bon split
random.shuffle(images)

# Répartition des images (80% train, 10% val, 10% test)
num_images = len(images)
train_split = int(0.8 * num_images)
val_split = int(0.9 * num_images)

train_images = images[:train_split]
val_images = images[train_split:val_split]
test_images = images[val_split:]

missing_files = 0  # Compteur d'images manquantes

def move_images(images_list, dest_folder):
    global missing_files  # Utilisation du compteur
    for img in images_list:
        img_filename = img["file_name"]  # Ex: "batch_9/000085.jpg"
        src_img_path = os.path.join(DATASET_DIR, img_filename)
        dest_img_path = os.path.join(dest_folder, img_filename)

        # Créer le dossier batch_i dans train/val/test si nécessaire
        dest_folder_path = os.path.dirname(dest_img_path)
        os.makedirs(dest_folder_path, exist_ok=True)

        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dest_img_path)
        else:
            print(f"⚠️ Image introuvable : {src_img_path}")
            missing_files += 1

# Déplacer les images après création des dossiers
move_images(train_images, IMG_TRAIN_DIR)
move_images(val_images, IMG_VAL_DIR)
move_images(test_images, IMG_TEST_DIR)

print("Images deplacees dans train, val et test")
print(f"{missing_files} images non trouvées ! Vérifiez les chemins.")
print("Pensez a transformer les annotations COCO en YOLO avec COCO_to_YOLO.py")
