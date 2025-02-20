'''
TACO utilise le format COCO -> fichier JSON contenant :
    liste des images du dataset
    liste des objets annotés pour chaque image
    categories des classe (platisque, métal, verre...)
YOLO -> fichier TXT par image avec une ligne par objet au format :
    class_id x_center y_center width height
    Normalisé entre 0 et 1

Conversion :
class_id = category_id - 1 (COCO commence à 1, YOLO commence à 0).
x_center = (x_min + w/2) / image_width
y_center = (y_min + h/2) / image_height
width = w / image_width
height = h / image_height


Script : Lit annotations.json du dataset TACO
Crée un dossier labels/ contenant un .txt par image
Convertit chaque annotation COCO en YOLO

'''
import json
import os
import shutil

# Définition des dossiers
TACO_YOLO_DIR = "TACO-YOLO"
IMG_TRAIN_DIR = os.path.join(TACO_YOLO_DIR, "images/train")
IMG_VAL_DIR = os.path.join(TACO_YOLO_DIR, "images/val")
IMG_TEST_DIR = os.path.join(TACO_YOLO_DIR, "images/test")

LABEL_TRAIN_DIR = os.path.join(TACO_YOLO_DIR, "labels/train")
LABEL_VAL_DIR = os.path.join(TACO_YOLO_DIR, "labels/val")
LABEL_TEST_DIR = os.path.join(TACO_YOLO_DIR, "labels/test")

# Créer les dossiers labels s'ils n'existent pas
for folder in [LABEL_TRAIN_DIR, LABEL_VAL_DIR, LABEL_TEST_DIR]:
    os.makedirs(folder, exist_ok=True)

# Charger le fichier COCO
with open("data/annotations.json", "r") as f:
    coco_data = json.load(f)

# Charger les classes (catégories COCO)
categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

# Fonction pour récupérer les noms d'images déjà triées
def get_image_filenames(directory):
    return {img for img in os.listdir(directory) if img.endswith(".jpg")}

# Récupérer les images présentes dans train, val et test
train_images = get_image_filenames(IMG_TRAIN_DIR)
val_images = get_image_filenames(IMG_VAL_DIR)
test_images = get_image_filenames(IMG_TEST_DIR)

# Traitement de chaque image présente dans l'annotation COCO
for image in coco_data["images"]:
    image_id = image["id"]
    img_filename = image["file_name"]
    
    # Vérifier dans quel dossier se trouve l'image
    if img_filename in train_images:
        label_output = os.path.join(LABEL_TRAIN_DIR, img_filename.replace(".jpg", ".txt"))
    elif img_filename in val_images:
        label_output = os.path.join(LABEL_VAL_DIR, img_filename.replace(".jpg", ".txt"))
    elif img_filename in test_images:
        label_output = os.path.join(LABEL_TEST_DIR, img_filename.replace(".jpg", ".txt"))
    else:
        continue  # Si l'image ne fait pas partie du dataset trié, on ignore

    labels = []
    for annotation in coco_data["annotations"]:
        if annotation["image_id"] == image_id:
            # Extraire bbox COCO
            x_min, y_min, width, height = annotation["bbox"]
            
            # Conversion en format YOLO
            img_w, img_h = image["width"], image["height"]
            x_center = (x_min + width / 2) / img_w
            y_center = (y_min + height / 2) / img_h
            width = width / img_w
            height = height / img_h
            
            # Ajuster class_id pour YOLO (YOLO commence à 0)
            class_id = annotation["category_id"] - 1  

            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # Sauvegarder le fichier YOLO
    with open(label_output, "w") as f:
        f.write("\n".join(labels))

print("✔ Conversion COCO → YOLO terminée avec succès !")
