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
import glob

# Définition des dossiers
TACO_YOLO_DIR = "TACO-YOLO"
IMG_TRAIN_DIR = os.path.join(TACO_YOLO_DIR, "images/train")
IMG_VAL_DIR = os.path.join(TACO_YOLO_DIR, "images/val")
IMG_TEST_DIR = os.path.join(TACO_YOLO_DIR, "images/test")

LABEL_TRAIN_DIR = os.path.join(TACO_YOLO_DIR, "labels/train")
LABEL_VAL_DIR = os.path.join(TACO_YOLO_DIR, "labels/val")
LABEL_TEST_DIR = os.path.join(TACO_YOLO_DIR, "labels/test")

# Charger le fichier COCO
json_path = "data/annot.json"
if not os.path.exists(json_path):
    print(f"❌ Erreur : Le fichier {json_path} est introuvable.")
    exit(1)

with open(json_path, "r") as f:
    coco_data = json.load(f)

print(f"✅ Nombre d'images dans COCO: {len(coco_data['images'])}")
print(f"✅ Nombre d'annotations dans COCO: {len(coco_data['annotations'])}")

# Charger les catégories COCO
categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

# Fonction pour récupérer la structure des fichiers avec les sous-dossiers
def get_image_structure(directory):
    images = {}
    if not os.path.exists(directory):
        print(f"⚠ Le dossier {directory} est introuvable.")
        return images

    for img_path in glob.glob(f"{directory}/**/*.jpg", recursive=True):
        img_name = os.path.basename(img_path)  # Nom du fichier
        relative_folder = os.path.relpath(os.path.dirname(img_path), directory)  # Ex: "0", "1", etc.
        images[img_name] = relative_folder  # Associe "image.jpg" → "0"
    return images

# Récupérer la structure des images dans train, val et test
train_images = get_image_structure(IMG_TRAIN_DIR)
val_images = get_image_structure(IMG_VAL_DIR)
test_images = get_image_structure(IMG_TEST_DIR)

print(f"📂 Images trouvées dans train : {len(train_images)}")
print(f"📂 Images trouvées dans val : {len(val_images)}")
print(f"📂 Images trouvées dans test : {len(test_images)}")

# Dictionnaire pour regrouper les annotations par image_id
annotations_by_image = {image["id"]: [] for image in coco_data["images"]}
for annotation in coco_data["annotations"]:
    annotations_by_image[annotation["image_id"]].append(annotation)

# Traitement des images et conversion en format YOLO
for image in coco_data["images"]:
    image_id = image["id"]
    img_filename = image["file_name"].replace("\\", "/").split("/")[-1]  # Prend juste le nom du fichier
    img_w, img_h = image["width"], image["height"]

    # Trouver dans quel dossier est l'image
    if img_filename in train_images:
        subfolder = train_images[img_filename]
        label_dir = os.path.join(LABEL_TRAIN_DIR, subfolder)
    elif img_filename in val_images:
        subfolder = val_images[img_filename]
        label_dir = os.path.join(LABEL_VAL_DIR, subfolder)
    elif img_filename in test_images:
        subfolder = test_images[img_filename]
        label_dir = os.path.join(LABEL_TEST_DIR, subfolder)
    else:
        print(f"🚫 Image ignorée : {image['file_name']} (pas dans train/val/test)")
        continue

    # Créer le sous-dossier pour les labels s'il n'existe pas encore
    os.makedirs(label_dir, exist_ok=True)

    # Définir le chemin du fichier label
    label_output = os.path.join(label_dir, img_filename.replace(".jpg", ".txt"))

    labels = []
    for annotation in annotations_by_image.get(image_id, []):
        x_min, y_min, width, height = annotation["bbox"]

        # Conversion COCO -> YOLO
        x_center = (x_min + width / 2) / img_w
        y_center = (y_min + height / 2) / img_h
        width = width / img_w
        height = height / img_h
        class_id = annotation["category_id"] - 1  # YOLO commence à 0

        labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Sauvegarde du fichier YOLO
    if labels:
        with open(label_output, "w") as f:
            f.write("\n".join(labels))
        print(f"✅ Fichier créé : {label_output} ({len(labels)} annotations)")
    else:
        print(f"⚠ Aucun objet annoté pour {img_filename}, fichier .txt non créé")

print("✔ Conversion COCO → YOLO terminée avec succès !")
