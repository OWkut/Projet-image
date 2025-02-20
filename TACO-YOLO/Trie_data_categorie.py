import os
import yaml
import json
import shutil

# Définition des chemins
DATA_DIR = "data"
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations.json")
DATASET_YAML = os.path.join("TACO-YOLO", "dataset.yaml")

# Charger les classes depuis dataset.yaml
with open(DATASET_YAML, "r") as f:
    dataset_info = yaml.safe_load(f)

if "names" not in dataset_info:
    raise ValueError("Le fichier dataset.yaml ne contient pas de clé 'names'.")

class_names = dataset_info["names"]  # Liste des classes (ex: {0: "bottle", 1: "plastic_bag", ...})

# Charger les annotations depuis data/annotations.json
with open(ANNOTATIONS_FILE, "r") as f:
    annotations = json.load(f)

# Dictionnaire image → classe
image_classes = {}

for annotation in annotations["annotations"]:
    image_id = annotation["image_id"]
    category_id = annotation["category_id"]

    # Trouver le nom de l'image associée
    image_info = next((img for img in annotations["images"] if img["id"] == image_id), None)
    if not image_info:
        print(f"⚠️ Pas d'info pour image_id {image_id}, ignoré.")
        continue

    image_name = image_info["file_name"]  # ex: "batch_9/000123.jpg"
    image_classes[image_name] = class_names[category_id]  # Associe l'image à sa classe

# Déplacer les images vers les dossiers de classes
for image_name, class_name in image_classes.items():
    source_path = os.path.join(DATA_DIR, image_name)  # L'image est dans data/batch_X/
    dest_folder = os.path.join(DATA_DIR, class_name)  # Nouveau dossier classé
    dest_path = os.path.join(dest_folder, os.path.basename(image_name))  # Garde juste le nom du fichier

    os.makedirs(dest_folder, exist_ok=True)  # Créer le dossier de classe si inexistant

    if os.path.exists(source_path):
        shutil.move(source_path, dest_path)
    else:
        print(f"Image introuvable : {source_path}")

# Supprimer les dossiers batch_1 à batch_15 s'ils sont vides
for i in range(1, 16):
    batch_folder = os.path.join(DATA_DIR, f"batch_{i}")
    if os.path.exists(batch_folder) and not os.listdir(batch_folder):  # Vérifie si le dossier est vide
        os.rmdir(batch_folder)
        print(f"Dossier supprimé : {batch_folder}")
    elif os.path.exists(batch_folder):
        print(f"Dossier {batch_folder} non supprimé (encore des fichiers)")

print("Tri des images par classe terminé et nettoyage effectué !")
