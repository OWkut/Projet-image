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

# Charger le fichier COCO
with open("annotations.json", "r") as f:
    coco_data = json.load(f)

# Dossier où seront stockées les annotations YOLO
output_dir = "labels/"
os.makedirs(output_dir, exist_ok=True)

# Charger les classes (catégories COCO)
categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

# Traitement de chaque image
for image in coco_data["images"]:
    image_id = image["id"]
    img_w, img_h = image["width"], image["height"]
    
    # Nom du fichier annotation YOLO correspondant
    txt_filename = os.path.join(output_dir, image["file_name"].replace(".jpg", ".txt"))
    
    labels = []
    for annotation in coco_data["annotations"]:
        if annotation["image_id"] == image_id:
            # Extraire bbox COCO
            x_min, y_min, width, height = annotation["bbox"]
            
            # Conversion en format YOLO
            x_center = (x_min + width / 2) / img_w
            y_center = (y_min + height / 2) / img_h
            width = width / img_w
            height = height / img_h
            
            # Ajuster class_id pour YOLO (YOLO commence à 0)
            class_id = annotation["category_id"] - 1  
            
            labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # Sauvegarder le fichier YOLO
    with open(txt_filename, "w") as f:
        f.write("\n".join(labels))
