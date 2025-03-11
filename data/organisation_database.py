import json
import os
import shutil

# Spécifie le répertoire de base 'data' pour les dossiers de 0 à 59
base_dir = "data"

# Crée les dossiers de 0 à 59 dans le répertoire 'data'
for id in range(60):
    os.makedirs(os.path.join(base_dir, str(id)), exist_ok=True)

with open("data/annotations.json", "r") as f:
    coco_data = json.load(f)

# Vérification du nom des images et de leur déplacement
for image in coco_data["images"]:
    id_img = image["id"]
    file_name = image["file_name"]
    src_path = os.path.join("data", file_name)  # Chemin source complet
    
    # Vérifie si le fichier existe avant de le déplacer
    if os.path.exists(src_path):
        ext = os.path.splitext(file_name)[1]
        for annotation in coco_data["annotations"]:
            if annotation["image_id"] == id_img:
                dest = os.path.join(str(annotation["category_id"]), f"{id_img}{ext}")
                shutil.copy(src_path, os.path.join(base_dir, f"{id_img}{ext}"))
                image["file_name"] = dest
        os.remove(file_name)

with open("data/annotations_final.json", "w") as f:
    json.dump(coco_data, f, indent=4)
print("Organisation des images terminée.")
