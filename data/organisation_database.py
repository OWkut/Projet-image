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

i = 0
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
                dest_dir = os.path.join(base_dir, str(annotation["category_id"]))
                os.makedirs(dest_dir, exist_ok=True)  # Crée le répertoire si nécessaire
                dest_path = os.path.join(dest_dir, f"{i}{ext}")
                
                # Déplacer l'image au lieu de copier puis supprimer
                shutil.move(src_path, dest_path)
                image["file_name"] = os.path.join(str(annotation["category_id"]), f"{i}{ext}")
                break  # Évite de boucler plusieurs fois pour une même image
    
    i += 1

# Sauvegarde des modifications dans le fichier JSON de sortie
with open("data/annot.json", "w") as f:
    json.dump(coco_data, f, indent=4)

print("Organisation des images terminée.")