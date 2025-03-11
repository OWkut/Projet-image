import os
import numpy as np
from PIL import Image
import re
import json
import os
import shutil
# Attention à bien se mettre dans DATA pour exécuter

# Dossier contenant les images
base_dir = "data"

# Paramètres de translation (décalage en pixels)
tx, ty = 300, 300  # Translation de 30 pixels en x et y

with open("annotations.json", "r") as f:
    coco_data = json.load(f)

for image in coco_data["images"]:
    img_id = image["id"]
for annot in coco_data["annotations"]:
    annot_id = annot["id"]

for image in coco_data["images"][:]:
    id = image["id"]
    file_name = image["file_name"]

    if os.path.exists(file_name):
        dir_name = os.path.dirname(file_name)
        base_name, ext = os.path.splitext(os.path.basename(file_name))

        img_pil = Image.open(file_name)
        width, height = img_pil.size
        transformations = {
            "rot_90": img_pil.rotate(90, expand=True),
            "rot_180": img_pil.rotate(180, expand=True),
            "rot_270": img_pil.rotate(270, expand=True),
        }

        for suffix, transformed_img in transformations.items():
            dest = os.path.join(dir_name, f"{base_name}_{suffix}{ext}")
            transformed_img.save(dest)

            img_id += 1
            new_image = {
                "id": img_id,
                "width": transformed_img.size[0],
                "height": transformed_img.size[1],
                "file_name": dest,
                "license": "CC",
                "flickr_url": None,
                "flickr_640_url": None,
                "coco_url": None,
                "date_captured": None
            }
            coco_data["images"].append(new_image)

            for annotation in coco_data["annotations"]:
                if annotation["image_id"] == id:

                    new_segmentation = []
                    for segment in annotation["segmentation"]:
                        new_segment = []
                        for i in range(0, len(segment), 2):
                            x, y = segment[i], segment[i+1]
                            new_segment.extend([y, width - x])
                        new_segmentation.append(new_segment)
                    
                    x, y, w, h = annotation["bbox"]
                    
                    new_annot = {
                        "id": annot_id,
                        "image_id": img_id,
                        "category_id": annotation["category_id"],
                        "segmentation": new_segmentation,
                        "area": annotation["area"],
                        "bbox": [y, width-(x + w), h, w],
                        "iscrowd": 0
                    }
                    coco_data["annotations"].append(new_annot)

with open("annotations_augment.json", "w") as f:
    json.dump(coco_data, f, indent=4)

print("Texte ajouté avec succès !")