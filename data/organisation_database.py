import json
import os
import shutil

for id in range (60):
    os.makedirs(str(id), exist_ok=True)

with open("annotations_augment.json", "r") as f:
    coco_data = json.load(f)

for image in coco_data["images"]:
    id_img = image["id"]
    file_name = image["file_name"]

    if os.path.exists(file_name):
        ext = os.path.splitext(file_name)[1]

        for annotation in coco_data["annotations"]:
            if annotation["image_id"] == id_img:
                dest = os.path.join(str(annotation["category_id"]), f"{id_img}{ext}")
                shutil.copy2(file_name, dest)
                image["file_name"] = dest
        os.remove(file_name)

with open("annotations_final.json", "w") as f:
    json.dump(coco_data, f, indent=4)