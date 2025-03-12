# ============================================================
# YOLO HANDLER - Gestion et Entraînement de Modèles YOLOv8
# ============================================================
# 📌 Description :
# `YoloHandler` est une classe qui facilite la gestion des données
# pour l'entraînement de modèles YOLOv8. Elle permet :
#   ✅ Chargement et filtrage des annotations COCO depuis un JSON
#   ✅ Organisation des images et labels en `train/val/test`
#   ✅ Conversion des annotations COCO vers le format YOLOv8
#   ✅ Génération d’un fichier `data.yaml` compatible avec YOLO
#   ✅ Entraînement YOLOv8 avec des paramètres dynamiques
#   ✅ Prédiction et évaluation du modèle YOLOv8
#
# 🛠️ Fonctionnalités principales :
#   - 💾 Extraction des images et annotations filtrées
#   - 📂 Organisation automatique des fichiers (`train/val/test`)
#   - 🔄 Conversion COCO → YOLO Format
#   - 📊 Création automatique du fichier `data.yaml`
#   - 🚀 Entraînement personnalisé du modèle YOLOv8
#   - 🎯 Prédiction et évaluation des performances

# ============================================================
# 🔥 Auteur : GOAREGUER Mael
# 📂 Version : 1.0.0
# ============================================================


import os
import json
import shutil
import re
import yaml
import random
from ultralytics import YOLO

"""
==================== VARIABLES PERSONNALISABLE ====================
BASE_DIR => Répertoire du script
ANNOTATION_FILE => Nom du fichier JSON
BATCH_PREFIX => Préfixe des images à utiliser (ex: batch_1/)
DATASET_DIR => Dossier du dataset YOLOv8
IMAGE_SOURCE => Dossier source des images
YOLO_MODEL => Modèle YOLOv8 (s, n, m, l, x)
EPOCHS => Nombre d'époques pour l'entraînement
IMG_SIZE => Taille des images (multiples de 32)
BATCH_SIZE => Taille du batch (ajuster selon GPU)
DEVICE => Utilisation de GPU ou CPU ("cuda" ou "cpu")
===================================================================
"""


class YoloHandler:
    def __init__(
        self,
        dataset_dir="dataset",
        image_source="images",
        annotation_file="annotations.json",
        batch_prefix="batch_1/",
        yolo_model="yolov8s.pt",
        epochs=50,
        img_size=640,
        batch_size=16,
        device="cpu",
    ):
        self.BASE_DIR = os.path.dirname(__file__)
        self.DATASET_DIR = dataset_dir
        self.IMAGE_SOURCE = image_source
        self.ANNOTATION_FILE = annotation_file
        self.BATCH_PREFIX = batch_prefix
        self.YOLO_MODEL = yolo_model
        self.EPOCHS = epochs
        self.IMG_SIZE = img_size
        self.BATCH_SIZE = batch_size
        self.DEVICE = device

        # ✅ Chargement des annotations
        self.data = self.load_annotations()
        self.images = self.extract_images()
        self.categories = self.extract_categories()
        self.annotations = self.extract_annotations()

        print(f"[INIT] ✅ Dataset : {self.DATASET_DIR}, Images : {self.IMAGE_SOURCE}")
        print(f"[INFO] Images trouvées: {len(self.images)}")
        print(f"[INFO] Annotations trouvées: {len(self.annotations)}")

        self.model = YOLO(self.YOLO_MODEL)

    def load_annotations(self):
        """Charge le fichier JSON contenant les annotations"""
        json_path = os.path.join(self.BASE_DIR, self.ANNOTATION_FILE)
        try:
            with open(json_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"[ERREUR] Impossible de charger {self.ANNOTATION_FILE} : {e}")
            return {}

    def extract_images(self):
        """Récupère uniquement les images correspondant au batch défini"""
        return {
            img["id"]: {
                "width": img["width"],
                "height": img["height"],
                "file_name": img["file_name"],
            }
            for img in self.data.get("images", [])
            if img.get("file_name", "").startswith(self.BATCH_PREFIX)
        }

    def extract_categories(self):
        """Récupère les catégories de l'annotation"""
        return {cat["id"]: cat["name"] for cat in self.data.get("categories", [])}

    def extract_annotations(self):
        """Récupère les annotations associées aux images sélectionnées"""
        image_ids = set(self.images.keys())
        return [
            {
                "image_id": ann["image_id"],
                "segmentation": ann["segmentation"],
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
            }
            for ann in self.data.get("annotations", [])
            if ann["image_id"] in image_ids
        ]

    def info_structure(self, data=None, indent=0):
        """Affiche la structure du JSON pour analyse"""
        if data is None:
            data = self.data

        if isinstance(data, dict):
            for key, value in data.items():
                print(" " * indent + f"{key}: {type(value).__name__}")
                if isinstance(value, (dict, list)):
                    self.info_structure(value, indent + 2)
        elif isinstance(data, list) and len(data) > 0:
            print(" " * indent + f"Liste[{len(data)}] -> {type(data[0]).__name__}")
            self.info_structure(data[0], indent + 2)

    def organize_files(self):
        """Organise les images et labels dans dataset/ pour YOLOv8 avec 80% train, 10% val, 10% test"""
        img_dirs = {
            "train": os.path.join(self.DATASET_DIR, "images", "train"),
            "val": os.path.join(self.DATASET_DIR, "images", "val"),
            "test": os.path.join(self.DATASET_DIR, "images", "test"),
        }

        lbl_dirs = {
            "train": os.path.join(self.DATASET_DIR, "labels", "train"),
            "val": os.path.join(self.DATASET_DIR, "labels", "val"),
            "test": os.path.join(self.DATASET_DIR, "labels", "test"),
        }

        for folder in img_dirs.values():
            os.makedirs(folder, exist_ok=True)
        for folder in lbl_dirs.values():
            os.makedirs(folder, exist_ok=True)

        images_list = list(self.images.values())
        random.shuffle(images_list)

        total_images = len(images_list)
        train_split = int(total_images * 0.8)
        val_split = train_split + int(total_images * 0.1)

        train_images = images_list[:train_split]
        val_images = images_list[train_split:val_split]
        test_images = images_list[val_split:]

        def move_files(images, img_dest, lbl_dest):
            for img in images:
                img_name = os.path.basename(img["file_name"])
                src_img = os.path.join(self.BASE_DIR, self.IMAGE_SOURCE, img_name)
                dest_img = os.path.join(img_dest, img_name)

                if os.path.exists(src_img):
                    shutil.copy(src_img, dest_img)
                else:
                    print(f"[ERREUR] Image introuvable : {src_img}")

                label_file = os.path.splitext(img_name)[0] + ".txt"
                src_label = os.path.join("labels", label_file)
                dest_label = os.path.join(lbl_dest, label_file)

                if os.path.exists(src_label):
                    shutil.move(src_label, dest_label)
                else:
                    print(f"[ERREUR] Label introuvable : {src_label}")

        move_files(train_images, img_dirs["train"], lbl_dirs["train"])
        move_files(val_images, img_dirs["val"], lbl_dirs["val"])
        move_files(test_images, img_dirs["test"], lbl_dirs["test"])

        print(
            f"[ORGANIZE_FILES] ✅ Train: {len(train_images)} images, Val: {len(val_images)}, Test: {len(test_images)}"
        )

    def convert_to_yolo(self, output_dir="labels"):
        """Convertir les annotations en format YOLO et les enregistrer directement dans labels/"""
        os.makedirs(output_dir, exist_ok=True)

        for img_id, img_data in self.images.items():
            img_width, img_height = img_data["width"], img_data["height"]
            file_name = (
                os.path.splitext(os.path.basename(img_data["file_name"]))[0] + ".txt"
            )
            label_path = os.path.join(output_dir, file_name)

            with open(label_path, "w") as f:
                for ann in self.annotations:
                    if ann["image_id"] == img_id:
                        x_min, y_min, width, height = ann["bbox"]
                        x_center = (x_min + width / 2) / img_width
                        y_center = (y_min + height / 2) / img_height
                        width /= img_width
                        height /= img_height
                        f.write(
                            f"{ann['category_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                        )

        print(f"[CONVERT_TO_YOLO] ✅ Annotations enregistrées dans {output_dir}/")

    def create_yaml(self):
        """Créer un fichier data.yaml pour YOLOv8 en ajoutant train, val et test"""
        yaml_path = os.path.join(self.DATASET_DIR, "data.yaml")
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        yaml_data = {
            "train": os.path.abspath(os.path.join(self.DATASET_DIR, "images/train")),
            "val": os.path.abspath(os.path.join(self.DATASET_DIR, "images/val")),
            "test": os.path.abspath(os.path.join(self.DATASET_DIR, "images/test")),
            "nc": len(self.categories),
            "names": list(self.categories.values()),
        }

        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)

        print(f"[CREATE_YAML] ✅ Fichier YAML créé avec chemins absolus : {yaml_path}")

    def train_yolo(self):
        """Entraîner YOLOv8 avec les paramètres globaux"""
        self.model.train(
            data=os.path.join(self.DATASET_DIR, "data.yaml"),
            epochs=self.EPOCHS,
            imgsz=self.IMG_SIZE,
            batch=self.BATCH_SIZE,
            device=self.DEVICE,
        )
        print("[TRAIN_YOLO] ✅ Entraînement YOLOv8 terminé !")

    def predict(self, image_path, save_results=True, conf=0.25):
        """
        Effectue une prédiction sur une image en utilisant YOLOv8.

        @Param:
            => image_path: str -> Chemin vers l'image à tester
            => save_results: bool -> Sauvegarde les résultats si True
            => conf: float -> Seuil de confiance pour la détection (par défaut: 0.25)
        """
        results = self.model(image_path, conf=conf, save=save_results)

        print(f"[PREDICT] ✅ Prédiction effectuée sur {image_path}")

        return results

    def evaluate_model(self):
        """
        Évalue les performances du modèle YOLOv8 sur l'ensemble de validation.

        Retourne:
            => dictionnaire contenant les métriques (mAP, précision, rappel)
        """
        metrics = self.model.val(data=os.path.join(self.DATASET_DIR, "data.yaml"))

        print("[EVALUATE_MODEL] ✅ Évaluation terminée !")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Précision: {metrics.box.mp:.4f}")
        print(f"Rappel: {metrics.box.mr:.4f}")

        return {
            "mAP50": metrics.box.map50,
            "mAP50-95": metrics.box.map,
            "Precision": metrics.box.mp,
            "Recall": metrics.box.mr,
        }

    # METHODE DE VERIF
    def validate_yolo_labels(self, labels_dir="labels"):
        """Vérifie que les fichiers .txt YOLO respectent le bon format avec Regex."""
        yolo_regex = re.compile(r"^\d+(\s\d*\.\d+){4}$")

        all_valid = True

        for label_file in os.listdir(labels_dir):
            if label_file.endswith(".txt"):
                with open(os.path.join(labels_dir, label_file), "r") as f:
                    for line in f:
                        if not yolo_regex.match(line.strip()):
                            print(
                                f"[ERREUR] Format incorrect dans {label_file}: {line.strip()}"
                            )
                            all_valid = False

        if all_valid:
            print("[VALIDATE_YOLO_LABELS] ✅ Tous les fichiers sont corrects !")
        return all_valid

    def validate_organized_files(self):
        """Vérifie que toutes les images et annotations sont bien organisées dans train, val et test."""
        img_dirs = {
            "train": os.path.join(self.DATASET_DIR, "images", "train"),
            "val": os.path.join(self.DATASET_DIR, "images", "val"),
            "test": os.path.join(self.DATASET_DIR, "images", "test"),
        }

        lbl_dirs = {
            "train": os.path.join(self.DATASET_DIR, "labels", "train"),
            "val": os.path.join(self.DATASET_DIR, "labels", "val"),
            "test": os.path.join(self.DATASET_DIR, "labels", "test"),
        }

        missing_images, missing_labels = [], []

        for img in self.images.values():
            img_name = os.path.basename(img["file_name"])
            label_name = os.path.splitext(img_name)[0] + ".txt"

            found_image = any(
                os.path.exists(os.path.join(img_dirs[split], img_name))
                for split in img_dirs
            )
            found_label = any(
                os.path.exists(os.path.join(lbl_dirs[split], label_name))
                for split in lbl_dirs
            )

            if not found_image:
                missing_images.append(img_name)

            if not found_label:
                missing_labels.append(label_name)

        if missing_images:
            print(
                f"[ERREUR] Images manquantes dans tous les dossiers : {missing_images}"
            )
            return False
        if missing_labels:
            print(
                f"[ERREUR] Labels manquants dans tous les dossiers : {missing_labels}"
            )
            return False

        print("[VALIDATE_ORGANIZED_FILES] ✅ Tout est bien organisé !")
        return True

    def validate_yaml(self):
        """Vérifie la syntaxe et le contenu du fichier data.yaml."""
        yaml_path = os.path.join(self.DATASET_DIR, "data.yaml")

        try:
            with open(yaml_path, "r") as f:
                yaml_content = yaml.safe_load(f)

            required_keys = ["train", "val", "test", "nc", "names"]
            for key in required_keys:
                if key not in yaml_content:
                    print(f"[ERREUR] Clé manquante dans {yaml_path}: {key}")
                    return False

            for key in ["train", "val", "test"]:
                if not os.path.exists(yaml_content[key]):
                    print(
                        f"[ERREUR] Le chemin spécifié dans {key} n'existe pas: {yaml_content[key]}"
                    )
                    return False

            print("[VALIDATE_YAML] ✅ Le fichier YAML est valide !")
            return True

        except yaml.YAMLError as e:
            print(f"[ERREUR] Problème de syntaxe dans {yaml_path} : {e}")
            return False
        except FileNotFoundError:
            print(f"[ERREUR] Le fichier {yaml_path} n'existe pas.")
            return False


if __name__ == "__main__":
    yolo = YoloHandler()
    yolo.convert_to_yolo()
    yolo.organize_files()
    yolo.create_yaml()

    # On valide
    yolo.validate_yolo_labels()
    yolo.validate_organized_files()
    yolo.validate_yaml()

    # YOLO
    yolo.train_yolo()
