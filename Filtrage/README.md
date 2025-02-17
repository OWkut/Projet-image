# 🖼️ ImageRenderer - Traitement et Filtrage d'Images en Python

## **📌 Description**
`ImageRenderer` est une classe Python permettant de **charger, afficher et appliquer des filtres** sur des images.  
Elle est conçue pour être modulaire et extensible, facilitant l'ajout de nouveaux filtres.

---

## **🚀 Fonctionnalités**
✅ **Chargement d'images** depuis un dossier.  
✅ **Génération d'images aléatoires** en niveaux de gris ou binaires.  
✅ **Affichage des images** (individuelles ou en grille).  
✅ **Application de filtres morphologiques et transformations** :  
   - `Érosion`
   - `Dilatation`
   - `Négatif`
✅ **Ajout de nouveaux filtres facilement**.  
✅ **Affichage des informations sur les images et les filtres appliqués**.  

---

## **📂 Structure du projet**
```
📦 ImageRenderer
├── 📜 ImageRenderer.py   # Classe principale
├── 📜 Filtre.py          # Filtres d'images (erosion, dilatation, negatif, etc.)
├── 📁 Ressources         # Dossier contenant des images pour les tests
└── 📜 README.md          # Documentation du projet
```

---

## **📦 Installation**
### **1️⃣ Prérequis**
- Python **3.7+**
- Bibliothèques nécessaires :
  ```bash
  pip install numpy matplotlib pillow
  ```
---

## **📜 Utilisation**
### **🔹 1️⃣ Importation et instanciation**
```python
from ImageRenderer import ImageRenderer

# Créer un objet ImageRenderer
renderer = ImageRenderer()
```

---

### **🔹 2️⃣ Chargement des images**
```python
renderer.load_images("Ressources")  # Charger toutes les images d'un dossier
print(renderer)  # Voir les détails de l'objet
```

---

### **🔹 3️⃣ Génération d'images aléatoires**
```python
renderer.generate_images(5, mode="gris")  # Générer 5 images en niveaux de gris
renderer.renderImages()  # Afficher les images générées
```

---

### **🔹 4️⃣ Application de filtres**
```python
# Appliquer un filtre unique
renderer.call_filter("negatif")

# Appliquer plusieurs filtres en séquence
renderer.call_filters([
    ("erosion", (), {}),
    ("dilatation", (), {})
])

# Afficher les images après les filtres
renderer.renderImages()
```

---

## **🛠️ Ajouter un Nouveau Filtre**
Vous pouvez ajouter de nouveaux filtres en **une seule ligne** !  
Exemple : Ajouter un filtre "flou" :
```python
renderer.add_filter("flou", flt.flou)
renderer.call_filter("flou")
```

>[!DANGER] ATTENTION: le nom flt.nom doit être le même que celui dans le fichier Filtre sans le flt bien-sûre
---

## **📌 Méthodes Principales**
| 📌 **Méthode**            | 🎯 **Description** |
|---------------------------|-------------------|
| `load_images(path)`       | Charge les images d'un dossier. |
| `generate_images(taille, mode)` | Génère des images binaires ou en niveaux de gris. |
| `renderImage(index)`      | Affiche une image spécifique. |
| `renderImages()`          | Affiche toutes les images chargées. |
| `add_filter(nom, fct)`    | Ajoute un filtre personnalisé. |
| `call_filter(nom, *args, **kwargs)` | Applique un filtre sur les images. |
| `call_filters(filtres)`   | Applique plusieurs filtres en séquence. |

---

## **📜 Exemple d'Affichage**
Lorsque vous utilisez `print(renderer)`, vous obtenez un résumé des images et des filtres appliqués :
```
🖼️ ImageRenderer
━━━━━━━━━━━━━━━━━━━━━━━━━━━
📏 => Taille des images : 512 x 512 pixels
📸 => Nombre d'images   : 5
🎨 => Filtres disponibles : 3 filtres
🛠️ => Liste des filtres  : negatif, dilatation, erosion
🔥 => Filtres appliqués : erosion, negatif
━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## **🔗 Contributions**
📢 **Vous voulez contribuer ?**  
- Ajoutez de nouveaux filtres !
- Optimisez le code
- Améliorez l'affichage des images

---

## **📜 Licence**
📝 Ce projet est sous licence **MIT**.

---

## **💡 Remerciements**
Merci MAEL
