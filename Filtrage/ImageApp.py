import sys
import numpy as np
import matplotlib
import os

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (
    QApplication,
    QTextEdit,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QFileDialog,
    QHBoxLayout,
    QMessageBox,
    QDialog,
    QComboBox,
    QMainWindow,
    QMenuBar,
)
from PySide6.QtGui import QIcon, QAction, QFont
from PySide6.QtCore import Qt, QThread, Signal
from PIL import Image
from ImageRenderer import ImageRenderer
from YoloHandler import YoloHandler


class YoloWorker(QThread):
    finished = Signal(dict)
    status_update = Signal(str)

    def __init__(self, image_renderer):
        super().__init__()
        self.image_renderer = image_renderer

    def clear_folder_save(self, folder):
        if os.path.exists(folder):
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def save_images(self):
        save_folder = os.path.join(os.path.dirname(__file__), "images_tmp")
        self.clear_folder_save(save_folder)
        os.makedirs(save_folder, exist_ok=True)

        for i, img_array in enumerate(self.image_renderer.images):
            original_name = self.image_renderer.image_names.get(i, f"image_{i}.png")
            img_path = os.path.join(save_folder, original_name)

            img = Image.fromarray(img_array)
            img.save(img_path)

        self.status_update.emit("✅ Images sauvegardées avec succès !")

    def run(self):
        self.save_images()

        self.status_update.emit("=> Conversion en YOLO format...")

        yolo = YoloHandler(image_source="images_tmp", epochs=1)
        yolo.convert_to_yolo()
        yolo.organize_files()
        yolo.create_yaml()
        yolo.train_yolo()

        self.status_update.emit("📊 Évaluation du modèle en cours...")
        metrics = yolo.evaluate_model()

        self.finished.emit(metrics)


class PerformanceDialog(QDialog):
    def __init__(self, image_renderer, parent=None):
        super().__init__(parent)
        self.image_renderer = image_renderer
        self.setWindowTitle("Performance du résultat")
        self.setGeometry(200, 200, 400, 200)

        self.status_label = QLabel("📤 Sauvegarde des images...")
        self.metrics_label = QLabel("📊 En attente de résultats...")

        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(self.metrics_label)
        self.setLayout(layout)

        self.worker = YoloWorker(self.image_renderer)
        self.worker.status_update.connect(self.update_status)
        self.worker.finished.connect(
            self.display_metrics
        )  # Affiche les résultats finaux

        self.worker.start()

    def update_status(self, message):
        self.status_label.setText(message)

    def display_metrics(self, metrics):
        result_text = (
            f"📊 Résultats YOLOv8\n"
            f"=> mAP50: {metrics['mAP50']:.4f}\n"
            f"=> mAP50-95: {metrics['mAP50-95']:.4f}\n"
            f"=> Précision: {metrics['Precision']:.4f}\n"
            f"=> Rappel: {metrics['Recall']:.4f}"
        )

        self.metrics_label.setText(result_text)
        self.status_label.setText("✅ Évaluation terminée !")


class FilterDialog(QDialog):
    def __init__(self, image_renderer, parent=None):
        super().__init__(parent)
        self.image_renderer = image_renderer
        self.setWindowTitle("Appliquer un Filtre")
        self.setGeometry(200, 200, 300, 150)

        self.filter_combo = QComboBox(self)
        self.filter_combo.addItems(self.image_renderer.getFiltres().keys())

        self.apply_button = QPushButton("Appliquer", self)
        self.apply_button.clicked.connect(self.apply_filter)

        layout = QVBoxLayout()
        layout.addWidget(self.filter_combo)
        layout.addWidget(self.apply_button)

        self.setLayout(layout)

    def apply_filter(self):
        selected_filter = self.filter_combo.currentText()
        if selected_filter:
            self.image_renderer.call_filter(selected_filter)
            self.accept()


class InfoDialog(QDialog):
    def __init__(self, info_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Informations sur l'ImageRenderer")
        self.setGeometry(300, 300, 500, 300)

        self.text_area = QTextEdit(self)
        self.text_area.setReadOnly(True)

        font = QFont("DejaVu Sans", 10)
        self.text_area.setFont(font)

        formatted_text = info_text.replace("\n", "<br>")
        self.text_area.setHtml(f"<pre>{formatted_text}</pre>")

        self.close_button = QPushButton("Fermer", self)
        self.close_button.clicked.connect(self.close)

        layout = QVBoxLayout()
        layout.addWidget(self.text_area)
        layout.addWidget(self.close_button)
        self.setLayout(layout)


class ImageApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.image_renderer = ImageRenderer()
        self.current_index = 0
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.create_menu()
        self.setMenuBar(self.menuBar())

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()
        self.central_widget.setLayout(layout)
        self.update_menu_state()

    def create_menu(self):
        self.menuBar().setNativeMenuBar(False)
        menubar = self.menuBar()

        file_menu = menubar.addMenu("Actions")

        self.save_action = QAction(
            QIcon("Ressources/icones/sauvegarder.png"), "Enregistrer les images", self
        )
        self.save_action.triggered.connect(self.save_images)
        file_menu.addAction(self.save_action)

        self.load_action = QAction(
            QIcon("Ressources/icones/importer.png"), "Charger les images", self
        )
        self.load_action.triggered.connect(self.load_images)
        file_menu.addAction(self.load_action)

        self.info_action = QAction(
            QIcon("Ressources/icones/information.png"), "Informations", self
        )
        self.info_action.triggered.connect(self.show_info)
        file_menu.addAction(self.info_action)

        self.performance_action = QAction(
            QIcon("Ressources/icones/performance.png"), "Performance", self
        )
        self.performance_action.triggered.connect(self.show_performance)
        file_menu.addAction(self.performance_action)

        self.contraste_action = QAction(
            QIcon("Ressources/icones/contraste.png"), "Normaliser le contraste", self
        )
        self.contraste_action.triggered.connect(self.handle_normalize)
        file_menu.addAction(self.contraste_action)

        self.setMenuBar(menubar)

    def update_menu_state(self):
        has_images = (
            self.image_renderer.images is not None
            and len(self.image_renderer.images) > 0
        )

        self.save_action.setEnabled(has_images)
        self.info_action.setEnabled(has_images)
        self.performance_action.setEnabled(has_images)
        self.contraste_action.setEnabled(has_images)

    def load_images(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Sélectionner un dossier contenant des images"
        )
        if folder:
            self.image_renderer.load_images(folder)
            self.current_index = 0
            self.setup_after_load()
            self.show_image()
            self.update_menu_state()

    def setup_after_load(self):
        for i in reversed(range(self.central_widget.layout().count())):
            widget = self.central_widget.layout().itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)

        self.canvas = FigureCanvas(plt.figure())
        self.filter_button = QPushButton("Appliquer Filtre")
        self.filter_button.clicked.connect(self.open_filter_dialog)

        self.prev_button = QPushButton("Précédente")
        self.prev_button.clicked.connect(self.show_previous_image)

        self.next_button = QPushButton("Suivante")
        self.next_button.clicked.connect(self.show_next_image)

        nav_layout = QHBoxLayout()
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)

        layout = self.central_widget.layout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.filter_button)
        layout.addLayout(nav_layout)

    def show_image(self):
        if (
            self.image_renderer.images is not None
            and len(self.image_renderer.images) > 0
        ):
            image_array = self.image_renderer.images[self.current_index]

            self.canvas.figure.clear()
            ax = self.canvas.figure.add_subplot(111)
            ax.imshow(image_array, cmap="gray")
            ax.set_title(
                f"Image {self.current_index + 1} / {len(self.image_renderer.images)}"
            )
            ax.axis("off")
            self.canvas.draw()

    def show_next_image(self):
        if (
            self.image_renderer.images is not None
            and self.current_index < len(self.image_renderer.images) - 1
        ):
            self.current_index += 1
            self.show_image()

    def show_previous_image(self):
        if self.image_renderer.images is not None and self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def open_filter_dialog(self):
        if self.image_renderer.images is not None:
            dialog = FilterDialog(self.image_renderer, self)
            if dialog.exec():
                self.show_image()

    def save_images(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Sélectionner un dossier pour enregistrer les images"
        )
        if folder:
            for i, img_array in enumerate(self.image_renderer.images):
                img = Image.fromarray(img_array)
                img.save(f"{folder}/image_{i}.png")
            QMessageBox.information(
                self,
                "Enregistrement réussi",
                "Les images ont été enregistrées avec succès !",
            )

    def show_info(self):
        info_text = str(self.image_renderer)
        dialog = InfoDialog(info_text, self)
        dialog.exec()

    def show_performance(self):
        dialog = PerformanceDialog(self.image_renderer, self)
        dialog.exec()

    def handle_normalize(self):
        if self.image_renderer.images is not None:
            self.image_renderer.normalize_contrast()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageApp()
    window.show()
    sys.exit(app.exec())
