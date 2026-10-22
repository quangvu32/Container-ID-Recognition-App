import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog, QHBoxLayout, QVBoxLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np

from main import main


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Container Recognition App")
        self.setFixedSize(1200, 600)

        self.file_path = None

        # Layouts
        main_layout = QHBoxLayout(self)

        # Left Panel (buttons)
        left_panel = QVBoxLayout()
        self.btn_load = QPushButton("Load Image")
        self.btn_load.setFixedHeight(40)
        self.btn_load.clicked.connect(self.browse_image)

        self.btn_process = QPushButton("Process Image")
        self.btn_process.setFixedHeight(40)
        self.btn_process.clicked.connect(self.process_image)

        left_panel.addWidget(self.btn_load)
        left_panel.addWidget(self.btn_process)
        left_panel.addStretch()

        # Right Panel (images)
        right_panel = QHBoxLayout()

        self.label_img1 = QLabel("No image loaded")
        self.label_img1.setAlignment(Qt.AlignCenter)
        self.label_img1.setFixedSize(500, 500)
        self.label_img1.setStyleSheet("border: 1px solid black;")

        self.label_img2 = QLabel("Processed image will appear here")
        self.label_img2.setAlignment(Qt.AlignCenter)
        self.label_img2.setFixedSize(500, 500)
        self.label_img2.setStyleSheet("border: 1px solid black;")

        right_panel.addWidget(self.label_img1)
        right_panel.addWidget(self.label_img2)

        # Add to main layout
        main_layout.addLayout(left_panel)
        main_layout.addLayout(right_panel)

    def browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if file_path:
            self.file_path = file_path
            pixmap = QPixmap(file_path).scaled(500, 600, Qt.KeepAspectRatio)
            self.label_img1.setPixmap(pixmap)
            self.label_img1.setText("")

    def process_image(self):
        if not self.file_path:
            QMessageBox.warning(self, "Warning", "No image selected. Please load an image first.")
            return

        print(f"Processing image: {self.file_path}")
        result_dict, processed_img, char_imgs = main(self.file_path)
        print("Recognition Result:", result_dict)

        # Convert numpy (processed_img) → QPixmap
        if isinstance(processed_img, np.ndarray):
            h, w = processed_img.shape[:2]
            if len(processed_img.shape) == 2:  # grayscale
                qimage = QImage(processed_img.data, w, h, w, QImage.Format_Grayscale8)
            else:  # color
                qimage = QImage(processed_img.data, w, h, 3 * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage).scaled(500, 600, Qt.KeepAspectRatio)
            self.label_img2.setPixmap(pixmap)
            self.label_img2.setText("")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
