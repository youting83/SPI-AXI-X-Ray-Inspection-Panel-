import platform
import pandas as pd
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from skimage.measure import label
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout,
                             QFileDialog, QSlider, QListWidget, QLineEdit, QTableWidget, QTableWidgetItem)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor
from PyQt5.QtCore import Qt, QPoint, QRect
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import sys
import io
import csv

# Check if estimate_noise is available
try:
    from skimage.restoration import estimate_noise
    NIQE_AVAILABLE = True
except ImportError:
    NIQE_AVAILABLE = False

class SPIAXIImageProcessor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.zoom_center = None
        self.width_map_centers = [None] * 4
        self.image = None
        self.sigma = None
        self.folder_path = None
        self.use_morphology = False
        self.kernel_size = 5  # Default kernel size
        self.selected_defect = None  # For filtering
        self.defect_counts = {'short': 0, 'spur': 0, 'spurious copper': 0, 'missing hole': 0, 'mouse bite': 0, 'open circuit': 0}
        self.initUI()

    def initUI(self):
        self.setWindowTitle('SPI/AXI/X-Ray Inspection Panel')
        self.setGeometry(100, 100, 2700, 900)
        self.setStyleSheet("""
            QMainWindow { background-color: #2E2E2E; color: #E0E0E0; }
            QWidget { border: 1px solid #4A4A4A; background-color: #3A3A3A; border-radius: 5px; padding: 5px; }
            QLabel { color: #E0E0E0; font-family: Arial; font-size: 12px; }
            QPushButton { background-color: #5A5A5A; color: #E0E0E0; border: 1px solid #7A7A7A; border-radius: 5px; padding: 5px; }
            QPushButton:hover { background-color: #6A6A6A; }
            QSlider { background-color: #4A4A4A; }
            QSlider::groove:horizontal { background: #5A5A5A; height: 8px; border-radius: 4px; }
            QSlider::handle:horizontal { background: #9ACD10; width: 12px; margin: -4px 0; border-radius: 6px; }
            QSlider::handle:vertical { background: #9ACD10; width: 16px; }
            QLineEdit { background-color: #4A4A4A; color: #E0E0E0; border: 1px solid #7A7A7A; border-radius: 3px; }
            QListWidget { background-color: #4A4A4A; color: #E0E0E0; border: 1px solid #7A7A7A; }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        main_widget.setLayout(main_layout)

        # Left Panel: Control Section
        control_widget = QWidget()
        control_layout = QVBoxLayout()
        control_layout.setSpacing(5)
        control_widget.setLayout(control_layout)

        header = QLabel("SPI/AXI/X-Ray Inspection Panel")
        header.setStyleSheet("font-weight: bold; color: #9ACD10; font-size: 14px;")
        header.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(header)

        file_path, _ = QFileDialog.getOpenFileName(self, "Load Initial Image", "", "Image Files (*.bmp *.png *.jpg *.jpeg)")
        if not file_path:
            raise ValueError("No image selected. Application will exit.")
        self.image = cv2.imread(file_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {file_path}")
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.sigma = np.zeros(self.image.shape[:2], dtype=float)
        self.sensitivity = 50
        self.threshold = 100
        self.xray_intensity = 75

        sensitivity_slider = QSlider(Qt.Horizontal)
        sensitivity_slider.setRange(0, 100)
        sensitivity_slider.setValue(self.sensitivity)
        self.sensitivity_value = QLabel(f"Sensitivity: {self.sensitivity}")
        self.sensitivity_value.setStyleSheet("color: #9ACD10; font-weight: bold;")
        sensitivity_slider.valueChanged.connect(lambda val: [self.update_sensitivity(val), self.sensitivity_value.setText(f"Sensitivity: {val}")])
        control_layout.addWidget(QLabel("Sensitivity (0-100)"))
        control_layout.addWidget(self.sensitivity_value)
        control_layout.addWidget(sensitivity_slider)

        threshold_slider = QSlider(Qt.Horizontal)
        threshold_slider.setRange(0, 255)
        threshold_slider.setValue(self.threshold)
        self.threshold_value = QLabel(f"Threshold: {self.threshold}")
        self.threshold_value.setStyleSheet("color: #9ACD10; font-weight: bold;")
        threshold_slider.valueChanged.connect(lambda val: [self.update_threshold(val), self.threshold_value.setText(f"Threshold: {val}")])
        control_layout.addWidget(QLabel("Threshold (0-255)"))
        control_layout.addWidget(self.threshold_value)
        control_layout.addWidget(threshold_slider)

        xray_slider = QSlider(Qt.Horizontal)
        xray_slider.setRange(0, 100)
        xray_slider.setValue(self.xray_intensity)
        self.xray_value = QLabel(f"X-Ray Intensity: {self.xray_intensity}")
        self.xray_value.setStyleSheet("color: #9ACD10; font-weight: bold;")
        xray_slider.valueChanged.connect(lambda val: [self.update_xray_intensity(val), self.xray_value.setText(f"X-Ray Intensity: {val}")])
        control_layout.addWidget(QLabel("X-Ray Intensity (0-100)"))
        control_layout.addWidget(self.xray_value)
        control_layout.addWidget(xray_slider)

        morphology_slider = QSlider(Qt.Horizontal)
        morphology_slider.setRange(0, 1)
        morphology_slider.setValue(0)
        self.morphology_value = QLabel("Morphology: Off")
        self.morphology_value.setStyleSheet("color: #9ACD10; font-weight: bold;")
        morphology_slider.valueChanged.connect(lambda val: [self.toggle_morphology(val), self.morphology_value.setText(f"Morphology: {'On' if val else 'Off'}")])
        control_layout.addWidget(QLabel("Morphology (0: Off, 1: On)"))
        control_layout.addWidget(self.morphology_value)
        control_layout.addWidget(morphology_slider)

        kernel_slider = QSlider(Qt.Horizontal)
        kernel_slider.setRange(0, 3)  # 0: 3x3, 1: 5x5, 2: 7x7, 3: 10x10
        kernel_slider.setValue(1)  # Default to 5x5
        self.kernel_value = QLabel("Kernel Size: 5x5")
        self.kernel_value.setStyleSheet("color: #9ACD10; font-weight: bold;")
        kernel_slider.valueChanged.connect(lambda val: [self.update_kernel_size(val), self.kernel_value.setText(f"Kernel Size: {self.get_kernel_size_label(val)}x{self.get_kernel_size_label(val)}")])
        control_layout.addWidget(QLabel("Kernel Size (0: 3x3, 1: 5x5, 2: 7x7, 3: 10x10)"))
        control_layout.addWidget(self.kernel_value)
        control_layout.addWidget(kernel_slider)

        suggest_button = QPushButton('Suggest Parameters')
        suggest_button.setStyleSheet("font-weight: bold;")
        suggest_button.clicked.connect(self.suggest_parameters)
        control_layout.addWidget(suggest_button)

        save_button = QPushButton('Save Results')
        save_button.setStyleSheet("font-weight: bold;")
        save_button.clicked.connect(self.save_results)
        control_layout.addWidget(save_button)

        load_folder_button = QPushButton('Load Folder')
        load_folder_button.setStyleSheet("font-weight: bold;")
        load_folder_button.clicked.connect(self.load_folder)
        control_layout.addWidget(load_folder_button)

        batch_button = QPushButton('Run Batch Process')
        batch_button.setStyleSheet("font-weight: bold;")
        batch_button.clicked.connect(self.open_batch_folder)
        control_layout.addWidget(batch_button)

        control_layout.addStretch()
        main_layout.addWidget(control_widget, 1)

        # Right Panel: Image and Analysis Section
        analysis_widget = QWidget()
        analysis_layout = QHBoxLayout()
        analysis_layout.setSpacing(10)
        analysis_widget.setLayout(analysis_layout)

        image_widget = QWidget()
        image_layout = QVBoxLayout()
        image_layout.setSpacing(5)
        image_widget.setLayout(image_layout)

        original_title = QLabel("Original PCB Image")
        original_title.setStyleSheet("font-weight: bold; color: #9ACD10;")
        original_title.setAlignment(Qt.AlignCenter)
        self.original_label = QLabel(self)
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 2px solid #7A7A7A;")
        image_layout.addWidget(original_title)
        image_layout.addWidget(self.original_label)

        # Changed from X-Ray View to Threshold Morphology Defect Map
        threshold_morph_title = QLabel("Threshold Morphology Defect Map")
        threshold_morph_title.setStyleSheet("font-weight: bold; color: #9ACD10;")
        threshold_morph_title.setAlignment(Qt.AlignCenter)
        self.threshold_morph_label = QLabel(self)
        self.threshold_morph_label.setAlignment(Qt.AlignCenter)
        self.threshold_morph_label.setStyleSheet("border: 2px solid #7A7A7A;")
        image_layout.addWidget(threshold_morph_title)
        image_layout.addWidget(self.threshold_morph_label)

        defect_title = QLabel("Defect Map (Classified)")
        defect_title.setStyleSheet("font-weight: bold; color: #9ACD10;")
        defect_title.setAlignment(Qt.AlignCenter)
        self.defect_label = QLabel(self)
        self.defect_label.setAlignment(Qt.AlignCenter)
        self.defect_label.setStyleSheet("border: 2px solid #7A7A7A;")
        image_layout.addWidget(defect_title)
        image_layout.addWidget(self.defect_label)

        defect_hist_title = QLabel("Defect Histogram")
        defect_hist_title.setStyleSheet("font-weight: bold; color: #9ACD10;")
        defect_hist_title.setAlignment(Qt.AlignCenter)
        self.defect_hist_label = QLabel(self)
        self.defect_hist_label.setAlignment(Qt.AlignCenter)
        self.defect_hist_label.setStyleSheet("border: 2px solid #7A7A7A;")
        image_layout.addWidget(defect_hist_title)
        image_layout.addWidget(self.defect_hist_label)

        image_layout.addStretch()
        analysis_layout.addWidget(image_widget, 1)

        tools_widget = QWidget()
        tools_layout = QVBoxLayout()
        tools_layout.setSpacing(5)
        tools_widget.setLayout(tools_layout)

        zoom_widget = QWidget()
        zoom_layout = QVBoxLayout()
        zoom_layout.setSpacing(5)
        zoom_widget.setLayout(zoom_layout)
        zoom_title = QLabel("Zoom Control")
        zoom_title.setStyleSheet("font-weight: bold; color: #9ACD10;")
        zoom_title.setAlignment(Qt.AlignCenter)
        zoom_layout.addWidget(zoom_title)
        self.zoom_slider = QSlider(Qt.Vertical)
        self.zoom_slider.setRange(1, 10)
        self.zoom_slider.setValue(1)
        self.zoom_slider.valueChanged.connect(self.update_zoom)
        zoom_layout.addWidget(self.zoom_slider)
        self.zoomed_label = QLabel(self)
        self.zoomed_label.setAlignment(Qt.AlignCenter)
        self.zoomed_label.setStyleSheet("border: 2px solid #7A7A7A;")
        self.zoomed_label.setFixedSize(200, 200)
        zoom_layout.addWidget(self.zoomed_label)
        self.zoom_value = QLabel("Zoom: 1x")
        self.zoom_value.setAlignment(Qt.AlignCenter)
        self.zoom_value.setStyleSheet("color: #9ACD10;")
        zoom_layout.addWidget(self.zoom_value)
        tools_layout.addWidget(zoom_widget)

        self.quality_label = QLabel(self)
        self.quality_label.setAlignment(Qt.AlignCenter)
        self.quality_label.setStyleSheet("border: 2px solid #7A7A7A; background-color: #4A4A4A;")
        tools_layout.addWidget(self.quality_label)

        image_list_widget = QWidget()
        image_list_layout = QVBoxLayout()
        image_list_layout.setSpacing(5)
        image_list_widget.setLayout(image_list_layout)
        image_list_title = QLabel("Image Queue")
        image_list_title.setStyleSheet("font-weight: bold; color: #9ACD10;")
        image_list_title.setAlignment(Qt.AlignCenter)
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.update_image_from_list)
        image_list_layout.addWidget(image_list_title)
        image_list_layout.addWidget(self.image_list)
        tools_layout.addWidget(image_list_widget)

        # Defect Statistics Chart
        stats_widget = QWidget()
        stats_layout = QVBoxLayout()
        stats_layout.setSpacing(5)
        stats_widget.setLayout(stats_layout)
        stats_title = QLabel("Defect Statistics")
        stats_title.setStyleSheet("font-weight: bold; color: #9ACD10;")
        stats_title.setAlignment(Qt.AlignCenter)
        self.stats_label = QLabel(self)
        self.stats_label.setAlignment(Qt.AlignCenter)
        self.stats_label.setStyleSheet("border: 2px solid #7A7A7A;")
        self.stats_label.setFixedSize(350, 250)  # Increased size to accommodate larger chart
        stats_layout.addWidget(stats_title)
        stats_layout.addWidget(self.stats_label)
        tools_layout.addWidget(stats_widget)

        tools_layout.addStretch()
        analysis_layout.addWidget(tools_widget, 1)

        main_layout.addWidget(analysis_widget, 2)

        self.original_label.mousePressEvent = self.set_zoom_center
        self.threshold_morph_label.mousePressEvent = self.set_zoom_center
        self.defect_label.mousePressEvent = self.set_zoom_center
        self.stats_label.mousePressEvent = self.filter_defects

        self.update_display()

    def set_zoom_center(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            self.zoom_center = QPoint(pos.x() * self.image.shape[1] // self.original_label.width(),
                                    pos.y() * self.image.shape[0] // self.original_label.height())
            self.update_zoom(self.zoom_slider.value())
            self.zoom_value.setText(f"Zoom: {self.zoom_slider.value()}x")

    def update_zoom(self, val):
        if self.zoom_center and self.image is not None:
            zoom_factor = val
            height, width = self.image.shape[:2]
            label_width, label_height = 200, 200

            center_x = self.zoom_center.x()
            center_y = self.zoom_center.y()
            half_width = label_width // (2 * zoom_factor)
            half_height = label_height // (2 * zoom_factor)

            x_start = max(0, min(center_x - half_width, width - label_width // zoom_factor))
            y_start = max(0, min(center_y - half_height, height - label_height // zoom_factor))
            x_end = min(width, x_start + label_width // zoom_factor)
            y_end = min(height, y_start + label_height // zoom_factor)

            zoomed_region = self.image[y_start:y_end, x_start:x_end]
            if zoomed_region.size > 0:
                zoomed_region = cv2.resize(zoomed_region, (label_width, label_height),
                                         interpolation=cv2.INTER_LINEAR)
                zoomed_qimage = QImage(zoomed_region.data, zoomed_region.shape[1], zoomed_region.shape[0],
                                      zoomed_region.strides[0], QImage.Format_RGB888)
                self.zoomed_label.setPixmap(QPixmap.fromImage(zoomed_qimage))
            self.zoom_value.setText(f"Zoom: {val}x")

    def update_sensitivity(self, val):
        self.sensitivity = val
        self.update_display()

    def update_threshold(self, val):
        self.threshold = val
        self.update_display()

    def update_xray_intensity(self, val):
        self.xray_intensity = val
        self.update_display()

    def toggle_morphology(self, val):
        self.use_morphology = bool(val)
        self.update_display()

    def update_kernel_size(self, val):
        self.kernel_size = [3, 5, 7, 10][val]  # Map slider value to kernel size
        self.update_display()

    def get_kernel_size_label(self, val):
        return str([3, 5, 7, 10][val])

    def process_image_for_inspection(self):
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        _, binary_image = cv2.threshold(gray_image, self.threshold, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        if self.use_morphology:
            # Apply Opening to remove small noise
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
            # Apply Closing to close small gaps
            binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        # Create threshold morphology defect map (3-channel for display)
        threshold_morph_map = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)

        dilated = cv2.dilate(binary_image, kernel, iterations=1)
        labels = label(dilated, background=0, connectivity=1)
        num_components = labels.max()

        # Create defect map with 4 channels (RGBA) for transparency
        defect_map = np.zeros((*self.image.shape[:2], 4), dtype=np.uint8)
        defect_map[..., :3] = self.image  # Copy RGB channels from the original image as background
        self.defect_counts = {'short': 0, 'spur': 0, 'spurious copper': 0, 'missing hole': 0, 'mouse bite': 0, 'open circuit': 0}

        for i in range(1, num_components + 1):
            component_mask = (labels == i)
            area = np.sum(component_mask)
            contours, _ = cv2.findContours((component_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(defect_map, contours, -1, (255, 0, 0, 255), 2)  # Red border with full opacity

            if area < 30:  # Missing hole (smallest area)
                defect_map[component_mask] = [255, 0, 0, 128]  # Red, semi-transparent
                self.defect_counts['missing hole'] += 1
            elif 30 <= area < 60:  # Mouse bite (small to medium area)
                defect_map[component_mask] = [255, 255, 0, 128]  # Yellow, semi-transparent
                self.defect_counts['mouse bite'] += 1
            elif 60 <= area < 100:  # Open circuit (medium area)
                defect_map[component_mask] = [0, 255, 0, 128]  # Green, semi-transparent
                self.defect_counts['open circuit'] += 1
            elif 100 <= area < 150:  # Spur (medium to large area)
                defect_map[component_mask] = [0, 0, 255, 128]  # Blue, semi-transparent
                self.defect_counts['spur'] += 1
            elif 150 <= area < 200:  # Spurious copper (larger area)
                defect_map[component_mask] = [128, 0, 128, 128]  # Purple, semi-transparent
                self.defect_counts['spurious copper'] += 1
            elif area >= 200:  # Short (largest area)
                defect_map[component_mask] = [0, 255, 255, 128]  # Cyan, semi-transparent
                self.defect_counts['short'] += 1

        # Convert to 3-channel RGB for display by blending with original image
        defect_color_map = defect_map.copy()
        alpha = defect_map[:, :, 3:4] / 255.0
        defect_color_map[:, :, :3] = (1 - alpha) * self.image + alpha * defect_map[:, :, :3]

        # Apply defect filtering if selected
        if self.selected_defect:
            mask = np.zeros_like(defect_color_map[:, :, 0], dtype=bool)
            if self.selected_defect == 'missing hole':
                mask[defect_map[:, :, 0] == 255] = True
            elif self.selected_defect == 'mouse bite':
                mask[defect_map[:, :, 1] == 255] = True
            elif self.selected_defect == 'open circuit':
                mask[defect_map[:, :, 1] == 255] = True
            elif self.selected_defect == 'spur':
                mask[defect_map[:, :, 2] == 255] = True
            elif self.selected_defect == 'spurious copper':
                mask[defect_map[:, :, 0] == 128] = True
            elif self.selected_defect == 'short':
                mask[defect_map[:, :, 1] == 255] = True
            defect_color_map[~mask] = self.image[~mask]

        # Remove alpha channel for final RGB output
        defect_color_map = defect_color_map[:, :, :3].astype(np.uint8)

        return threshold_morph_map, defect_color_map

    def calculate_metrics(self, original, processed):
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        ssim_value = ssim(original_gray, processed_gray)
        psnr_value = psnr(original_gray, processed_gray, data_range=255)
        niqe_value = estimate_noise(processed_gray) if NIQE_AVAILABLE else 0.0
        return f"SSIM: {ssim_value:.4f}\nPSNR: {psnr_value:.2f}dB\nNIQE: {'N/A' if not NIQE_AVAILABLE else f'{niqe_value:.4f}'}"


    def plot_histogram(self, image, title):
        plt.figure(figsize=(6, 3))
        plt.hist(image.ravel(), bins=256, range=(0, 255), color='gray', alpha=0.7)
        plt.title(title)
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        buf.close()
        plt.close()
        return pixmap.scaled(300, 150, Qt.KeepAspectRatio)

    def plot_defect_stats(self):
        labels = list(self.defect_counts.keys())
        sizes = list(self.defect_counts.values())
        if sum(sizes) == 0:
            return QPixmap().scaled(350, 250, Qt.KeepAspectRatio)

        # Dynamically set figure size based on label dimensions, maintaining aspect ratio
        label_width, label_height = 350, 250
        fig_width = label_width / 100  # Convert to inches with a scaling factor
        fig_height = label_height / 100
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        bar_width = 0.4
        x = np.arange(len(labels))
        bars = ax.bar(x, sizes, width=bar_width, color=['cyan', 'yellow', 'green', 'blue', 'purple', 'red'])

        # Add value annotations on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10, color='black', weight='bold')

        # Enhance label readability and center the plot
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=12)
        ax.set_title('Defect Distribution', fontsize=14, pad=10)
        ax.set_xlabel('Defect Type', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)

        # Adjust layout to center and fit within the figure
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.getvalue())
        buf.close()
        plt.close()

        # Scale pixmap proportionally to fit label while maintaining aspect ratio
        pixmap = pixmap.scaled(label_width, label_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return pixmap

    def update_display(self):
        threshold_morph_map, defect_map = self.process_image_for_inspection()

        original_qimage = QImage(self.image.data, self.image.shape[1], self.image.shape[0],
                                 self.image.strides[0], QImage.Format_RGB888)
        self.original_label.setPixmap(QPixmap.fromImage(original_qimage).scaled(500, 500, Qt.KeepAspectRatio))

        # Display Threshold Morphology Defect Map
        threshold_morph_qimage = QImage(threshold_morph_map.data, threshold_morph_map.shape[1], threshold_morph_map.shape[0],
                                       threshold_morph_map.strides[0], QImage.Format_RGB888)
        self.threshold_morph_label.setPixmap(QPixmap.fromImage(threshold_morph_qimage).scaled(500, 500, Qt.KeepAspectRatio))

        defect_qimage = QImage(defect_map.data, defect_map.shape[1], defect_map.shape[0],
                               defect_map.strides[0], QImage.Format_RGB888)
        self.defect_label.setPixmap(QPixmap.fromImage(defect_qimage).scaled(500, 500, Qt.KeepAspectRatio))

        defect_gray = cv2.cvtColor(defect_map, cv2.COLOR_RGB2GRAY)
        self.defect_hist_label.setPixmap(self.plot_histogram(defect_gray, "Defect Histogram"))

        self.stats_label.setPixmap(self.plot_defect_stats())

        quality_text = self.calculate_metrics(self.image, defect_map)  # Compare with defect map instead of x-ray
        self.quality_label.setText(quality_text)

        if self.zoom_center:
            self.update_zoom(self.zoom_slider.value())

    def filter_defects(self, event):
        if event.button() == Qt.LeftButton:
            pos = event.pos()
            labels = list(self.defect_counts.keys())
            width_per_label = 350 / len(labels)  # Updated to match new label size
            for i, label in enumerate(labels):
                if i * width_per_label <= pos.x() < (i + 1) * width_per_label and self.defect_counts[label] > 0:
                    self.selected_defect = label if self.selected_defect != label else None
                    break
            self.update_display()

    def suggest_parameters(self):
        self.sensitivity = 75
        self.threshold = 120
        self.xray_intensity = 80
        self.sensitivity_value.setText(f"Sensitivity: {self.sensitivity}")
        self.threshold_value.setText(f"Threshold: {self.threshold}")
        self.xray_value.setText(f"X-Ray Intensity: {self.xray_intensity}")
        self.update_display()

    def save_results(self):
        threshold_morph_map, defect_map = self.process_image_for_inspection()
        save_dir = "C:/path_to_save/"
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f"{save_dir}threshold_morph_map.png", cv2.cvtColor(threshold_morph_map, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{save_dir}defect_map.png", cv2.cvtColor(defect_map, cv2.COLOR_RGB2BGR))

        with open(f"{save_dir}results.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Parameter", "Value", "Unit", "Effective Range"])
            writer.writerow(["Sensitivity", self.sensitivity, "", "0-100"])
            writer.writerow(["Threshold", self.threshold, "", "0-255"])
            writer.writerow(["X-Ray Intensity", self.xray_intensity, "", "0-100"])
            writer.writerow(["Morphology", "On" if self.use_morphology else "Off", "", "On/Off"])
            writer.writerow(["Kernel Size", f"{self.kernel_size}x{self.kernel_size}", "", "3x3, 5x5, 7x7, 10x10"])
            metrics = self.calculate_metrics(self.image, defect_map).split('\n')
            for metric in metrics:
                writer.writerow([metric.split(': ')[0], metric.split(': ')[1], "", ""])
            writer.writerow(["Defect Type", "Count", "", ""])
            for defect, count in self.defect_counts.items():
                writer.writerow([defect, count, "", ""])

    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder to Load", "")
        if folder_path:
            image_paths = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.bmp', '.jpg', '.jpeg'))]
            if not image_paths:
                print("No .png, .bmp, .jpg, or .jpeg files found in the selected folder.")
                return
            self.image_list.clear()
            self.image_list.addItems(image_paths)
            self.folder_path = folder_path
            self.zoom_center = None
            self.width_map_centers = [None] * 4

    def update_image_from_list(self, item):
        if hasattr(self, 'folder_path'):
            image_path = os.path.join(self.folder_path, item.text())
            self.image = cv2.imread(image_path)
            if self.image is None:
                print(f"Failed to load image: {image_path}")
                return
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.sigma = np.zeros(self.image.shape[:2], dtype=float)
            self.zoom_center = None
            self.width_map_centers = [None] * 4
            self.update_display()

    def open_batch_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder to Process", "")
        if folder_path:
            save_dir = "C:/path_to_save/batch_processed/"
            os.makedirs(save_dir, exist_ok=True)
            ssim_sum, psnr_sum, count = 0, 0, 0
            for filename in tqdm(os.listdir(folder_path)):
                if filename.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(folder_path, filename)
                    self.image = cv2.imread(image_path)
                    if self.image is None:
                        print(f"Failed to load {image_path}")
                        continue
                    self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
                    self.sigma = np.zeros(self.image.shape[:2], dtype=float)
                    threshold_morph_map, defect_map = self.process_image_for_inspection()
                    output_dir = os.path.join(save_dir, filename.split('.')[0])
                    os.makedirs(output_dir, exist_ok=True)
                    cv2.imwrite(f"{output_dir}/threshold_morph_map.png", cv2.cvtColor(threshold_morph_map, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(f"{output_dir}/defect_map.png", cv2.cvtColor(defect_map, cv2.COLOR_RGB2BGR))
                    metrics = self.calculate_metrics(self.image, defect_map).split('\n')
                    ssim_val = float(metrics[0].split(': ')[1])
                    psnr_val = float(metrics[1].split(': ')[1].replace('dB', ''))
                    ssim_sum += ssim_val
                    psnr_sum += psnr_val
                    count += 1
                    with open(f"{output_dir}/results.csv", 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["Parameter", "Value", "Unit", "Effective Range"])
                        writer.writerow(["Sensitivity", self.sensitivity, "", "0-100"])
                        writer.writerow(["Threshold", self.threshold, "", "0-255"])
                        writer.writerow(["X-Ray Intensity", self.xray_intensity, "", "0-100"])
                        writer.writerow(["Morphology", "On" if self.use_morphology else "Off", "", "On/Off"])
                        writer.writerow(["Kernel Size", f"{self.kernel_size}x{self.kernel_size}", "", "3x3, 5x5, 7x7, 10x10"])
                        for metric in metrics:
                            writer.writerow([metric.split(': ')[0], metric.split(': ')[1], "", ""])
                        writer.writerow(["Defect Type", "Count", "", ""])
                        for defect, count in self.defect_counts.items():
                            writer.writerow([defect, count, "", ""])
            if count > 0:
                avg_ssim = ssim_sum / count
                avg_psnr = psnr_sum / count
                print(f"Average SSIM: {avg_ssim:.4f}, Average PSNR: {avg_psnr:.2f} dB")
            print("Batch processing completed!")

def main():
    app = QApplication(sys.argv)
    ex = SPIAXIImageProcessor()
    ex.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    if platform.system() == "Emscripten":
        import asyncio
        asyncio.ensure_future(main())
    else:
        main()