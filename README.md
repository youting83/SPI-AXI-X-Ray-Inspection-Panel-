# SPI/AXI/X-Ray Inspection Panel README

## Overview
This is a Python application designed for SPI (Solder Paste Inspection), AXI (Automated X-Ray Inspection), and X-Ray image processing. The tool provides an interactive GUI built with PyQt5 to analyze PCB (Printed Circuit Board) images, detect defects, and visualize results.

## Features
- **Image Loading**: Load individual images (BMP, PNG, JPG, JPEG) or a folder of images.
- **Defect Detection**: Identify defects such as No Solder, Solder Issues, Short Circuits, and Spots/Dirt using threshold and morphological operations.
- **Visualization**: Display Original PCB Image, Threshold Morphology Defect Map, and Classified Defect Map.
- **Interactive Controls**: Adjust Sensitivity, Threshold, X-Ray Intensity, Morphology, and Kernel Size via sliders.
- **Zoom Functionality**: Zoom into specific regions of the image.
- **Defect Filtering**: Click on the Defect Statistics chart to filter and display only the selected defect type.
- **Quality Metrics**: Calculate SSIM, PSNR, and NIQE (if available) for image quality assessment.
- **Batch Processing**: Process multiple images in a folder and save results.
- **Data Export**: Save processed images and results (including metrics and defect counts) to CSV files.

## Requirements
- Python 3.x
- Required Libraries:
  - `pandas`
  - `numpy`
  - `opencv-python` (cv2)
  - `tqdm`
  - `scikit-image`
  - `matplotlib`
  - `PyQt5`
- Optional: `scikit-image.restoration` for NIQE metric (install with `pip install scikit-image[restoration]`)

## Installation
1. Install the required Python packages:
   ```
   pip install pandas numpy opencv-python tqdm scikit-image matplotlib PyQt5
   ```
2. Ensure all dependencies are met before running the application.

## Usage
1. **Launch the Application**:
   - Run the script `spi_axi_xray_inspection.py`:
     ```
     python spi_axi_xray_inspection.py
     ```
2. **Load an Image**:
   - Use the "Load Initial Image" dialog to select a PCB image file.
3. **Adjust Parameters**:
   - Use sliders to adjust Sensitivity (0-100), Threshold (0-255), X-Ray Intensity (0-100), Morphology (On/Off), and Kernel Size (3x3, 5x5, 7x7, 10x10).
4. **Interact with the GUI**:
   - Click on images to set the zoom center.
   - Use the Zoom Control slider to magnify the selected area.
   - Click on the Defect Statistics chart to filter defects.
5. **Save Results**:
   - Click "Save Results" to save the Threshold Morphology Defect Map, Defect Map, and a CSV file with analysis data to `C:/path_to_save/`.
6. **Batch Processing**:
   - Click "Load Folder" to select a directory, then "Run Batch Process" to process all images and save results to `C:/path_to_save/batch_processed/`.

## Notes
- Update the `save_dir` paths (`C:/path_to_save/` and `C:/path_to_save/batch_processed/`) to valid directories on your system.
- The application exits if no image is selected during initialization.
- The "Spots/Dirt" defect detection includes a random 10% chance for demonstration; adjust the logic for real-world use.
- Ensure PyQt5 is properly configured for GUI rendering.

## License
This code is provided as-is without any warranty. Feel free to modify and distribute for personal or educational use.

## Contact
For issues or suggestions, please refer to the source code or adapt it based on your needs.
