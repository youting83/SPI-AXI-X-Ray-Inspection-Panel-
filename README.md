# SPI/AXI/X-Ray Inspection Panel README

## Overview
This is a multi-language application designed for SPI (Solder Paste Inspection), AXI (Automated X-Ray Inspection), and X-Ray image processing. The tool provides an interactive GUI to analyze PCB (Printed Circuit Board) images, detect defects, and visualize results. It is available in both Python and C++ implementations.

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

## Dataset
This application utilizes the following publicly available datasets for training and validation:
- **PCB-AoI Public Dataset**: A dataset designed for Solder Paste Inspection (SPI) with Automated Optical Inspection (AOI) devices, containing annotated images to detect defects such as missing paste, bridging, and misalignments.
- **HRIPCB Dataset**: A synthesized dataset with 1386 images of naked PCBs, featuring 6 types of defects (e.g., missing holes, mouse bites, open circuits, shorts, spurs, spurious copper) for detection, classification, and registration tasks.

## Requirements
### Python
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

### C++
- C++17 compatible compiler (e.g., g++ 7.0 or later)
- Required Libraries:
  - OpenCV 4.x (e.g., `libopencv-dev` on Ubuntu)
  - Qt 5 (e.g., `qt5-default` on Ubuntu)
- Development Tools:
  - `pkg-config` for dependency linking
- Installation:
  - On Ubuntu: `sudo apt install g++ libopencv-dev qt5-default pkg-config`
  - On other systems, use equivalent package managers (e.g., `brew install opencv qt5 pkg-config` on macOS)

## Installation
### Python
1. Install the required Python packages:
   ```
   pip install pandas numpy opencv-python tqdm scikit-image matplotlib PyQt5
   ```
2. Ensure all dependencies are met before running the application.

### C++
1. Install the required libraries and tools as listed above.
2. Compile the C++ code (e.g., `spi_axi_xray_inspection.cpp`) with:
   ```
   g++ -o spi_axi_xray_inspection spi_axi_xray_inspection.cpp `pkg-config --cflags --libs opencv4 Qt5Widgets` -std=c++17
   ```
3. Ensure the OpenCV and Qt libraries are correctly linked during compilation.

## Usage
### Python
1. **Launch the Application**:
   - Run the script `spi_axi_xray_inspection.py`:
     ```
     python spi_axi_xray_inspection.py
     ```
2. **Load an Image**:
   - Use the "Load Initial Image" dialog to select a PCB image file from the PCB-AoI or HRIPCB dataset.
3. **Adjust Parameters**:
   - Use sliders to adjust Sensitivity (0-100), Threshold (0-255), X-Ray Intensity (0-100), Morphology (On/Off), and Kernel Size (3x3, 5x5, 7x7, 10x10).
4. **Interact with the GUI**:
   - Click on images to set the zoom center.
   - Use the Zoom Control slider to magnify the selected area.
   - Click on the Defect Statistics chart to filter defects.
5. **Save Results**:
   - Click "Save Results" to save the Threshold Morphology Defect Map, Defect Map, and a CSV file with analysis data to `C:/path_to_save/`.
6. **Batch Processing**:
   - Click "Load Folder" to select a directory containing PCB-AoI or HRIPCB dataset images, then "Run Batch Process" to process all images and save results to `C:/path_to_save/batch_processed/`.

### C++
1. **Compile the Application**:
   - Use the compilation command provided above to generate the executable.
2. **Run the Application**:
   - Execute the compiled binary:
     ```
     ./spi_axi_xray_inspection
     ```
3. **Follow the Same GUI Interactions**:
   - The C++ version mirrors the Python GUI workflow (load image from PCB-AoI or HRIPCB dataset, adjust parameters, zoom, filter defects, save results).

## Notes
- Update the `save_dir` paths (`C:/path_to_save/` and `C:/path_to_save/batch_processed/`) to valid directories on your system.
- The Python application exits if no image is selected during initialization.
- The "Spots/Dirt" defect detection includes a random 10% chance for demonstration; adjust the logic for real-world use with dataset-specific defects.
- Ensure PyQt5 is properly configured for Python GUI rendering, and Qt5 is correctly set up for C++.
- Some advanced features (e.g., NIQE, detailed histogram plotting) are simplified or omitted in the C++ version due to library limitations.
- Download the PCB-AoI Public Dataset from Kaggle and the HRIPCB dataset from relevant sources (e.g., Wiley Online Library or ResearchGate) to use with this application.

## License
This code is provided as-is without any warranty. Feel free to modify and distribute for personal or educational use.

## Contact
For issues or suggestions, please refer to the source code or adapt it based on your needs.

## Demo Result
<img width="2048" height="1112" alt="SPI_AOI" src="https://github.com/user-attachments/assets/032b12fb-3301-4999-8dfc-b54e1148e9ae" />
<img width="2558" height="1398" alt="SPI_AOI2" src="https://github.com/user-attachments/assets/23b91209-8bb8-4d2b-9fdd-b6072137eb79" />
