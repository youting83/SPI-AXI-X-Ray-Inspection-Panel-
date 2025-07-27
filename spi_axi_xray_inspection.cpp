#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QListWidget>
#include <QFileDialog>
#include <QImage>
#include <QPainter>
#include <QPoint>
#include <QRect>
#include <QColor>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>

class SPIAXIImageProcessor : public QMainWindow {
    Q_OBJECT

private:
    QPoint zoomCenter;
    cv::Mat image;
    cv::Mat sigma;
    QString folderPath;
    bool useMorphology = false;
    int kernelSize = 5;
    QString selectedDefect;
    std::map<QString, int> defectCounts = {{"No Solder", 0}, {"Solder Issue", 0}, {"Short Circuit", 0}, {"Spots/Dirt", 0}};
    int sensitivity = 50;
    int threshold = 100;
    int xrayIntensity = 75;

    QLabel *originalLabel;
    QLabel *thresholdMorphLabel;
    QLabel *defectLabel;
    QLabel *defectHistLabel;
    QLabel *statsLabel;
    QLabel *zoomedLabel;
    QLabel *qualityLabel;
    QSlider *zoomSlider;
    QListWidget *imageList;

    void initUI();
    void updateDisplay();
    cv::Mat processImageForInspection();
    QString calculateMetrics(const cv::Mat& original, const cv::Mat& processed);
    void setZoomCenter(QMouseEvent *event);
    void updateZoom(int val);
    void updateSensitivity(int val);
    void updateThreshold(int val);
    void updateXrayIntensity(int val);
    void toggleMorphology(int val);
    void updateKernelSize(int val);
    QString getKernelSizeLabel(int val);
    void suggestParameters();
    void saveResults();
    void loadFolder();
    void updateImageFromList(QListWidgetItem *item);

public:
    SPIAXIImageProcessor(QWidget *parent = nullptr) : QMainWindow(parent) {
        initUI();
    }
};

void SPIAXIImageProcessor::initUI() {
    setWindowTitle("SPI/AXI/X-Ray Inspection Panel");
    setGeometry(100, 100, 2700, 900);
    setStyleSheet("QMainWindow { background-color: #2E2E2E; color: #E0E0E0; }"
                  "QWidget { border: 1px solid #4A4A4A; background-color: #3A3A3A; border-radius: 5px; padding: 5px; }"
                  "QLabel { color: #E0E0E0; font-family: Arial; font-size: 12px; }"
                  "QPushButton { background-color: #5A5A5A; color: #E0E0E0; border: 1px solid #7A7A7A; border-radius: 5px; padding: 5px; }"
                  "QPushButton:hover { background-color: #6A6A6A; }"
                  "QSlider { background-color: #4A4A4A; }"
                  "QSlider::groove:horizontal { background: #5A5A5A; height: 8px; border-radius: 4px; }"
                  "QSlider::handle:horizontal { background: #9ACD10; width: 12px; margin: -4px 0; border-radius: 6px; }"
                  "QSlider::handle:vertical { background: #9ACD10; width: 16px; }"
                  "QLineEdit { background-color: #4A4A4A; color: #E0E0E0; border: 1px solid #7A7A7A; border-radius: 3px; }"
                  "QListWidget { background-color: #4A4A4A; color: #E0E0E0; border: 1px solid #7A7A7A; }");

    QWidget *mainWidget = new QWidget(this);
    setCentralWidget(mainWidget);
    QHBoxLayout *mainLayout = new QHBoxLayout();
    mainLayout->setSpacing(10);
    mainWidget->setLayout(mainLayout);

    // Left Panel: Control Section
    QWidget *controlWidget = new QWidget();
    QVBoxLayout *controlLayout = new QVBoxLayout();
    controlLayout->setSpacing(5);
    controlWidget->setLayout(controlLayout);

    QLabel *header = new QLabel("SPI/AXI/X-Ray Inspection Panel");
    header->setStyleSheet("font-weight: bold; color: #9ACD10; font-size: 14px;");
    header->setAlignment(Qt::AlignCenter);
    controlLayout->addWidget(header);

    QString filePath = QFileDialog::getOpenFileName(this, "Load Initial Image", "", "Image Files (*.bmp *.png *.jpg *.jpeg)");
    if (filePath.isEmpty()) {
        throw std::runtime_error("No image selected. Application will exit.");
    }
    image = cv::imread(filePath.toStdString());
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + filePath.toStdString());
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    sigma = cv::Mat::zeros(image.size(), CV_64F);

    QSlider *sensitivitySlider = new QSlider(Qt::Horizontal);
    sensitivitySlider->setRange(0, 100);
    sensitivitySlider->setValue(sensitivity);
    QLabel *sensitivityValue = new QLabel("Sensitivity: " + QString::number(sensitivity));
    sensitivityValue->setStyleSheet("color: #9ACD10; font-weight: bold;");
    connect(sensitivitySlider, &QSlider::valueChanged, this, [this, sensitivityValue](int val) {
        updateSensitivity(val);
        sensitivityValue->setText("Sensitivity: " + QString::number(val));
    });
    controlLayout->addWidget(new QLabel("Sensitivity (0-100)"));
    controlLayout->addWidget(sensitivityValue);
    controlLayout->addWidget(sensitivitySlider);

    QSlider *thresholdSlider = new QSlider(Qt::Horizontal);
    thresholdSlider->setRange(0, 255);
    thresholdSlider->setValue(threshold);
    QLabel *thresholdValue = new QLabel("Threshold: " + QString::number(threshold));
    thresholdValue->setStyleSheet("color: #9ACD10; font-weight: bold;");
    connect(thresholdSlider, &QSlider::valueChanged, this, [this, thresholdValue](int val) {
        updateThreshold(val);
        thresholdValue->setText("Threshold: " + QString::number(val));
    });
    controlLayout->addWidget(new QLabel("Threshold (0-255)"));
    controlLayout->addWidget(thresholdValue);
    controlLayout->addWidget(thresholdSlider);

    QSlider *xraySlider = new QSlider(Qt::Horizontal);
    xraySlider->setRange(0, 100);
    xraySlider->setValue(xrayIntensity);
    QLabel *xrayValue = new QLabel("X-Ray Intensity: " + QString::number(xrayIntensity));
    xrayValue->setStyleSheet("color: #9ACD10; font-weight: bold;");
    connect(xraySlider, &QSlider::valueChanged, this, [this, xrayValue](int val) {
        updateXrayIntensity(val);
        xrayValue->setText("X-Ray Intensity: " + QString::number(val));
    });
    controlLayout->addWidget(new QLabel("X-Ray Intensity (0-100)"));
    controlLayout->addWidget(xrayValue);
    controlLayout->addWidget(xraySlider);

    QSlider *morphologySlider = new QSlider(Qt::Horizontal);
    morphologySlider->setRange(0, 1);
    morphologySlider->setValue(0);
    QLabel *morphologyValue = new QLabel("Morphology: Off");
    morphologyValue->setStyleSheet("color: #9ACD10; font-weight: bold;");
    connect(morphologySlider, &QSlider::valueChanged, this, [this, morphologyValue](int val) {
        toggleMorphology(val);
        morphologyValue->setText("Morphology: " + QString(val ? "On" : "Off"));
    });
    controlLayout->addWidget(new QLabel("Morphology (0: Off, 1: On)"));
    controlLayout->addWidget(morphologyValue);
    controlLayout->addWidget(morphologySlider);

    QSlider *kernelSlider = new QSlider(Qt::Horizontal);
    kernelSlider->setRange(0, 3);
    kernelSlider->setValue(1);
    QLabel *kernelValue = new QLabel("Kernel Size: 5x5");
    kernelValue->setStyleSheet("color: #9ACD10; font-weight: bold;");
    connect(kernelSlider, &QSlider::valueChanged, this, [this, kernelValue](int val) {
        updateKernelSize(val);
        kernelValue->setText("Kernel Size: " + getKernelSizeLabel(val) + "x" + getKernelSizeLabel(val));
    });
    controlLayout->addWidget(new QLabel("Kernel Size (0: 3x3, 1: 5x5, 2: 7x7, 3: 10x10)"));
    controlLayout->addWidget(kernelValue);
    controlLayout->addWidget(kernelSlider);

    QPushButton *suggestButton = new QPushButton("Suggest Parameters");
    suggestButton->setStyleSheet("font-weight: bold;");
    connect(suggestButton, &QPushButton::clicked, this, &SPIAXIImageProcessor::suggestParameters);
    controlLayout->addWidget(suggestButton);

    QPushButton *saveButton = new QPushButton("Save Results");
    saveButton->setStyleSheet("font-weight: bold;");
    connect(saveButton, &QPushButton::clicked, this, &SPIAXIImageProcessor::saveResults);
    controlLayout->addWidget(saveButton);

    QPushButton *loadFolderButton = new QPushButton("Load Folder");
    loadFolderButton->setStyleSheet("font-weight: bold;");
    connect(loadFolderButton, &QPushButton::clicked, this, &SPIAXIImageProcessor::loadFolder);
    controlLayout->addWidget(loadFolderButton);

    controlLayout->addStretch();
    mainLayout->addWidget(controlWidget, 1);

    // Right Panel: Image and Analysis Section
    QWidget *analysisWidget = new QWidget();
    QHBoxLayout *analysisLayout = new QHBoxLayout();
    analysisLayout->setSpacing(10);
    analysisWidget->setLayout(analysisLayout);

    QWidget *imageWidget = new QWidget();
    QVBoxLayout *imageLayout = new QVBoxLayout();
    imageLayout->setSpacing(5);
    imageWidget->setLayout(imageLayout);

    QLabel *originalTitle = new QLabel("Original PCB Image");
    originalTitle->setStyleSheet("font-weight: bold; color: #9ACD10;");
    originalTitle->setAlignment(Qt::AlignCenter);
    originalLabel = new QLabel(this);
    originalLabel->setAlignment(Qt::AlignCenter);
    originalLabel->setStyleSheet("border: 2px solid #7A7A7A;");
    imageLayout->addWidget(originalTitle);
    imageLayout->addWidget(originalLabel);

    QLabel *thresholdMorphTitle = new QLabel("Threshold Morphology Defect Map");
    thresholdMorphTitle->setStyleSheet("font-weight: bold; color: #9ACD10;");
    thresholdMorphTitle->setAlignment(Qt::AlignCenter);
    thresholdMorphLabel = new QLabel(this);
    thresholdMorphLabel->setAlignment(Qt::AlignCenter);
    thresholdMorphLabel->setStyleSheet("border: 2px solid #7A7A7A;");
    imageLayout->addWidget(thresholdMorphTitle);
    imageLayout->addWidget(thresholdMorphLabel);

    QLabel *defectTitle = new QLabel("Defect Map (Classified)");
    defectTitle->setStyleSheet("font-weight: bold; color: #9ACD10;");
    defectTitle->setAlignment(Qt::AlignCenter);
    defectLabel = new QLabel(this);
    defectLabel->setAlignment(Qt::AlignCenter);
    defectLabel->setStyleSheet("border: 2px solid #7A7A7A;");
    imageLayout->addWidget(defectTitle);
    imageLayout->addWidget(defectLabel);

    QLabel *defectHistTitle = new QLabel("Defect Histogram");
    defectHistTitle->setStyleSheet("font-weight: bold; color: #9ACD10;");
    defectHistTitle->setAlignment(Qt::AlignCenter);
    defectHistLabel = new QLabel(this);
    defectHistLabel->setAlignment(Qt::AlignCenter);
    defectHistLabel->setStyleSheet("border: 2px solid #7A7A7A;");
    imageLayout->addWidget(defectHistTitle);
    imageLayout->addWidget(defectHistLabel);

    imageLayout->addStretch();
    analysisLayout->addWidget(imageWidget, 1);

    QWidget *toolsWidget = new QWidget();
    QVBoxLayout *toolsLayout = new QVBoxLayout();
    toolsLayout->setSpacing(5);
    toolsWidget->setLayout(toolsLayout);

    QWidget *zoomWidget = new QWidget();
    QVBoxLayout *zoomLayout = new QVBoxLayout();
    zoomLayout->setSpacing(5);
    zoomWidget->setLayout(zoomLayout);
    QLabel *zoomTitle = new QLabel("Zoom Control");
    zoomTitle->setStyleSheet("font-weight: bold; color: #9ACD10;");
    zoomTitle->setAlignment(Qt::AlignCenter);
    zoomLayout->addWidget(zoomTitle);
    zoomSlider = new QSlider(Qt::Vertical);
    zoomSlider->setRange(1, 10);
    zoomSlider->setValue(1);
    connect(zoomSlider, &QSlider::valueChanged, this, &SPIAXIImageProcessor::updateZoom);
    zoomLayout->addWidget(zoomSlider);
    zoomedLabel = new QLabel(this);
    zoomedLabel->setAlignment(Qt::AlignCenter);
    zoomedLabel->setStyleSheet("border: 2px solid #7A7A7A;");
    zoomedLabel->setFixedSize(200, 200);
    zoomLayout->addWidget(zoomedLabel);
    QLabel *zoomValue = new QLabel("Zoom: 1x");
    zoomValue->setAlignment(Qt::AlignCenter);
    zoomValue->setStyleSheet("color: #9ACD10;");
    zoomLayout->addWidget(zoomValue);
    toolsLayout->addWidget(zoomWidget);

    qualityLabel = new QLabel(this);
    qualityLabel->setAlignment(Qt::AlignCenter);
    qualityLabel->setStyleSheet("border: 2px solid #7A7A7A; background-color: #4A4A4A;");
    toolsLayout->addWidget(qualityLabel);

    QWidget *imageListWidget = new QWidget();
    QVBoxLayout *imageListLayout = new QVBoxLayout();
    imageListLayout->setSpacing(5);
    imageListWidget->setLayout(imageListLayout);
    QLabel *imageListTitle = new QLabel("Image Queue");
    imageListTitle->setStyleSheet("font-weight: bold; color: #9ACD10;");
    imageListTitle->setAlignment(Qt::AlignCenter);
    imageList = new QListWidget();
    connect(imageList, &QListWidget::itemClicked, this, &SPIAXIImageProcessor::updateImageFromList);
    imageListLayout->addWidget(imageListTitle);
    imageListLayout->addWidget(imageList);
    toolsLayout->addWidget(imageListWidget);

    QLabel *statsTitle = new QLabel("Defect Statistics");
    statsTitle->setStyleSheet("font-weight: bold; color: #9ACD10;");
    statsTitle->setAlignment(Qt::AlignCenter);
    statsLabel = new QLabel(this);
    statsLabel->setAlignment(Qt::AlignCenter);
    statsLabel->setStyleSheet("border: 2px solid #7A7A7A;");
    statsLabel->setFixedSize(300, 200);
    toolsLayout->addWidget(statsTitle);
    toolsLayout->addWidget(statsLabel);

    toolsLayout->addStretch();
    analysisLayout->addWidget(toolsWidget, 1);

    mainLayout->addWidget(analysisWidget, 2);

    originalLabel->installEventFilter(this);
    thresholdMorphLabel->installEventFilter(this);
    defectLabel->installEventFilter(this);
    statsLabel->installEventFilter(this);

    updateDisplay();
}

bool SPIAXIImageProcessor::eventFilter(QObject *obj, QEvent *event) {
    if (event->type() == QEvent::MouseButtonPress) {
        QMouseEvent *mouseEvent = static_cast<QMouseEvent*>(event);
        if (mouseEvent->button() == Qt::LeftButton) {
            if (obj == originalLabel || obj == thresholdMorphLabel || obj == defectLabel) {
                setZoomCenter(mouseEvent);
            } else if (obj == statsLabel) {
                // Simplified filter defects (placeholder)
                std::cout << "Filter defects clicked" << std::endl;
            }
        }
    }
    return QMainWindow::eventFilter(obj, event);
}

void SPIAXIImageProcessor::setZoomCenter(QMouseEvent *event) {
    if (event->button() == Qt::LeftButton) {
        zoomCenter = event->pos();
        zoomCenter.setX(zoomCenter.x() * image.cols / originalLabel->width());
        zoomCenter.setY(zoomCenter.y() * image.rows / originalLabel->height());
        updateZoom(zoomSlider->value());
    }
}

void SPIAXIImageProcessor::updateZoom(int val) {
    if (!zoomCenter.isNull() && !image.empty()) {
        int zoomFactor = val;
        int labelWidth = 200, labelHeight = 200;
        int centerX = zoomCenter.x();
        int centerY = zoomCenter.y();
        int halfWidth = labelWidth / (2 * zoomFactor);
        int halfHeight = labelHeight / (2 * zoomFactor);

        int xStart = std::max(0, std::min(centerX - halfWidth, image.cols - labelWidth / zoomFactor));
        int yStart = std::max(0, std::min(centerY - halfHeight, image.rows - labelHeight / zoomFactor));
        int xEnd = std::min(image.cols, xStart + labelWidth / zoomFactor);
        int yEnd = std::min(image.rows, yStart + labelHeight / zoomFactor);

        cv::Mat zoomedRegion = image(cv::Rect(xStart, yStart, xEnd - xStart, yEnd - yStart));
        if (!zoomedRegion.empty()) {
            cv::resize(zoomedRegion, zoomedRegion, cv::Size(labelWidth, labelHeight), 0, 0, cv::INTER_LINEAR);
            QImage qImage(zoomedRegion.data, zoomedRegion.cols, zoomedRegion.rows, zoomedRegion.step, QImage::Format_RGB888);
            zoomedLabel->setPixmap(QPixmap::fromImage(qImage));
        }
    }
}

void SPIAXIImageProcessor::updateSensitivity(int val) { sensitivity = val; updateDisplay(); }
void SPIAXIImageProcessor::updateThreshold(int val) { threshold = val; updateDisplay(); }
void SPIAXIImageProcessor::updateXrayIntensity(int val) { xrayIntensity = val; updateDisplay(); }
void SPIAXIImageProcessor::toggleMorphology(int val) { useMorphology = val; updateDisplay(); }
void SPIAXIImageProcessor::updateKernelSize(int val) { kernelSize = (val == 0) ? 3 : (val == 1) ? 5 : (val == 2) ? 7 : 10; updateDisplay(); }
QString SPIAXIImageProcessor::getKernelSizeLabel(int val) { return QString::number((val == 0) ? 3 : (val == 1) ? 5 : (val == 2) ? 7 : 10); }

cv::Mat SPIAXIImageProcessor::processImageForInspection() {
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_RGB2GRAY);
    cv::Mat binaryImage;
    cv::threshold(grayImage, binaryImage, threshold, 255, cv::THRESH_BINARY_INV);
    cv::Mat kernel = cv::Mat::ones(kernelSize, kernelSize, CV_8U);

    if (useMorphology) {
        cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(binaryImage, binaryImage, cv::MORPH_CLOSE, kernel);
    }

    cv::Mat thresholdMorphMap;
    cv::cvtColor(binaryImage, thresholdMorphMap, cv::COLOR_GRAY2RGB);

    cv::Mat dilated;
    cv::dilate(binaryImage, dilated, kernel, cv::Point(-1, -1), 1);
    cv::Mat labels;
    int numComponents = cv::connectedComponents(dilated, labels, 8, CV_32S);

    cv::Mat defectMap(image.rows, image.cols, CV_8UC4, cv::Scalar(0, 0, 0, 0));
    image.copyTo(defectMap(cv::Rect(0, 0, image.cols, image.rows)));

    defectCounts["No Solder"] = 0;
    defectCounts["Solder Issue"] = 0;
    defectCounts["Short Circuit"] = 0;
    defectCounts["Spots/Dirt"] = 0;

    for (int i = 1; i < numComponents; ++i) {
        cv::Mat componentMask = (labels == i);
        int area = cv::countNonZero(componentMask);
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(componentMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        if (!contours.empty()) {
            cv::drawContours(defectMap, contours, -1, cv::Scalar(255, 0, 0, 255), 2);
        }

        if (area < 50) {
            defectMap.setTo(cv::Scalar(255, 0, 0, 128), componentMask);
            defectCounts["No Solder"]++;
        } else if (area <= 100) {
            defectMap.setTo(cv::Scalar(255, 255, 0, 128), componentMask);
            defectCounts["Solder Issue"]++;
        } else if (area > 200) {
            defectMap.setTo(cv::Scalar(0, 0, 255, 128), componentMask);
            defectCounts["Short Circuit"]++;
        } else if (rand() % 10 == 0) {
            defectMap.setTo(cv::Scalar(128, 0, 128, 128), componentMask);
            defectCounts["Spots/Dirt"]++;
        }
    }

    cv::Mat defectColorMap;
    cv::cvtColor(defectMap, defectColorMap, cv::COLOR_RGBA2RGB);
    return thresholdMorphMap, defectColorMap;
}

QString SPIAXIImageProcessor::calculateMetrics(const cv::Mat& original, const cv::Mat& processed) {
    cv::Mat originalGray, processedGray;
    cv::cvtColor(original, originalGray, cv::COLOR_RGB2GRAY);
    cv::cvtColor(processed, processedGray, cv::COLOR_RGB2GRAY);
    double ssimValue = 0.0; // Placeholder, requires OpenCV contrib for actual SSIM
    double psnrValue = cv::PSNR(originalGray, processedGray);
    return QString("SSIM: %1\nPSNR: %2 dB").arg(ssimValue, 0, 'f', 4).arg(psnrValue, 0, 'f', 2);
}

void SPIAXIImageProcessor::updateDisplay() {
    cv::Mat thresholdMorphMap, defectMap;
    std::tie(thresholdMorphMap, defectMap) = processImageForInspection();

    QImage originalQImage(image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
    originalLabel->setPixmap(QPixmap::fromImage(originalQImage).scaled(500, 500, Qt::KeepAspectRatio));

    QImage thresholdMorphQImage(thresholdMorphMap.data, thresholdMorphMap.cols, thresholdMorphMap.rows, thresholdMorphMap.step, QImage::Format_RGB888);
    thresholdMorphLabel->setPixmap(QPixmap::fromImage(thresholdMorphQImage).scaled(500, 500, Qt::KeepAspectRatio));

    QImage defectQImage(defectMap.data, defectMap.cols, defectMap.rows, defectMap.step, QImage::Format_RGB888);
    defectLabel->setPixmap(QPixmap::fromImage(defectQImage).scaled(500, 500, Qt::KeepAspectRatio));

    // Placeholder for histogram and stats (requires additional rendering logic)
    qualityLabel->setText(calculateMetrics(image, defectMap));

    if (!zoomCenter.isNull()) {
        updateZoom(zoomSlider->value());
    }
}

void SPIAXIImageProcessor::suggestParameters() {
    sensitivity = 75;
    threshold = 120;
    xrayIntensity = 80;
    updateDisplay();
}

void SPIAXIImageProcessor::saveResults() {
    cv::Mat thresholdMorphMap, defectMap;
    std::tie(thresholdMorphMap, defectMap) = processImageForInspection();
    std::string saveDir = "C:/path_to_save/";
    std::filesystem::create_directories(saveDir);
    cv::imwrite(saveDir + "threshold_morph_map.png", thresholdMorphMap);
    cv::imwrite(saveDir + "defect_map.png", defectMap);
}

void SPIAXIImageProcessor::loadFolder() {
    folderPath = QFileDialog::getExistingDirectory(this, "Select Folder to Load", "");
    if (!folderPath.isEmpty()) {
        imageList->clear();
        QDir dir(folderPath);
        QStringList imageFiles = dir.entryList(QStringList() << "*.png" << "*.bmp" << "*.jpg" << "*.jpeg", QDir::Files);
        imageList->addItems(imageFiles);
        zoomCenter = QPoint();
    }
}

void SPIAXIImageProcessor::updateImageFromList(QListWidgetItem *item) {
    if (!folderPath.isEmpty()) {
        QString imagePath = folderPath + "/" + item->text();
        image = cv::imread(imagePath.toStdString());
        if (image.empty()) {
            std::cout << "Failed to load image: " << imagePath.toStdString() << std::endl;
            return;
        }
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        sigma = cv::Mat::zeros(image.size(), CV_64F);
        zoomCenter = QPoint();
        updateDisplay();
    }
}

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    SPIAXIImageProcessor window;
    window.show();
    return app.exec();
}