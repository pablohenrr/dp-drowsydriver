# 🚗 Drowsiness Detection for Drivers

This project aims to develop a real-time drowsiness detection system using computer vision and deep learning techniques. The system processes video inputs and classifies whether a driver is **drowsy** or **not drowsy**, helping to enhance road safety.

![Architecture Overview](project-architecture/project-architecture.png)

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Training the Model](#training-the-model)
- [Classifying Videos](#classifying-videos)
- [Environment Variables](#environment-variables)
- [Dependencies](#dependencies)

---

## Overview

This project implements a driver drowsiness detection system using **YOLOv5** for image detection and **OpenCV** for real-time video processing. The goal is to detect whether a driver is **drowsy** or **not drowsy** based on the visual data from a video input.

The project is divided into two main tasks:
1. **Training a custom model** to classify driver states.
2. **Classifying videos** using the trained model to detect drowsiness.

## Architecture

The architecture follows the diagram below:

- **Dataset**: Images are labeled using **Label Studio**, stored in **Azure Blob Storage**, and sourced from a Kaggle dataset.
- **Model Training**: The **YOLOv5** model is trained using custom data to classify between "drowsy" and "not drowsy" drivers.
- **Classification**: The trained model is applied in real-time to videos using **OpenCV**.
- **Output**: Video is displayed with bounding boxes and labels indicating driver state.

## Dataset

The dataset is sourced from Kaggle and consists of labeled images of drivers in a "drowsy" and "non-drowsy" state. The dataset is preprocessed and stored in **Azure Blob Storage** for training the model.

**Tools used:**
- **Kaggle**: To source the dataset.
- **Label Studio**: For labeling the dataset.
- **Azure Blob Storage**: For storing and accessing the dataset during training.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/drowsiness-detector.git
    cd drowsiness-detector
    ```

2. Set up a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Before running the project, configure the following environment variables in a `.env` file:

```bash
STORAGE_ACCOUNT_NAME=your_azure_storage_account
STORAGE_ACCOUNT_KEY=your_azure_storage_key
CONTAINER_NAME=your_container_name
 ```

## Project Structure

```bash
drowsiness-detector/
│
├── blob_dataset_save.py            # Upload local dataset to Azure Blob Storage
├── classificar_video_sonolencia.py # Classify driver state in video using YOLOv5
├── gerar_labels_yolo.py            # Generate YOLOv5 labels from CSV annotations
├── treinar_modelo_deteccao_sonolencia.py # Train YOLOv5 model for drowsiness detection
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .env                            # Environment variables (not included in repo)
└── drowsy-driver-dataset/          # Dataset directory
    ├── images/                     # Images for training and validation
    └── labels/                     # Corresponding YOLO labels
 ```

## Training the Model
To train the YOLOv5 model for drowsiness detection, run the following command:

```bash
python treinar_modelo_deteccao_sonolencia.py
 ```

This script will:

Download the dataset from Azure Blob Storage.
Split the dataset into training and validation sets.
Configure the YOLOv5 model for the task.
Train the model for 50 epochs with a batch size of 16.

## Classifying Videos

After training the model, you can classify a new video to detect if the driver is drowsy.

Steps:
- Load the pre-trained YOLOv5 model.
- Open a video file (or stream) for frame-by-frame classification.
- For each frame, YOLOv5 will classify the driver’s state as drowsy or not drowsy.
- Bounding boxes and classification labels are drawn on the video, providing visual feedback.
  
To classify a video:

```bash
python classificar_video_sonolencia.py
 ```
Output:
- Real-time Video: The video will be displayed with colored bounding boxes:
- Red for drowsy.
- Green for not drowsy.
- Driver State: The driver's state will be displayed with a confidence score, e.g., Drowsy 0.85.
Example:
- Below is an example of a frame showing a driver classified as Drowsy:

1. Training the Model
To train the YOLOv5 model for drowsiness detection, follow the steps below.

Steps:
- Download the dataset from Azure Blob Storage, containing labeled images of drivers categorized as "drowsy" or "not drowsy".
- Data Split: The dataset is split into training and validation sets (80% training, 20% validation).
- Model Configuration: The YOLOv5 configuration file (data.yaml) is set up to recognize two classes: drowsy and not drowsy.
- Training: The YOLOv5 model is trained using 50 epochs and a batch size of 16.
  
To execute the training process:

```bash
python treinar_modelo_deteccao_sonolencia.py
 ```

This will initiate the training pipeline, and the model will be saved under yolov5/runs/train/ upon completion.

Output:
- Training Logs: The progress of the model will be shown in the terminal with metrics like loss, accuracy, and IoU.
- Saved Model: The best model weights will be saved in the yolov5/runs/train/ directory for future use.

## Environment Variables

The following environment variables need to be set up in the `.env` file to allow interaction with Azure Blob Storage:

```bash
STORAGE_ACCOUNT_NAME=your_azure_storage_account
STORAGE_ACCOUNT_KEY=your_azure_storage_key
CONTAINER_NAME=your_container_name
 ```

## Dependencies

Make sure you have the following dependencies installed. These are listed in the requirements.txt file:

- Azure SDK: To interact with Azure Blob Storage.
- YOLOv5: For object detection and model training.
- OpenCV: For video processing.
- Torch: PyTorch library for deep learning models.
- Python Dotenv: To manage environment variables from a .env file.
- Scikit-learn: For splitting the dataset.
- TQDM: To show progress bars in terminal during dataset download.

To install the required dependencies, simply run:

```bash
pip install -r requirements.txt
 ```

## Explanation of Key Dependencies

Make sure you have the following dependencies installed. These are listed in the requirements.txt file:

- YOLOv5: This is the core object detection framework used for training the model to detect drowsiness in drivers.
- Azure Storage Blob SDK: Used to interact with Azure Blob Storage, where the dataset is uploaded and downloaded.
- OpenCV: A powerful library for processing video inputs and displaying outputs with bounding boxes and classification results.
- PyTorch: A widely used deep learning framework to train and run the YOLOv5 model.
