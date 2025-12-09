# Plasmodium Vivax (malaria) parasite detector and counter using YOLOv8n Model version 2

## Introduction

Malaria, a life-threatening disease caused by Plasmodium parasites transmitted by female Anopheles mosquitoes, is most commonly found in tropical and subtropical regions. In 2023, the African Region was home to 94% and 95% of malaria cases and deaths, respectively. Sub-Saharan Africa carries a disproportionately high share of the global malaria burden. Plasmodium vivax is one of the 5 parasites that cause malaria associated with disease relapses due to dormant liver stages, contributing significantly to morbidity.  

The Plasmodium vivax lifecycle alternates between humans and Anopheles mosquitoes. Infection begins when a mosquito injects sporozoites into a human, which then move to the liver to either develop or remain dormant. Merozoites released from the liver infect red blood cells, progressing through ring, trophozoite, and schizont stages, causing malaria symptoms. Some parasites develop into gametocytes, enabling further transmission via mosquitoes.

Diagnosis of Plasmodium vivax malaria relies on examining stained blood smears via microsocopy and rapid diagnostic tests, with accurate detection being key to effective treatment and relapse prevention. Advanced methods like deep learning models, such as  Regional Based CNN, YOLOv8, and others would offer improved reliability and efficiency in identifying infections.

<img width="755" height="472" alt="image" src="https://github.com/user-attachments/assets/d55d4034-1d38-4192-a656-e65af73a03a8" />


Plasmodium Vivax Lifecycle

## Problem Statement

The detection of Plasmodium vivax malaria remains challenging, primarily due to the morphological similarities between parasite stages and other blood components observed in microscopic images. Conventional microscopy is a labor-intensive process that is susceptible to human error and demands specialized expertise, which may be limited in under-resourced settings. Consequently, there is an urgent need for automated, efficient, and dependable solutions capable of detecting and quantifying Plasmodium vivax parasites and their developmental stages within blood smears. Such advancements would enhance diagnostic accuracy, promote effective treatment, and strengthen malaria control initiatives.

## Project Objectives

1.	To develop a computer vision model for object detection and counting
2.	To detect uninfected red blood cells and leukocytes in human blood cells in a microscopic image from a blood smear using the developed YOLOv8 model
3.	To count the number of uninfected red blood cells and leukocytes in human blood cells in a microscpic image from a blood smear using the developed YOLOv8 model
4.	To detect the growth stage of a Plasmodium Vivax (malaria parasite) in human blood cells in a microscopic image from a blood smear using the developed YOLOv8 model
5.	 To count the number of detected growth stages of the plasmodium vivax parasites per microscopic image using the developed YOLOv8 model

## Image Dataset

- Images obtained from Broad Bioimage Benchmark Collection website <https://bbbc.broadinstitute.org/BBBC041/>
- The dataset consists of image folder, training json file & test json file
- There are 1,328 microscopic images of blood smears.
- Image resolution:1600x1200
- Class label & set of bounding box coordinates given for each image.
- 7 Class labels: red blood cell (uninfected), trophozoite, gametocyte, schizont, difficult, ring & leukocyte (uninfected)
- The Red Blood Cell and Leukocyte classes are blood cells that are not infected with the malaria parasite.
- The Trophozoite, Schizont, Ring and Gametocyte are different growth stages of the malaria parasite
- The difficult class label is for observed growth stages that couldn't be grouped into either of the 4 growth classes.
- Sample images from the dataset

<img width="491" height="325" alt="image" src="https://github.com/user-attachments/assets/24c25e2a-f8b3-47c6-837b-bb9d800e3430" />

<img width="447" height="319" alt="image" src="https://github.com/user-attachments/assets/dfdab58d-827d-496b-8da6-bcee41ffbd42" />

## Project Description
This project employs YOLOv8 for multiclass object detection, leveraging its advanced capabilities to accurately identify and localize multiple object types within images or video frames.

## Project Overview

<img width="909" height="225" alt="image" src="https://github.com/user-attachments/assets/93ad134d-cf7d-479a-a33d-67afb28369d8" />


## Deep learning Model Architecture

The You Only Look Once (YOLO) model is a single stage detector that predicts bounding boxes and class probabilities directly from the entire input image in a single forward pass, which makes the model faster than other object detection models. The model treats object detection as a single regression problem. 

The YOLO version 8 model (YOLOv8) will be used for object detection and counting. The model size to be used is yolov8n (Nano) which has about 3 million parameters, is the fastest, suitable for small datasets and computers with limited GPU. However, it's accuracy is lower than other bigger sizes of YOLOv8 models. 

The model is dividied into three main components:
- **Backbone (feature extractor)** - this consists of the CNN that is responsible for extracting hierarchical features from the input image.
- **Neck** - this merges/fuses feature maps from the different stages of the backbone to capture information at various scales.
- **Head** - this is responsible for making predictions. It takes the merged features from the neck and outputs bounding box coordinates, class probabilities, and confidence scores for detected objects. The Head typically consists of multiple detection heads, each connected to a different output scale from the Neck, enabling the prediction of objects at various sizes. Post-processing techniques like non-maximum suppression (NMS) are applied to filter out redundant or overlapping bounding box predictions, resulting in the final set of detected objects.

<img width="1207" height="1122" alt="image" src="https://github.com/user-attachments/assets/4665efe1-8dd4-4cbc-b2f1-d57c7475b34c" />

**Object detection evaluation metrics** used will be precision, recall, F1 Score and mean average precision (mAP).
  
- **Precision**: This is the ratio of correctly predicted positive detections (True Positives) to the total number of positive detections (True Positives + False Positives).It tells you how accurate the model is when it predicts an object is present. High precision = fewer false detections.
  
Precision = TP / (TP + FP)

- **Recall**: This is the ratio of correctly predicted positive detections (True Positives) to the total number of actual positive objects in the image (True Positives + False Negatives). It tells you how many of the actual objects the model was able to find. High recall = fewer missed detections.

Recall = TP / (TP + FN)

- **F1 Score**: Harmonic mean of precision and recall.YOLOv8 often reports best F1 (at optimal confidence threshold).

<img width="214" height="35" alt="Screenshot 2025-11-02 at 17 57 44" src="https://github.com/user-attachments/assets/293697af-cd78-4848-9c66-97ac24540aca" />

- **mAP@0.5** — IoU threshold = 0.5 (i.e., boxes overlap ≥ 50% to count as correct) - mean average precision calculated at a fixed IOU threshold of 0.50. This generally assesses whether the model can generaly detect the presence and approximate location of an object, and is a less less stric metric. 
 
- **mAP@0.5:0.95** — Mean mAP across IoU thresholds 0.5 to 0.95 (step 0.05) - average of the mean average precision calculated across multiple IoU thresholds, ranging from 0.50 to 0.95 in steps of 0.05 (i.e 0.50, 0.55, 0.60,...,0.95). 

Other metrics to help understand mAP@0.5 & mAP@0.5-0.95 performance metrics in YOLOv8:
- **Intersection over Union (IoU)**: This measures the overlap between the model's predicted bounding box and the actual ground truth bounding box. An IoU of 1 means perfect overlap, while 0 means no overlap. A common threshold (e.g., 0.5) is set to consider a detection as a True Positive. Higher IoU = better localization accuracy. 

<img width="172" height="38" alt="Screenshot 2025-11-02 at 17 58 54" src="https://github.com/user-attachments/assets/7a788e3c-8038-46ee-90b7-c98468378cb3" />

## Results

### Data Preparation

Class Imbalance noted in the train, val and test image subsets as shown in the charts below. The imbalance is severe with 96% of the objects being from the red blood cell class. This imbalance is inherent to human blood smears as they have more red blood cells than other cells found in blood. 

Data preparation entailed:
1. 

### Model Training

#### Error Analysis

### Model Evaluation

### Hyperparameter tuning

### Model Evaluation on Test Images

### Predicting on test images

### Model Selection

### Deployment
https://malaria-parasite-detector-version-2-kuhwhgejcrbgyxsma4pzrp.streamlit.app/

## Recommendations/Future work
1. Use larger YOLOv8 model e.g. medium variant
2. Use 2 stages for detecting & classifying malaria parasites
  - 1st stage: detect uninfected red blood cells vs. infected red blood cells with YOLOv8 model
  - 2nd stage: classify infected red blood cells into various parasite stages with an image classifier model e.g. AlexNet, EfficientNet, GoogLeNet, ResNet , MobileNet, Vision Transformers (ViT) etc..

## Acknowledgements & Attributions
1. We used image set BBBC041v1, available from the Broad Bioimage Benchmark Collection (Ljosa et al., Nature Methods, 2012)
2. Image of YOLOv8 model architecture from: https://abintimilsina.medium.com/yolov8-architecture-explained-a5e90a560ce5
3. Image of P.Vivax lifecylce. Quique Bassat, CC BY 4.0 <https://creativecommons.org/licenses/by/4.0>, via Wikimedia Commons

## References
1. https://www.who.int/health-topics/malaria#tab=tab_1
2. https://docs.ultralytics.com/datasets/
3. https://abintimilsina.medium.com/yolov8-architecture-explained-a5e90a560ce5]
4. Link to data: Broad Bioimage Benchmark Collection website https://bbbc.broadinstitute.org/BBBC041/


