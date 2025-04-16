# 312551185_HW1
Student ID: 312551185\
Name: 郭晏涵

## Introduction
This project aims to tackle a digit recognition problem using Faster R-CNN. The dataset consists of RGB images, with 30,062 samples for training, 3340 samples for validation, and 13068 samples for testing. For task 1, the goal is to recognize the class and bounding box of each digit in the image. For task 2, the goal is to predict the whole number in the image using the results from Task 1.

To enhance digit recognition performance, I employed several key strategies including transfer learning, adaptive learning rate scheduling, and post-processing techniques. The core of my method is built upon a Faster R-CNN framework with a ResNet-50-FPN backbone, fine-tuned for digit detection. MeanShift clustering filters out outliers, and digits are ordered spatially to reconstruct full numbers.

## Performance Snapshot
![image](https://github.com/slovengel/312551185_HW2/blob/main/codabench_snapshot.PNG)
