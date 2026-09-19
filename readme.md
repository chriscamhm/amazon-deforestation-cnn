# Transfer learning to identify land cover and use in the amazon forest

## Overview

This repository contains the code and methodology used to classify land cover and detect land use changes in the Amazon rainforest using deep learning. The Amazon faces severe deforestation threats, with significant primary forest loss occurring in regions like Brazil and the Vaupés department in Colombia.

This project leverages a transfer learning approach with Convolutional Neural Networks (CNNs) to classify satellite imagery into various land cover types (e.g., primary forest, water bodies) and land uses (e.g., agriculture, logging, mining).

For a comprehensive review of the theoretical background, methodology, and visual analyses, please refer to the complete course paper: **PTNY8_2.pdf**.

## Dataset

The model is trained on a distilled subset of the **Planet: Amazon Satellite Dataset** (originally from the 2017 Kaggle competition), compiled to reduce computational complexity.

### Data Characteristics & Preprocessing

* **Size:** 40,479 images.

* **Format:** $256\times256$ pixel RGB images in JPEG format.

* **Labels:** Multi-label dataset containing 17 distinct classes covering both land types (e.g., primary, agriculture, water) and atmospheric conditions (e.g., clear, cloudy, haze).

* **Processing:** Images were loaded into a custom PyTorch dataset, converted from BGR to RGB, cropped to $224\times224$ pixels, transformed into tensors, and normalized. Labels were binarized into a 17-dimensional vector.

* **Challenge:** The dataset features a significant class imbalance, heavily skewing toward "primary" forest and "clear" weather conditions.

## Methodology

* **Model Architecture:** Transfer learning using a pre-trained **ResNet-50** CNN. The final layer was modified to output 17 classes.

* **Loss Function:** Binary Cross-Entropy (BCE) Loss was used to treat each label as an independent binary classification task, suitable for multi-label problems.

* **Training Details:** The model was trained over 10 epochs using a learning rate of 0.001.

* **Metrics:** Evaluated using Accuracy and the F2 Score (which heavily weighs recall, crucial for identifying rare deforestation events).

## Study Area Application

To test the model's ability to generalize to unseen, real-world data, it was applied to a **Sentinel-2** satellite image of the **Vaupés department, Colombia**, captured in January 2025. The high-resolution image was divided into 3,920 patches of $256\times256$ pixels and fed into the trained model.

## Key Results

* **Metrics Achieved:** The model achieved a **test accuracy of 94.93%** and a **test F2 score of 0.78** by the 10th epoch, with steady loss reduction indicating no overfitting.

* **Generalization Success:** When applied to the Vaupés region, the model effectively identified major land cover categories like primary forests, clear conditions, and water bodies.

* **Limitations:** The model struggled to correctly classify less frequent and highly specific land use categories (e.g., roads, cultivation, habitations, logging, and mines).

* **Future Work:** As discussed in the report, future improvements should address the class imbalance by implementing mosaic data augmentation, weighted loss functions, or ensemble modeling.