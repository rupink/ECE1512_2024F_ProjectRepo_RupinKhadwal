
## Overview

This project contains two Jupyter notebooks analyzing and performing machine learning tasks on two datasets: the **MHIST dataset** and the **MNIST dataset**. These notebooks implement deep learning models using PyTorch to classify images, evaluate model performance, and visualize results.

## Files

1. **mhist_analysis.ipynb**
   - This notebook focuses on the **MHIST dataset**, which is a histopathology image dataset commonly used for medical image analysis. The analysis likely includes tasks such as data loading, preprocessing, training a convolutional neural network (CNN), and evaluating the model's accuracy in classifying medical images. This type of task helps in automating medical diagnosis through machine learning.
   
   - Main sections:
     - **Data Loading**: Loading the MHIST dataset and preparing it for training.
     - **Model Definition**: Defining a convolutional neural network (CNN) architecture suitable for image classification tasks.
     - **Training the Model**: Training the model on the MHIST dataset using PyTorch's optimization methods.
     - **Evaluation**: Evaluating the performance of the model and visualizing metrics such as accuracy and loss.

2. **mnist_analysis.ipynb**
   - This notebook performs an analysis on the **MNIST dataset**, which is a widely used dataset of handwritten digits for machine learning and computer vision tasks. The notebook includes steps such as data loading, model training, and evaluation.
   
   - Main sections:
     - **Data Loading**: Downloading and preparing the MNIST dataset, which consists of images of handwritten digits (0–9).
     - **Model Definition**: Setting up a CNN or other neural network architecture to classify the images.
     - **Training**: Training the model with several epochs to improve classification accuracy.
     - **Evaluation**: Evaluating the model on a test set and reporting metrics such as loss and accuracy.

This is done multiple times in each book to analyze different hyper parameters set

## Requirements

- Python 3.x
- Jupyter Notebook
- PyTorch
- NumPy
- Matplotlib (for visualizations)
- Torchvision (for datasets and image transformations)

You can install the required packages by running the following command:

```bash
pip install torch torchvision numpy matplotlib
```

## Image Directories

In the cells there are img pathways for the mhist dataset. Please provide the path for those files when running locally.

## Usage

1. Clone the repository or download the notebooks.
2. Open the notebooks using Jupyter by running:
   ```bash
   jupyter notebook
   ```
3. Navigate to either `mhist_analysis.ipynb` or `mnist_analysis.ipynb` and run the cells sequentially.

## Notes

- Both notebooks include sections where the datasets are loaded, and data transformations are applied.
- The models are trained using standard machine learning techniques like **Stochastic Gradient Descent (SGD)** with data augmentation options available.
- Ensure you have sufficient hardware resources (e.g., GPU) if you plan on running the full training pipelines for better performance.

## Results

- The notebooks print out results such as training and testing accuracy and loss after each epoch.
- The MHIST analysis aims at improving medical image classification, which has applications in healthcare and diagnostics.
- The MNIST analysis serves as an introduction to CNN-based image classification tasks, often used in introductory machine learning courses.