# Predicting-finding-a-new-home-with-Tensorflow-and-Keras
# Pet Adoption Prediction

This project implements a deep learning model for pet adoption rate prediction using TensorFlow and Keras. And in fact, the purpose of this project is to compare these two libraries

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

The purpose of this project is to compare two deep learning libraries, namely keras and tensorflow, and as you can see, in this project, after performing the necessary pre-processing on a dataset, we used each type of library separately, so that the difference of doing the project at a low level (with keras) and be well visible at the hight level (with tensorflow).

## Features

- Data preprocessing using pandas and scikit-learn
- Deep learning model implementation using Keras
- Model training and evaluation
- Custom TensorFlow model class for fine-tuning

## Installation

Before running the project, ensure you have the following software installed:

- Python 3.7+
- TensorFlow 2.x
- Pandas
- NumPy
- Scikit-learn
- Keras
- Category Encoders

You can install the necessary packages using pip:

bash
pip install tensorflow pandas numpy scikit-learn keras category-encoders

## Data Preprocessing

The dataset is loaded from a CSV file. Before training the model, various preprocessing steps are performed:

1. Data Loading: The dataset is loaded using pandas:
 

python
   train = pd.read_csv('/content/drive/MyDrive/Quera/new_home_tensorflow/data/petfinder_train.csv')
  

2. Target Mapping: The 'AdoptionSpeed' column is mapped to a binary target indicating whether a pet is adopted quickly.
 

python
   mapping = {0: 1, 1: 1, 2: 1, 3: 1, 4: 0}
  

3. Feature Encoding: Categorical features are encoded using
LabelEncoder
and
BinaryEncoder
for non-numeric columns.

4. Data Normalization: All features are normalized to have a mean of 0 and a standard deviation of 1.

5. Train-Validation Split: The dataset is split into training and validation sets using
train_test_split
.

## Model Architecture

The project implements two models: one using Keras Sequential API and another using a custom TensorFlow model.

### Keras Model

The Keras model architecture is defined as follows:
- Input Layer: Shape of 26 features
- Three Dense Layers: Each with ReLU activation
- Output Layer: Dense layer with a sigmoid activation for binary classification

### Custom TensorFlow Model

A custom model class
CustomModel
is implemented, which contains:
- Dense layers with weight initialization
- Forward propagation logic with ReLU and sigmoid activations
- Custom training loop

## Training the Model

The Keras model is trained using the fit method:

python
history = model1.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=epochs, validation_data=(X_valid, y_valid))

The custom TensorFlow model is trained with a defined method that includes loss calculation and validation metrics (F1 Score) after each epoch.
`


python
model.fit(train_dataset, valid_dataset, epochs=EPOCHS, BATCH_SIZE=BATCH_SIZE)
`


## Usage

To use the model:
1. Prepare your dataset in the specified format.
2. Preprocess the dataset according to the methods shown in the `Data Preprocessing` section.
3. Train the model using the provided training routines.
4. Use the model to make predictions on new data.
