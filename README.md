# HyperParameter-Tuning
# Fashion MNIST Classification with Hyperparameter Tuning

This repository contains code for training and tuning neural networks on the Fashion MNIST dataset. The goal is to achieve the highest accuracy possible on a test set of 10,000 images.

## Dataset
The Fashion MNIST dataset consists of 60,000 training images and 10,000 test images. Each image is a 28x28 grayscale image belonging to one of 10 classes.

## Model Architecture
The neural network architecture consists of densely connected layers with ReLU activation functions and dropout layers to prevent overfitting. The final layer uses softmax activation for multi-class classification.

## Training Procedure
1. **Data Preparation:** The dataset is loaded using TensorFlow's built-in `fashion_mnist` dataset module. Pixel values are normalized to be between 0 and 1 and reshaped into 1D arrays.

2. **Model Creation:** The `create_model` function defines a neural network with configurable hyperparameters such as hidden layer size, dropout rate, and learning rate. The model is compiled with the Adam optimizer and sparse categorical crossentropy loss.

3. **Hyperparameter Tuning:** Hyperparameter tuning is performed using grid search over a predefined parameter grid. The grid includes different combinations of hidden layer sizes, dropout rates, and learning rates.

4. **Validation:** The training dataset is split into training and validation sets. The model is trained on the training set and evaluated on the validation set. Early stopping is used to prevent overfitting.

5. **Evaluation:** After training, the model with the best performance on the validation set is selected. The selected model is evaluated on the test set to obtain the final accuracy.

## Results
The best performing model achieved an accuracy of 87.98 on the test set. The hyperparameters for this model are:
- Hidden Layer Size: 128
- Dropout Rate: 0.2
- Learning Rate: 0.001

## Usage
You can use this code to train and evaluate different neural network architectures on the Fashion MNIST dataset. Feel free to modify the hyperparameter grid or experiment with different architectures to improve performance.

