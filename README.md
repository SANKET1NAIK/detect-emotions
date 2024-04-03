# This repository contains the implementation of a computer vision model for emotion detection from facial images using convolutional neural networks (CNNs).

Dataset
The dataset used for this project is a pre-processed facial image dataset labeled with emotions such as happy, sad, angry, and neutral. It is split into training, validation, and test sets.

Model Architecture:
The convolutional neural network (CNN) architecture designed for emotion detection comprises several convolutional layers followed by max-pooling layers for feature extraction. The final layers consist of fully connected layers with softmax activation for emotion classification. The input images are preprocessed to grayscale and resized to a fixed size.

Training:
The model undergoes training using the training dataset with hyperparameters optimized to achieve the best performance within the given time frame. The Adam optimizer is employed with categorical crossentropy loss, and accuracy is monitored during training.

Evaluation:
The performance evaluation of the trained model is conducted on the validation set, utilizing accuracy as the primary metric for emotion detection.

Testing:
Upon completion of training, the trained model is deployed to predict emotions for the final test set images. The predicted emotions are then compared with the ground truth labels, and the evaluation results are shared.

Validation Accuracy: 0.538777232170105
Test Accuracy: 0.47157894736842104

i had only done with 10 epochs for good accuracy we have to train our model for more time then the accuracy will increase
