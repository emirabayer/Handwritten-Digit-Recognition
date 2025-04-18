# Handwritten Digit Recognition

This project is an interactive application that lets the user draw digits (0-9) with a mouse and uses a Convolutional Neural Network (CNN) to predict the drawn digit in real-time. Built with Python, Pygame, and Keras, it combines a simple drawing interface with a pre-trained MNIST model to recognize handwritten digits. (99.02% Overall Accuracy)


<p align="center">
   
![Image](https://github.com/user-attachments/assets/275cf419-fc5b-447e-b435-f44e4980f11f)

| Digits | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| 0      | 0.9929    | 0.9929 | 0.9929   | 980     |
| 1      | 0.9930    | 0.9947 | 0.9938   | 1135    |
| 2      | 0.9893    | 0.9884 | 0.9889   | 1032    |
| 3      | 0.9930    | 0.9891 | 0.9911   | 1010    |
| 4      | 0.9889    | 0.9969 | 0.9929   | 982     |
| 5      | 0.9673    | 0.9955 | 0.9812   | 892     |
| 6      | 0.9958    | 0.9843 | 0.9900   | 958     |
| 7      | 0.9836    | 0.9922 | 0.9879   | 1028    |
| 8      | 0.9907    | 0.9815 | 0.9861   | 974     |
| 9      | 0.9919    | 0.9722 | 0.9820   | 1009    |

_See the confusion matrix at the end of this README_
</p>


## Features
- **Draw Digits**: Using a mouse to draw digits on a 640x480 canvas.
- **Real-Time Prediction**: A pre-trained CNN predicts the digit as soon as drawing is finished.
- **Visual Feedback**: The predicted digit is displayed above a red bounding box around the drawing.
- **Clear Canvas**: Pressing Backspace resets the board to start over.
- **Customizable**: Easy to tweak for different models or drawing styles.

## How It Works
1. **Drawing Interface**: 
   - Built with Pygame, the app creates a window where the user can draw by holding the left mouse button.
   - White circles (4-pixel radius) are drawn at mouse positions, mimicking handwriting.

2. **Image Processing**:
   - When the mouse is released, the app finds the bounding box around the drawing.
   - The drawn area is extracted, resized to 28x28 pixels (the size MNIST models expect), and normalized to a 0-1 range.

3. **Prediction**:
   - The processed image is fed into a pre-trained CNN (trained on the MNIST dataset).
   - The model outputs a probability for each digit (0-9), and the highest one is selected.
   - The prediction (e.g., "Five" for 5) is displayed in red above the drawing.

4. **Reset**: Pressing Backspace clears the screen for a new drawing.

## Model Details
- **Architecture**: A simple CNN with:
  - 2 Conv2D layers (32 and 64 filters, 3x3 kernels, ReLU activation).
  - 2 MaxPooling2D layers (2x2 pool size).
  - Flatten, Dropout (0.25), and Dense (10 units, softmax) layers.
- **Training**: Trained on the MNIST dataset (60,000 training, 10,000 test images) for 5 epochs with early stopping.
- **Accuracy**: Achieves ~99% accuracy on the test set.
- **File**: `bestmodel.h5` (included or trainable via the notebook in `training/`).

## Requirements
- Python 3.6+
- Required libraries for the application: <br/>
  numpy, keras, pygame, opencv-python

- Additional libraries necessary for retraining and analyzing the model: <br/>
  matplotlib, sklearn, seaborn

  
![digit_detector_confusion_matrix](https://github.com/user-attachments/assets/46a2e3b5-2bf4-4a99-94b7-75689d4cdfbf)
