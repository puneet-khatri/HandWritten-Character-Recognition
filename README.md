# Handwritten Digit Recognition using Pygame and Keras

This project demonstrates a handwritten digit recognition system using Pygame for drawing digits and a trained Keras convolutional neural network (CNN) to recognize and classify the drawn digits.

## Overview

The project consists of two main files:

- `main.py`: Contains the code for training the CNN model using the MNIST dataset and saving the best performing model.
- `app.py`: Implements a Pygame interface where users can draw digits. The drawn digit is then recognized using the saved model and displayed on the interface.

## Requirements

Ensure you have the following dependencies installed:

- Python 3.x
- Keras
- Pygame
- NumPy
- OpenCV (cv2)
- Matplotlib

Install dependencies using pip:

```bash
pip install keras pygame numpy opencv-python matplotlib
```

## Usage

### Training the Model

1. To train the model, navigate to the project directory in the terminal.
2. Run the following command:

```bash
python main.py
```

3. This script will load the MNIST dataset, preprocess the data, train a CNN model, and save the best performing model as `bestmodel.h5`.

### Running the Application

1. To run the application, execute the following command:

```bash
python app.py
```

2. This will open a Pygame window where you can draw digits using the mouse.
3. Release the mouse to trigger digit recognition.
4. The recognized digit will be displayed on the screen.

### Keyboard Commands in the Application

- Press `n` to clear the drawing board.

## Project Structure

- `main.py`:
  - Loads the MNIST dataset.
  - Preprocesses the data and trains a CNN model.
  - Saves the best performing model as `bestmodel.h5`.
  
- `app.py`:
  - Initializes a Pygame window for drawing digits.
  - Recognizes drawn digits using the trained model.
  - Displays the recognized digit on the interface.

## Acknowledgments

The MNIST dataset used for training the model is sourced from the Keras library.

## Notes

- Ensure that the required models (`bestmodel.h5`) are present in the respective directories before running the application.

---

This detailed README provides instructions on how to set up, train the model, and run the application, along with keyboard commands and dependencies. Adjustments can be made to this file to include more information or specific instructions tailored to your project's needs.
