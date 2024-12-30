# Forest Fire Detection

This project demonstrates a binary classification model for detecting forest fires from images using Convolutional Neural Networks (CNNs) built with TensorFlow/Keras. The model predicts whether an input image indicates fire or no fire.
## Dataset : https://www.kaggle.com/datasets/brsdincer/wildfire-detection-image-data
## Table of Contents
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training and Validation](#training-and-validation)
- [Testing](#testing)
- [Results](#results)

## Dataset
The dataset is organized into the following structure:
```
forest_fire/
├── Training and Validation/
│   ├── fire/          # Contains images of fire
│   └── nofire/        # Contains images of no fire
└── Testing/
    ├── fire/          # Contains images of fire
    └── nofire/        # Contains images of no fire
```
Ensure that the dataset is placed at the appropriate path as specified in the code.

## Dependencies
Install the required libraries before running the code:
```bash
pip install tensorflow numpy matplotlib opencv-python
```

## Usage
1. Place the dataset in the correct directory structure as described in the "Dataset" section.
2. Run the code to train the model and test its predictions.

### Key Functions
- `predictImage(filename)`: Predicts whether the input image has a fire or not and displays the result.

### Example
To predict an image:
```python
predictImage(r"C:\path_to_image\image.jpg")
```

## Model Architecture
The model is a sequential CNN with the following layers:
- **Convolutional layers**: Extract features from input images.
- **Pooling layers**: Reduce spatial dimensions and computation.
- **Dense layers**: Fully connected layers for classification.
- **Activation functions**: `ReLU` for intermediate layers and `Sigmoid` for the output layer.

### Code Snippet:
```python
model = keras.Sequential([
    keras.layers.Input(shape=(150, 150, 3)),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
```

## Training and Validation
The model is trained using the following:
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy

### Code Snippet:
```python
r = model.fit(train_dataset, epochs=5, validation_data=test_dataset)
```

## Testing
The model predicts on the test dataset and individual images using the `predictImage` function. Images are preprocessed to match the input shape required by the model.

### Code Snippet:
```python
val = model.predict(X)
if val == 1:
    plt.xlabel("No Fire", fontsize=30)
elif val == 0:
    plt.xlabel("Fire", fontsize=30)
```

## Results
The training and validation results are plotted for loss and accuracy metrics using Matplotlib.

### Example Output:
```python
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
```

## Notes
- Ensure all file paths are correct.
- The dataset should be well-balanced for optimal performance.
- Adjust the number of epochs and batch size as needed based on computational resources.
