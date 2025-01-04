# Fashion Recommendation System

This project implements a Fashion Recommendation System that uses a pre-trained VGG16 model to generate embeddings for fashion items and recommend similar items based on image similarity.

## Dataset
kaggel : https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
## Requirements

Ensure you have the following libraries installed:

```bash
numpy
pandas
matplotlib
opencv-python
tensorflow
keras
scikit-learn
pathlib
```

Install them using pip if not already installed:

```bash
pip install numpy pandas matplotlib opencv-python tensorflow keras scikit-learn
```

## Project Structure

### Load Required Libraries

```python
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pathlib
import tensorflow as tf
import tensorflow.keras as keras
from keras.applications import VGG16
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from sklearn.metrics.pairwise import linear_kernel
```

### Define Dataset Path

Specify the dataset path containing images and metadata:

```python
path = 'C:\\Users\\off1c\\OneDrive\\Desktop\\Datsets\\Fashion\\'
dataset_path = pathlib.Path(path)
dirs_names = os.listdir(dataset_path)
print(dirs_names)
```

### Display Sample Images

Visualize a subset of images:

```python
plt.figure(figsize=(20,20))
for i in range(20, 30):
    plt.subplot(6, 10, i - 19)
    cloth_img = mpimg.imread(path + 'images/100' + str(i) + '.jpg')
    plt.imshow(cloth_img)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
```

### Load Metadata

Load metadata from a CSV file:

```python
styles_df = pd.read_csv(path + "styles.csv", nrows=6000, on_bad_lines='skip')
styles_df['image'] = styles_df.apply(lambda x: str(x['id']) + ".jpg", axis=1)
print(styles_df.shape)
styles_df.head(5)
```

### Prepare the VGG16 Model

Load the pre-trained VGG16 model and set it to non-trainable:

```python
img_width, img_height, chnls = 100, 100, 3
vgg16 = VGG16(include_top=False, weights='imagenet', input_shape=(img_width, img_height, chnls))
vgg16.trainable = False
vgg16_model = keras.Sequential([vgg16, GlobalMaxPooling2D()])
vgg16_model.summary()
```

### Define Utility Functions

#### Generate Image Path

```python
def img_path(img):
    return path + 'images/' + img
```

#### Predict Embeddings

```python
def predict(model, img_name):
    img = image.load_img(img_path(img_name), target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.vgg16.preprocess_input(img)
    return model.predict(img)
```

#### Generate Embeddings for Dataset

```python
def get_embeddings(df, model):
    df_embeddings = df['image'].apply(lambda x: predict(model, x).reshape(-1))
    df_embeddings = df_embeddings.apply(pd.Series)
    return df_embeddings

df_embeddings = get_embeddings(styles_df, vgg16_model)
```

### Visualize a Sample Image

```python
url = r"C:\\Users\\off1c\\OneDrive\\Desktop\\Datsets\\Fashion\\images\\22647.jpg"
a = mpimg.imread(url)
plt.imshow(a)
plt.show()
```

### Compute Similarity

#### Predict for Sample Image

```python
sample_image = predict(vgg16_model, '22647.jpg')
sample_image.shape
```

#### Compute Similarity Matrix

```python
sample_similarity = linear_kernel(sample_image, df_embeddings)
```

#### Normalize Similarity

```python
def normalize_sim(similarity):
    x_min = similarity.min(axis=1)
    x_max = similarity.max(axis=1)
    norm = (similarity - x_min) / (x_max - x_min)[:, np.newaxis]
    return norm

sample_similarity_norm = normalize_sim(sample_similarity)
```

### Generate Recommendations

```python
def get_recommendation(df, similarity, top_n=5):
    sim_scores = list(enumerate(similarity[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:top_n]
    cloth_indices = [i[0] for i in sim_scores]
    return df['image'].iloc[cloth_indices]

recommendation = get_recommendation(styles_df, sample_similarity_norm)
recommendation_list = recommendation.to_list()
```

### Display Recommendations

```python
plt.figure(figsize=(20, 20))
j = 0
for i in recommendation_list:
    plt.subplot(6, 10, j + 1)
    cloth_img = mpimg.imread(path + 'images/' + i)
    plt.imshow(cloth_img)
    plt.axis("off")
    j += 1
plt.title("Recommendation images", loc='right')
plt.subplots_adjust(wspace=-0.5, hspace=1)
plt.show()
```

### Results

Upload a sample image, and the system will display the top 5 recommended images based on similarity.

