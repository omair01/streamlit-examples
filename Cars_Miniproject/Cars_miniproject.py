#!/usr/bin/env python
# coding: utf-8

# # Importing the Libraries
# 

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# # Loading the Dataset

# In[43]:


dataset = 'cars196'


# In[44]:


data = tfds.load(dataset, split ='train', shuffle_files=True)


# # Exploring the dataset

# In[39]:


import tensorflow_datasets as tfds

dataset, info = tfds.load('cars196', split='train+test',  with_info=True, shuffle_files=True)

# Prefetch the dataset
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Convert the dataset to a NumPy array
numpy_array = list(dataset.take(1).as_numpy_iterator())[0]

# Access the first image and label using standard indexing
image = numpy_array['image']
label = numpy_array['label']


# In[40]:


image = next(iter(dataset))['image']


# In[41]:


import matplotlib.pyplot as plt

plt.imshow(image)
plt.show()


# In[49]:



# Print the dataset info
print(info)

# Print the number of images in the dataset
num_images = info.splits['train'].num_examples + info.splits['test'].num_examples
print('Number of images:', num_images)

# Print the number of classes in the dataset
num_classes = info.features['label'].num_classes
print('Number of classes:', num_classes)

# Plot a histogram of the class distribution
class_names = [info.features['label'].int2str(i) for i in range(num_classes)]
class_counts = [0] * num_classes




# In[56]:


import seaborn as sns
plt.figure(figsize=(12, 6))
sns.barplot(x=class_names, y=class_counts)
plt.xticks(rotation=90)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.show()


# In[58]:


# Plot a sample of images from the dataset
fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(10, 6))

for i, example in enumerate(dataset.take(15)):
    image = example['image'].numpy()
    label = example['label'].numpy()
    ax = axes[i//5, i%5]
    ax.imshow(image)
    ax.set_title(class_names[label])
    ax.axis('off')

plt.tight_layout()
plt.show()


# # Preprocessing the images

# In[68]:


import tensorflow_datasets as tfds
import tensorflow as tf

# Load the cars196 dataset
train_dataset, info = tfds.load('cars196', split='train', with_info=True, shuffle_files=True)
test_dataset = tfds.load('cars196', split='test', shuffle_files=True)

# Define preprocessing functions
IMG_SIZE = 224

def preprocess_train(example):
    image = tf.cast(example['image'], tf.float32)
    label = example['label']
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    return image, label

def preprocess_test(example):
    image = tf.cast(example['image'], tf.float32)
    label = example['label']
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.resnet_v2.preprocess_input(image)
    return image, label

# Apply preprocessing functions to datasets
train_dataset = train_dataset.map(preprocess_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.map(preprocess_test, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Batch and prefetch datasets
BATCH_SIZE = 32

train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)


# # Creating the model
# 

# In[69]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(196, activation='softmax')
])


# # Traning the model

# In[71]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_dataset, epochs=10)


# In[73]:


#Predicting the model 
#model(image)


# In[74]:


#Saving the model
model.save('cars_model.h5')


# # Creating the Streamlit App

# In[76]:


get_ipython().system('pip install streamlit')


# In[ ]:


get_ipython().run_cell_magic('writefile', 'app.py', 'import streamlit as st\nfrom PIL import Image\n\n@st.cache(allow_output_mutation=True)\ndef load_model():\n    model = tf.keras.models.load_model(\'cars_model.h5\')\n    return model\n\nmodel = load_model()\n\ndef predict(image):\n    image = tf.image.resize(image, [224, 224])\n    image = tf.keras.preprocessing.image.img_to_array(image)\n    image = image / 255.0\n    image = np.expand_dims(image, axis=0)\n    prediction = model.predict(image)\n    prediction = np.argmax(prediction)\n    return prediction\n\nst.title("Cars196 Image Classifier")\n\nuploaded_file = st.file_uploader("Choose an image...", type="jpg")\nif uploaded_file is not None:\n    image = Image.open(uploaded_file)\n    st.image(image, caption=\'Uploaded Image.\', use_column_width=True)\n    prediction = predict(image)\n    st.write("Prediction:", info.features[\'label\'].int2str(prediction))')


# In[ ]:


get_ipython().system('streamlit run app.py')


# In[ ]:




