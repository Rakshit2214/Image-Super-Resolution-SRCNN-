#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from PIL import Image

data_dir='AnalyticsArena_DataSet/HighResolution_Train'
hr_filenames=[os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith('.jpg')]

data_dir='AnalyticsArena_DataSet/LowReolution_3x_Train'
lr_filenames=[os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith('.jpg')]

train_indices = min(len(hr_filenames),len(lr_filenames))
  # Save the image pairs as numpy arrays
train_hr = []
train_lr = []
val_hr = []
val_lr = []
test_hr = []
test_lr = []
    
for idx in range(1500):
      hr_img = Image.open(hr_filenames[idx])
      lr_img = Image.open(lr_filenames[idx])
      train_hr.append(np.array(hr_img))
      train_lr.append(np.array(lr_img))

np.savez('train.npz', hr=train_hr, lr=train_lr)

data_dir='AnalyticsArena_DataSet/HighResolution_Valid'
hr_filenames=[os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith('.jpg')]

data_dir='AnalyticsArena_DataSet/LowResolution_3x_Valid'
lr_filenames = [os.path.join(data_dir, f) for f in sorted(os.listdir(data_dir)) if f.endswith('.jpg')]

val_indices = min(len(hr_filenames),len(lr_filenames))
for idx in range(800):
      hr_img = Image.open(hr_filenames[idx])
      lr_img = Image.open(lr_filenames[idx])
      val_hr.append(np.array(hr_img))
      val_lr.append(np.array(lr_img))
np.savez('val.npz', hr=val_hr, lr=val_lr)      


print(f'images: {len(train_hr)} training, {len(val_hr)} validation, and {len(test_hr)} testing images')

    


# In[2]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2DTranspose, Conv2D, Activation, BatchNormalization, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError


# In[3]:


# Load training and testing data
train_data_hr = np.load('train.npz')
train_data_hr =train_data_hr ['hr']

train_data_lr = np.load('train.npz')
train_data_lr=train_data_lr['lr']

test_data_hr = np.load('val.npz')
test_data_hr =test_data_hr ['hr']

test_data_lr = np.load('val.npz')
test_data_lr=test_data_lr['lr']


# In[4]:


# def scaling(input_image):
#     input_image = input_image / 255.0
#     return input_image

# Scale from (0, 255) to (0, 1)
# y_norm_train = (train_data_hr-np.min(train_data_hr))/(np.max(train_data_hr)-np.min(train_data_hr))
# x_norm_train = (train_data_lr-np.min(train_data_lr))/(np.max(train_data_lr)-np.min(train_data_lr))
# y_norm_test = (test_data_hr-np.min(test_data_hr))/(np.max(test_data_hr)-np.min(test_data_hr))
# x_norm_test = (test_data_lr-np.min(test_data_lr))/(np.max(test_data_lr)-np.min(test_data_lr))


# convert from integers to floats
train_data_hr = train_data_hr.astype('float32')
train_data_lr = train_data_lr.astype('float32')
test_data_hr = test_data_hr.astype('float32')
test_data_lr = test_data_lr.astype('float32')
# normalize to the range 0-1
train_data_hr /= 255.0
train_data_lr /= 255.0
test_data_hr /= 255.0
test_data_lr /= 255.0




# In[5]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, UpSampling2D

# define the model architecture
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(None, None, 3)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D(size=(3, 3)))

# compile the model
model.compile(optimizer='adam', loss='mae')

# train the model
model.fit(train_data_lr, train_data_hr, validation_data=(test_data_lr, test_data_hr), batch_size=4, epochs=30)


# In[ ]:


radius = 3
amount = 1.5
threshold = 0

# Apply the unsharp mask filter to the predicted values
sharpened_predictions = np.zeros_like(y_test)
for i in range(X_test.shape[0]):
    img = X_test[i]
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    sharpened = cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)
    sharpened = np.clip(sharpened, 0, 1)
    sharpened_predictions[i] = np.argmax(model.predict(np.expand_dims(sharpened, axis=0)))

# Evaluate the performance of the sharpened predictions
accuracy = np.mean(sharpened_predictions == y_test)
print(f'Sharpened model accuracy: {accuracy:.4f}')


# In[6]:


model.save('srcnn_model_6.h5')


# In[7]:


y_pred=model.predict(test_data_lr)


# In[11]:


from matplotlib import pyplot as plt
plt.imshow(test_data_hr[1], interpolation='nearest')
plt.show()


# In[12]:


from matplotlib import pyplot as plt
plt.imshow(y_pred[1], interpolation='nearest')
plt.show()


# In[10]:


get_ipython().system('pip install scikit-image')


# In[11]:


import os

# Create a folder named "predicted_images" in the current working directory
folder_path = "predicted_images"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


# In[18]:


import os
from PIL import Image

# Assuming y_predicted is a NumPy array of predicted images
for i in range(y_pred.shape[0]):
    # Convert the i-th predicted image to a PIL Image object
    img = Image.fromarray(y_pred[i])
    # Save the i-th predicted image to the folder
    img.save(os.path.join(folder_path, f"predicted_{i}.png"))


# In[ ]:




