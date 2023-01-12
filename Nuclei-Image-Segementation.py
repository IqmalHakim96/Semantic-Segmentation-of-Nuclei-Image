#%%
#1. Import necessary packages

import cv2,os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow.keras.callbacks import TensorBoard
from tensorflow_examples.models.pix2pix import pix2pix

#%% Data Preparation
# Data Loading
train_path = r"C:\Users\USER\Desktop\STEP_AI01_Muhammad_Iqmal_Hakim_Assesment_4\data-science-bowl-2018-2\train"
test_path = r"C:\Users\USER\Desktop\STEP_AI01_Muhammad_Iqmal_Hakim_Assesment_4\data-science-bowl-2018-2\test"

#%%
# Prepare empty list to hold the data

train_images = []
train_masks = []
test_images = []
test_masks = []

# Load Images & Masks for train & test using openCV
train_image_dir = os.path.join(train_path, "inputs")
for train_image in os.listdir(train_image_dir):
    img = cv2.imread(os.path.join(train_image_dir, train_image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128,128))
    train_images.append(img)
    
train_masks_dir = os.path.join(train_path, "masks")
for train_mask in os.listdir(train_masks_dir):
    mask = cv2.imread(os.path.join(train_masks_dir, train_mask), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (128,128))
    train_masks.append(mask)

test_image_dir = os.path.join(test_path, 'inputs')
for test_image in os.listdir(test_image_dir):
    img = cv2.imread(os.path.join(test_image_dir, test_image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128,128))
    test_images.append(img)
    
test_masks_dir = os.path.join(test_path, "masks")
for test_mask in os.listdir(test_masks_dir):
    mask = cv2.imread(os.path.join(test_masks_dir, test_mask), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (128,128))
    test_masks.append(mask)

#%%
#Convert the list of np array into a np array
train_images_np = np.array(train_images)
train_masks_np = np.array(train_masks)

test_images_np = np.array(test_images)
test_masks_np = np.array(test_masks)

#%% Data preprocessing
# Expand the mask dimension
train_masks_np_exp = np.expand_dims(train_masks_np, axis=-1)
test_masks_np_exp = np.expand_dims(test_masks_np, axis=-1)
#%%

# Convert the mask value from [0,255] into [0,1]
train_conv_masks = np.round(train_masks_np_exp/255)
train_conv_masks = 1 - train_conv_masks

test_conv_masks = np.round(test_masks_np_exp/255)
test_conv_masks = 1 - test_conv_masks

#%%
# Normalize the images
train_conv_images = train_images_np/255.0
test_conv_images = test_images_np/255.0

#%% Perform Train-Test Split
#Seems like the dataset already have train test dataset, the train-test-split will not be done.

#Convert numpy arrays to tensor slices
X_train = tf.data.Dataset.from_tensor_slices(train_conv_images)
X_test = tf.data.Dataset.from_tensor_slices(test_conv_images)
y_train = tf.data.Dataset.from_tensor_slices(train_conv_masks)
y_test = tf.data.Dataset.from_tensor_slices(test_conv_masks)

#%%
#Combine the images and masks using Zip method
train_dataset = tf.data.Dataset.zip((X_train, y_train))
test_dataset = tf.data.Dataset.zip((X_test,y_test))

#%%
# Build dataset
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE
BUFFER_SIZE = 1000
STEP_PER_EPOCH = 800 // BATCH_SIZE
VALIDATION_STEPS = 200 // BATCH_SIZE
train_batches = (
    train_dataset
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .prefetch(buffer_size = tf.data.AUTOTUNE)
)
test_batches = test_dataset.batch(BATCH_SIZE)
#%%
# Visualize some pictures as example
def display(display_list):
    plt.figure(figsize=(15,15))
    title = ["Input Image", "True Mask", "Predicted Mask"]
    
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()
    
for images, masks in train_batches.take(2):
    sample_image, sample_mask = images[0], masks[0]
    display([sample_image,sample_mask])
    
#%% Model Development
# Use pretrained model as the feature extractor
base_model = tf.keras.applications.MobileNetV2(input_shape = [128,128,3], include_top = False)
# Use these activation layers as the outputs from the feature extractor(some of these outputs will be used to perform contenation at the upsampling path)
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

#Instantiate the feature extractor
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

#Define the upsampling path
up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

# Use functional API to construct the entire U-net
def unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[128, 128, 3])

  # Downsample through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # #Build the upsampling path and establish the concatenatio
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()
    x = concat([x, skip])

  # Use a transpose convolution layer to perform the last upsampling, this will become output layer
  last = tf.keras.layers.Conv2DTranspose(filters=output_channels, kernel_size=3, strides=2, padding='same')  #64x64 -> 128x128
  outputs = last(x)
  
  model = keras.Model(inputs=inputs,outputs=outputs)
  
  return model

# Use the function to create the model
OUTPUT_CLASSES = 2
model = unet_model(output_channels= OUTPUT_CLASSES)

#%% Model Compile
losses = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer = 'adam', loss = losses, metrics=["accuracy"])
keras.utils.plot_model(model)
model.summary()

#%%
# Create function to show predicitons

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)[0]])
        else:
            display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis,...]))[0]])

#%% 
#Create a callback function to make use of the show_predictions function
class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print("\n Sample prediction after epoch {}\n".format(epoch+1))
        
# Tensorboard callbacks 
LOGS_PATH = os.path.join(os.getcwd(),'Logs',datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = TensorBoard(log_dir = LOGS_PATH)

#%% Model training
EPOCH = 10

history = model.fit(train_batches, epochs = EPOCH, steps_per_epoch = STEP_PER_EPOCH, validation_steps= VALIDATION_STEPS, validation_data= test_batches, callbacks=[DisplayCallback(),tb]) 

#%% Model Deployment
show_predictions(test_batches, 3)

#%% Model Saving
#save trained tf model
model.save('model.h5')