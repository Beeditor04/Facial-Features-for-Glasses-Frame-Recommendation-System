# Import necessary libraries
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Create an instance of ImageDataGenerator with specified augmentations
datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True)

def feature_engineering_and_augmentation(path, img_size=(224, 224)):
    labels = []
    images = []
    shape_types = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
    
    for shape in shape_types:
        print(f'Loading {shape} images...')
        shape_dir = os.path.join(path, shape)
        if not os.path.exists(shape_dir):
            print(f"Directory {shape_dir} does not exist.")
            continue
        for imgName in os.listdir(shape_dir):
            img_path = os.path.join(shape_dir, imgName)
            if not os.path.isfile(img_path):
                print(f"File {img_path} does not exist.")
                continue
            img = cv2.imread(img_path)
            if img is None:
                print(f"Image {img_path} could not be read.")
                continue
            # Resize image to the desired size (224, 224)
            img = cv2.resize(img, img_size)
            # Normalize image to range [0, 1]
            img = img.astype('float32') / 255.0
            images.append(img)
            labels.append(shape_types.index(shape))

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = to_categorical(labels, num_classes=len(shape_types))

    # Augment the images
    augmented_images = []
    for img in images:
        img = np.expand_dims(img, axis=0)  # Prepare the image for the generator
        augmented_iter = datagen.flow(img, batch_size=1)
        augmented_img = next(augmented_iter)[0]  # Get the augmented image
        augmented_images.append(augmented_img)

    augmented_images = np.array(augmented_images)

    return augmented_images, labels
