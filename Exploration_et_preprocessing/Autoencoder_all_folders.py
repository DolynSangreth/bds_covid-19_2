import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Importation des données 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import tensorflow as tf
import pathlib

import os
import numpy as np
from PIL import Image
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# === PARAMÈTRES ===
IMG_SIZE = 256 
encoding_dim = 50
COVID_DIR = r"C:\Users\ambro\Desktop\Projet Metier\dataset_original_images\COVID"
NORMAL_DIR = r"C:\Users\ambro\Desktop\Projet Metier\dataset_original_images\Normal"
LUNG_OP_DIR = r"C:\Users\ambro\Desktop\Projet Metier\dataset_original_images\Lung_Opacity"
VIRAL_PNEUM_DIR = r"C:\Users\ambro\Desktop\Projet Metier\dataset_original_images\Viral_Pneumonia"
image_extensions = ('.png', '.jpg', '.jpeg')

# === CHARGEMENT ET PRÉPARATION DES IMAGES ===
## Listing des chemins d'accès
DIR_LISTE = [COVID_DIR, NORMAL_DIR, LUNG_OP_DIR, VIRAL_PNEUM_DIR]

for folder_path in DIR_LISTE:
    images = []
    filenames = []

    for fname in os.listdir(folder_path):
        if fname.lower().endswith(image_extensions):
            img_path = os.path.join(folder_path, fname)
            img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='grayscale')
            img_array = img_to_array(img).astype('float32') / 255.0
            images.append(img_array.reshape(-1))  # Aplatir
            filenames.append(img_path)

    X = np.array(images)

    # === AUTOENCODEUR ===
    input_img = Input(shape=(IMG_SIZE * IMG_SIZE,))
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(IMG_SIZE * IMG_SIZE, activation='sigmoid')(decoded)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # === ENTRAÎNEMENT ===
    autoencoder.fit(X, X,
                    epochs=20,
                    batch_size=32,
                    shuffle=True)

    # === TRANSFORMATION DES IMAGES ===
    X_decoded = autoencoder.predict(X)

    # === ENREGISTREMENT DES IMAGES ÉCRASÉES ===
    for img_array, path in zip(X_decoded, filenames):
        img_reshaped = img_array.reshape((IMG_SIZE, IMG_SIZE))  # Remettre en forme
        img_pil = array_to_img(np.expand_dims(img_reshaped, axis=-1))  # Ajouter canal
        img_pil.save(path)  # Remplace l’original
