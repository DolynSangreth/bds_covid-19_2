import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, accuracy_score, recall_score

# Définir le chemin de base vers le dossier contenant les images par catégorie
base_dir = r"C:\Users\ambro\Desktop\Projet Metier\dataset_laplacien_30"

# Créer un dossier temporaire pour stocker les 1000 images sélectionnées par catégorie
selected_images_dir = os.path.join(base_dir, "selected")
os.makedirs(selected_images_dir, exist_ok=True)

categories = ["COVID", "Normal", "Lung_Opacity", "Viral_Pneumonia"]  # Catégories
num_samples = 1000  # Nombre d'échantillons par catégorie

# Fixer les paramètres aléatoires des différentes bibliothèques
np.random.seed(42)
tf.random.set_seed(42)
random.seed(1234)

# Copier 1000 images aléatoires de chaque catégorie dans le dossier temporaire
for category in categories:
    src_folder = os.path.join(base_dir, category)
    dst_folder = os.path.join(selected_images_dir, category)
    os.makedirs(dst_folder, exist_ok=True)
    images = os.listdir(src_folder)
    selected = random.sample(images, min(num_samples, len(images)))
    for img in selected:
        src_path = os.path.join(src_folder, img)
        dst_path = os.path.join(dst_folder, img)
        if not os.path.exists(dst_path):
            with open(src_path, 'rb') as fsrc:
                with open(dst_path, 'wb') as fdst:
                    fdst.write(fsrc.read())

# Définir les paramètres des générateurs d'images
batch_size = 16
img_size = (224, 224)  # Taille d'entrée attendue par MobileNet

# Création du générateur avec prétraitement spécifique à MobileNet et validation split 20%
datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2) # Alternative : pipeline tf.data

# Générateur pour l'entraînement
train_generator = datagen.flow_from_directory(
    selected_images_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',  # Multi-class classification
    subset='training',
    shuffle=True,
    seed=42 # On fixe le germe aléatoire pour la reproductibilité
)

# Générateur pour la validation
val_generator = datagen.flow_from_directory(
    selected_images_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # On garde l'ordre pour l'évaluation
)

# Chargement du modèle MobileNet pré-entraîné sans la couche de classification finale
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Ajout des couches personnalisées pour la classification spécifique à notre problème
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Réduit la dimension sans perdre l'information spatiale
x = Dropout(0.5)(x)  # Dropout pour éviter l'overfitting
predictions = Dense(len(categories), activation='softmax')(x)  # Couche finale avec softmax

# Création du modèle final
model = Model(inputs=base_model.input, outputs=predictions)

# Geler les couches de MobileNet pour ne pas les réentraîner
for layer in base_model.layers:
    layer.trainable = False

# Compilation du modèle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle avec les générateurs
model.fit(train_generator, epochs=5, validation_data=val_generator)

# Évaluation sur le jeu de validation
val_generator.reset()
pred_probs = model.predict(val_generator)  # Probabilités prédites
preds = np.argmax(pred_probs, axis=1)  # Classes prédites

# Classes réelles
y_true = val_generator.classes

# Calcul des métriques
acc = accuracy_score(y_true, preds)
recall = recall_score(y_true, preds, average='macro')
# Affichage des résultats
print(f"Accuracy: {acc:.4f}")
print(f"Recall: {recall:.4f}")
print("Classification Report:")
print(classification_report(y_true, preds, target_names=val_generator.class_indices.keys(), labels=[0,1,2,3]))