import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Dossiers des fichiers de base
COVID_DIR = r"C:\Users\ambro\Desktop\Projet Metier\dataset_original_images\COVID"
NORMAL_DIR = r"C:\Users\ambro\Desktop\Projet Metier\dataset_original_images\Normal"
LUNG_OP_DIR = r"C:\Users\ambro\Desktop\Projet Metier\dataset_original_images\Lung_Opacity"
VIRAL_PNEUM_DIR = r"C:\Users\ambro\Desktop\Projet Metier\dataset_original_images\Viral_Pneumonia"
BLUR_THRESHOLD = 30 # Le seuil de flou

# Dossier de destination pour les images "nettes"
DESTINATION_DIR = r"C:\Users\ambro\Desktop\Projet Metier\dataset_laplacien_30"

# Extensions d'image supportées
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

# --- Fonction pour calculer la variance du Laplacien (modifiée) ---
def calculate_blur_variance_and_load(image_path):
    """
    Charge une image, la convertit en niveaux de gris, applique le Laplacien,
    et retourne la variance du résultat ainsi que l'image originale chargée.
    Retourne (None, None) en cas d'erreur.
    """
    try:
        img = cv2.imread(image_path)

        # Vérifier si l'image a été chargée correctement
        if img is None:
            print(f"Erreur : Impossible de charger l'image {image_path}")
            return None, None

        # Convertir en niveaux de gris (au cas où l'image serait encodée malgré tout en BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Appliquer l'opérateur Laplacien
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Calculer la variance
        variance = np.var(laplacian)

        return variance, img # Retourne la variance et l'image chargée

    except Exception as e:
        print(f"Erreur lors du traitement de l'image {image_path}: {e}")
        return None, None # En cas d'erreur

# Créé un DataFrame pour accueillir les variances
# Post-Laplacien
df_var_laplacien = pd.DataFrame([])
# Pré-Laplacien
df_var_originale = pd.DataFrame([])

# --- Traitement des dossiers ---
if __name__ == "__main__":
    # Liste des dossiers sources à traiter
    directories_to_process = [COVID_DIR, NORMAL_DIR, LUNG_OP_DIR, VIRAL_PNEUM_DIR]

    # --- Créer le dossier de destination s'il n'existe pas ---
    # exist_ok=True permet d'éviter une erreur au cas où le dossier existe
    os.makedirs(DESTINATION_DIR, exist_ok=True)
    print(f"Dossier de destination pour les images nettes : {DESTINATION_DIR}")
    print("-" * 50)

    print(f"Début du traitement des images avec un seuil de flou de {BLUR_THRESHOLD}...")
    print(f"Les images avec une variance >= {BLUR_THRESHOLD} seront considérées comme nettes et sauvegardées.")
    print(f"Les images avec une variance < {BLUR_THRESHOLD} seront considérées comme floues et ignorées.")
    print("-" * 50)


    # Parcourir chaque dossier source
    for current_dir in directories_to_process: # Répète la boucle pour tous les dossiers d'images
        print(f"\n--- Traitement du dossier source : {current_dir} ---")

        # Extraire le nom du sous-dossier pour la destination
        subdir_name = os.path.basename(current_dir) # Récupère le nom du dossier d'images
        destination_subdir = os.path.join(DESTINATION_DIR, subdir_name) # Lie le dossier d'images au dossier de destination

        # Créer le sous-dossier de destination (COVID, Normal, etc.)
        os.makedirs(destination_subdir, exist_ok=True)
        print(f"  Dossier de destination pour ce type d'image : {destination_subdir}")


        # Lister tous les fichiers dans le dossier source
        all_files = os.listdir(current_dir)

        saved_images_count = 0
        ignored_images_count = 0
        processed_images_count = 0

        # Créé une liste par sous-dossier pour stocker les variances

        # Post-Laplacien
        l = pd.Series([])

        # Pré-Laplacien
        l_original = pd.Series([])
        # Créer un compteur pour le pré-Laplacien
        n_origine = 0
        # Parcourir chaque fichier
        for filename in all_files:
            file_path = os.path.join(current_dir, filename)

            # Vérifier si c'est un fichier et s'il a une extension d'image supportée
            if os.path.isfile(file_path):
                file_extension = os.path.splitext(filename)[1].lower()
                if file_extension in IMAGE_EXTENSIONS:
                    processed_images_count += 1
                    # Calculer la variance de flou et charger l'image
                    variance, img = calculate_blur_variance_and_load(file_path)
                    # Ajouter la variance du fichier, quelle qu'elle soit
                    l_original[n_origine] = variance
                    n_origine += 1
                    # Si le calcul a réussi (variance n'est pas None)
                    if variance is not None:
                        # Comparer la variance au seuil
                        if variance >= BLUR_THRESHOLD:
                            # --- L'image est considérée comme nette : on la sauvegarde ---
                            destination_file_path = os.path.join(destination_subdir, filename)
                            cv2.imwrite(destination_file_path, img)
                            print(f"  [SAUVÉ] - Variance : {variance:.2f} - {filename}")
                            saved_images_count += 1

                            # On ajoute la variance du laplacien au df du dossier
                            l[processed_images_count] = variance
                        else:
                            ignored_images_count += 1
                   
        print(f"\nRésumé pour {subdir_name}:")
        print(f"  Images traitées : {processed_images_count}")
        print(f"  Images sauvegardées (nettes) : {saved_images_count}")
        print(f"  Images ignorées (floues ou erreur) : {ignored_images_count}")

        df_var_laplacien[current_dir[66:]] = l
        df_var_originale[current_dir[66:]] = l_original
    print("\n" + "=" * 50)
    print("--- Traitement global terminé ---")
    print(f"Les images considérées comme nettes (variance >= {BLUR_THRESHOLD:.2f}) ont été sauvegardées dans {DESTINATION_DIR}.")
    print(f"Note : Le seuil de {BLUR_THRESHOLD} est très bas pour la variance du Laplacien. "
          "Tu pourrais avoir besoin d'ajuster ce seuil (souvent dans les centaines ou milliers) "
          "en fonction de la résolution et du contenu de tes images pour obtenir de meilleurs résultats.")
    print("=" * 50)

df = df_var_laplacien.iloc[:1000,:].dropna()
print(df_var_laplacien.head(10))

# Statistiques groupées par dossier
stats = df_var_laplacien.agg(['mean', 'median', 'std', 'min', 'max']).reset_index()
stats.to_excel("Statistiques_t30.xlsx")
print("Statistiques sur les variances du Laplacien :\n")
print(stats)




# Visualisation
print("Aperçu du DataFrame :")
print(df.head())
print("\n" + "="*50 + "\n")

# On exclut les colonnes non num (au cas où)
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

if not numerical_cols:
    print("Aucune colonne numérique trouvée dans le DataFrame pour tracer les densités.")
else:
    print(f"Colonnes numériques trouvées : {numerical_cols}")

    # --- Déterminer la taille de la grille pour les sous-graphiques ---
    n_cols = 3 # Nombre de colonnes souhaité
    n_rows = math.ceil(len(numerical_cols) / n_cols) # Nombre de lignes nécessaire

    print(f"Création d'une grille de {n_rows}x{n_cols} pour les graphiques.")

    # --- Créer la figure et les sous-graphiques ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4)) # Ajuster la taille de la figure

    # Aplatir le tableau d'axes si plus d'une ligne/colonne pour faciliter l'itération
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]

    # --- Boucler sur chaque colonne numérique et tracer la courbe de densité ---
    for i, col in enumerate(numerical_cols):
        ax = axes[i] # Sélectionner le sous-graphique courant
        sns.kdeplot(data=df, x=col, fill=True, ax=ax)
        ax.set_title(f'Distribution de la colonne : {col}') 
        ax.set_xlabel(col)
        ax.set_ylabel('Densité')

    # --- Masquer les sous-graphiques inutilisés s'il y en a ---
    for j in range(len(numerical_cols), len(axes)):
        fig.delaxes(axes[j]) # Supprimer les axes en trop

    # --- Ajuster la mise en page et afficher les graphiques ---
    plt.tight_layout() # Ajuste automatiquement les paramètres de sous-graphique pour éviter le chevauchement
    plt.show()