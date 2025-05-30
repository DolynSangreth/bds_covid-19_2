import os
import glob
import pickle
import random
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
st.set_page_config(page_title="Reconnaissance de radios pulmonaires", layout="wide")

# ─── ENSURE SESSION-STATE KEYS EXIST ───
for key in ("loaded_model_name", "model", "preproc", "y_pred_probs"):
    if key not in st.session_state:
        st.session_state[key] = None

#Imports TensorFlow, sklearn, PIL, etc.
import tensorflow as tf
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from PIL import Image
try:
    import cv2
except ImportError:
    cv2 = None

# ─── SIDEBAR: MENU & MODEL LOADER ───
st.sidebar.image(
    "https://www.radiofrance.fr/s3/cruiser-production/2020/09/a8733967-cda8-48b5-9051-a17d218240be/1200x680_gettyimages-1213090148.jpg",
    use_container_width=True
)
st.sidebar.title("🧭 Présentation du projet")
# Définitions des pages
pages = [
    "Introduction", "Données", "Exploration & Visualisation",
    "Pré-traitement", "Modélisation & Résultats", "Conclusions & Perspectives"
]
selected_page = st.sidebar.radio("Navigation", pages)

# Sélecteur de modèle (inchangé)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
@st.cache_data
def get_model_files(base_dir: str):
    return sorted(
        glob.glob(os.path.join(base_dir, "*.h5")) +
        glob.glob(os.path.join(base_dir, "*.keras")) +
        glob.glob(os.path.join(base_dir, "*.pkl"))
    )
all_models = get_model_files(BASE_DIR)
model_choice = st.sidebar.selectbox(
    "Modèle local (.keras/.h5/.pkl)",
    [os.path.basename(p) for p in all_models]
)

# Initialisation de session
if "loaded_model_name" not in st.session_state:
    st.session_state.loaded_model_name = None
    st.session_state.model = None
    st.session_state.preproc = None

# Purge si sélection différente
if st.session_state.get("loaded_model_name") != model_choice:
    st.session_state.loaded_model_name = None
    st.session_state.model = None
    st.session_state.preproc = None
    st.session_state.y_pred_probs      = None

# Bouton de chargement
if st.sidebar.button("🗲 Charger le modèle"):
    path = os.path.join(BASE_DIR, model_choice)
    ext = os.path.splitext(path)[1].lower()
    with st.spinner(f"Chargement de {model_choice}…"):
        if ext in (".h5", ".keras"):
            mdl = keras_load_model(
                path,
                custom_objects={
                    "TFOpLambda": tf.keras.layers.Lambda,
                    "tf": tf
                },
                compile=False
            )
        elif ext == ".pkl":
            with open(path, "rb") as f:
                mdl = pickle.load(f)
        else:
            st.sidebar.error("Extension non supportée.")
            mdl = None
    if mdl:
        st.session_state.model = mdl
        st.session_state.loaded_model_name = model_choice
        low = model_choice.lower()

        if "mobilenet" in low:
            st.session_state.preproc = tf.keras.applications.mobilenet.preprocess_input
        elif "efficientnet" in low:
            st.session_state.preproc = tf.keras.applications.efficientnet.preprocess_input
        elif "vgg" in low:
            st.session_state.preproc = tf.keras.applications.vgg16.preprocess_input
        elif "resnet" in low:
            st.session_state.preproc = tf.keras.applications.resnet_v2.preprocess_input
        else:
            st.session_state.preproc = lambda x: x

        st.sidebar.success(f"{model_choice} chargé ✅")
        
# Options de cache & rafraîchissement
with st.sidebar.expander("⚙️ Options", expanded=False):
    if st.button("🧹 Vider cache & état"):
        try:
            st.cache_data.clear()
            st.cache_resource.clear()
        except:
            pass
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.success("Cache et état réinitialisés !")
    if st.button("🔄 Rafraîchir la liste des modèles"):
        get_model_files.clear()
        st.success("Liste des modèles rafraîchie ! Sélectionnez à nouveau.")

# Vérifier modèle chargé
if st.session_state.get("model") is None:
    st.warning("▶️ Sélectionnez un modèle puis cliquez sur “Charger le modèle” dans la sidebar.")
    st.stop()

# Récupération du modèle
model = st.session_state.model
preproc = st.session_state.preproc
model_path = os.path.join(BASE_DIR, model_choice)

# Dimensions d'entrée
inp_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape
_, h, w, _ = inp_shape
h, w = int(h), int(w)

# ─── PAGE CONTENT ───

if selected_page == "Introduction":
    st.title("🎯 Introduction au projet")

    # Contexte
    st.subheader("📖 Contexte")
    st.markdown(
        """
        Depuis le début de la pandémie de COVID-19, les services de radiologie ont été fortement saturés.  
        Les tests RT-PCR, bien que fiables, sont coûteux et présentent un taux non négligeable de faux négatifs selon le stade de l’infection.
        """
    )

    # Problématique & enjeux
    st.subheader("❓ Problématique & Enjeux")
    st.markdown(
        """
        - **Accélérer le tri** des radiographies pour les patients suspects  
        - **Alléger la charge** de travail des radiologues  
        - **Améliorer la fiabilité** et la réactivité du diagnostic  
        """
    )

    # Objectifs du projet
    st.subheader("🎯 Objectifs")
    st.markdown(
        """
        1. Développer un outil d’aide au diagnostic basé sur du **Deep Learning**.  
        2. **Classer** les radiographies pulmonaires selon 4 classes :  
           - COVID-19  
           - Lung Opacity  
           - Viral Pneumonia  
           - Normal  
        3. Fournir une **interface interactive** pour la visualisation et l’interprétation des résultats.  
        """
    )

    # CTA / Plan de la présentation
    st.info(
        """
        Dans les onglets suivants, vous découvrirez :  
        - 📊 **Données** : volumétrie, architecture et limitations  
        - 🔍 **Exploration & DataViz** : analyses visuelles et validations statistiques  
        - 🧠 **Modélisation** : premiers résultats et optimisations
        """
    )

elif selected_page == "Données":
    st.title("📁 Données utilisées")

    # Description textuelle enrichie avec citations
    st.markdown(
        """
        Le jeu de données utilisé ici provient d’une initiative de chercheurs de l’université du Qatar et de Dhaka, 
        en collaboration avec des docteurs malaisiens et pakistanais, 
        pour regrouper et annoter un important volume d’images de radiologie à rayons X pulmonaires.

        Les patients sont répartis en **4 catégories** distinctes :  
        - **Lung Opacity** : anomalies de type occlusions.  
        - **COVID** : patients infectés par le SARS-CoV-2 avec lésions visibles.  
        - **Viral Pneumonia** : infections virales pulmonaires.  
        - **Normal** : patients sains sans anomalies.  

        **Ce dataset (806.84 MB, .zip) est librement accessible sur [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database),**  
        ce qui nous a permis de l’exploiter sans contraintes réglementaires.  
        
        Note : l’accès à d’autres imageries médicales est souvent limité par le secret professionnel  
        et des bases non structurées en interne des CHU,  
        d’où le choix de ce jeu de données centralisé.  

        Chaque image est fournie avec un masque de filtre qui isole la région pulmonaire,  
        facilitant ainsi les étapes de nettoyage et de feature engineering.  

        <div style="color:red; font-weight:bold;">
        **Limites identifiées :**  
        <ul>
        <li>Disparité des sources : 6 origines différentes pour la classe COVID vs. 2 pour les autres.</li>
        <li>Déséquilibre des classes : la classe Normal contient 6× plus d’images que Viral Pneumonia, pouvant induire un sur-apprentissage des classes majoritaires.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Figure 1 : Sources des images
    st.subheader("Figure 1 : Origine des images par classe")
    st.image(
        "assets/Sources_Dataset.jpeg",
        caption="",
        width=700
    )

    # Figure 2 : Déséquilibre des classes
    st.subheader("Figure 2 : Déséquilibre des classes")
    st.image(
        "assets/Nombre_image_par_classes.jpeg",
        caption="",
        width=700
    )

elif selected_page == "Exploration & Visualisation":
    st.title("📊 Exploration & Visualisation")

    # Introduction courte
    st.markdown(
        """
        Nous avons extrait trois variables clés pour explorer les caractéristiques du dataset :
        - Luminosité moyenne par image et par classe  
        - Contraste par classe  
        - Aire du masque pulmonaire par classe  
        """
    )

    # Figure 3 : Luminosité moyenne
    st.subheader("Figure 3 : Luminosité moyenne par classe")
    st.image(
        "assets/Luminosite_par_classe.jpeg",
        caption="Pic à 125 pour Normal vs 150 pour COVID (ANOVA p < 0,05)",
        width=600
    )
    st.markdown(
        """
        **Constat**  
        - La classe *Normal* présente une luminosité moyenne d’environ **125**,  
        - La classe *COVID* atteint une luminosité moyenne d’environ **150**.  

        **Interprétation**  
        - Cette différence significative (ANOVA p < 0,05) reflète des conditions de prise de vue hétérogènes  
          ou des protocoles d’acquisition différents selon les centres.  
        """
    )

    # Figure 4 : Distribution du contraste
    st.subheader("Figure 4 : Distribution du contraste")
    st.image(
        "assets/Distribution_contraste_par_classe.jpeg",
        caption="Distribution plus large pour la classe COVID",
        width=600
    )
    st.markdown(
        """
        **Constat**  
        - La distribution du contraste est plus large pour la classe *COVID* que pour les autres classes.  

        **Interprétation**  
        - Cette variabilité accrue peut provenir du grand nombre de sources pour les images COVID,  
          introduisant des artefacts de contraste selon l’appareil ou le protocole.  
        """
    )

    # Figure 5 : Aire du masque pulmonaire
    st.subheader("Figure 5 : Aire du masque pulmonaire")
    st.image(
        "assets/Distribution_air_masques_classe.jpeg",
        caption="Variances différentes entre classes",
        width=600
    )
    st.markdown(
        """
        **Constat**  
        - Les moyennes d’aire des masques sont similaires entre classes,  
        - En revanche, les variances diffèrent notablement.  

        **Interprétation**  
        - Les différences de variance sont vraisemblablement dues au déséquilibre des effectifs  
          (classes majoritaires vs classes minoritaires) plutôt qu’à une réelle variation anatomique.  
        """
    )

    # Conclusion ciblée
    st.markdown(
        """
        <span style="color:red; font-weight:bold;"> Conclusion : 
        La qualité générale des données est satisfaisante et les annotations sont fiables.  
        Seul le rééquilibrage des classes est nécessaire avant la phase de modélisation.
        </span>
        """,
        unsafe_allow_html=True
    )

elif selected_page == "Pré-traitement":
    st.title("🧪 Pipeline de prétraitement")
    
    st.markdown(
        """
        Après avoir exploré plusieurs pistes, nous avons finalement opté pour un pipeline simple mais efficace, 
        axé sur la suppression des doublons et le rééquilibrage des classes :

        **1. Suppression des doublons**  
        - **Constat** : de nombreuses images quasi-identiques créaient de la redondance dans le jeu de données.  
        - **Méthode** : calcul de l’Indice de Similarité Structurelle (SSIM) sur versions 50×50,  
          puis hash conceptuel pour confirmer les doublons (seuil SSIM=0.8).  
        - **Illustration** : un exemple de paire détectée et supprimée."""  
        )
    
    st.image(
        "assets/Exemple_doublon_SSIM.jpeg",
        caption="Exemple de doublon détecté par SSIM et hash conceptuel",
        width=350
    )

    st.markdown(
        """
        **2. Augmentation ciblée des classes minoritaires**  
        - **Constat** : COVID et Viral Pneumonia restaient sous-représentées après nettoyage.  
        - **Méthode** : application de rotations (±15°), zooms (±10 %), flips horizontaux et variations de luminosité/contraste  
          **uniquement** sur les images de ces deux classes.  
        - **Bénéfice** : passage d’un ratio 1:6 à environ 1:1 entre classes majoritaires et minoritaires,  
          sans altérer la qualité clinique des images. """
    )

    st.image(
        "assets/Augmentation_image_COVID.jpeg",
        caption="Variations appliquées : rotation, zoom, flip, contraste",
        width=350
    )

    st.markdown(
        """
        **Méthodes testées puis abandonnées**  
        - **Filtrage par Laplacien** : éliminait des images utiles et n’améliorait pas les métriques.  
        - **Réduction de dimension (PCA & auto‐encodeur)** : introduisait un bruit et complexifiait le pipeline sans gain notable.  

        **Pipeline final retenu**  
        1. Détection & suppression des doublons (Figure 8)  
        2. Augmentation d’images ciblée pour COVID & Viral Pneumonia """
    )

    st.image(
        "assets/preprocessing.png",
        caption="Le Dataset sélectionné pour la modélisation est le V2V2",
        width=600
    )

    st.markdown(
        """
        <span style="color:red; font-weight:bold;">
        Ce choix garantit des données propres, équilibrées, et cliniquement valides pour la phase de modélisation.
        </span>
        """,
        unsafe_allow_html=True
    )

    

elif selected_page == "Modélisation & Résultats":
    st.title("🧠 Modélisation & Résultats")
    # Utiliser vos tabs d’évaluation et classification
    st.markdown("""
    Les modèles testés reposent sur du **transfert learning** à partir de réseaux pré-entraînés :
    - **VGG19**
    - **ResNet**
    - **MobileNet** 
    - **EfficientNet** 
                
    📊  Variation du poids d'entrainement de la classe COVID : 
    - **Le but** : augmenter les performences de prédiction pour cette classe
    - Pondération de la classe testé empiriquement 
    - Choix d'un gradiant de 1 à 3 """)

    st.image(
        "assets/image_model.png",
        caption="",
        width=600
    )
                  
    st.markdown(""" # plot 1 : tests empirique de la valeur du poids 
                
    ⚙️ **Paramètres d'entraînement** :
    - Split : 65 % train / 15 % val / 20 % test
    - Optimiseur : Adam
    - Perte : categorical cross-entropy
    - Fine-tuning : ouverture des dernières couches convolutives""")
    
           
    st.markdown("""
    # table 1 : metriques des différents modèles
    📈 **Métriques utilisées** :
    - Precision
    - Recall
    - F1-score (macro)
    - Matrice de confusion
    """)

    st.image(
        "assets/table_1.png",
        caption="",
        width=900
    )

    # Informations principales sur le modèle
    st.subheader("📋 Informations sur le modèle testé")

    # Architecture
    with st.expander("🔧 Afficher / masquer l'architecture du modèle", expanded=False):
        stream = []
        model.summary(print_fn=lambda x: stream.append(x))
        summary_text = "\n".join(stream)
        st.code(summary_text, language="")

    # Paramètres
    total = model.count_params()
    trainable = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable = total - trainable

    # Date de création du fichier
    ts = os.path.getmtime(model_path)
    dt = datetime.fromtimestamp(ts).strftime("%d/%m/%Y %H:%M")
    st.write(f"**Date fichier modèle :** {dt}")

    # Versions
    c4, c5 = st.columns(2)
    c4.write(f"**TensorFlow :** {tf.__version__}")
    c5.write(f"**Python :** {sys.version.split()[0]}")

    # Grad-CAM helper
    def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, tf.argmax(predictions[0])]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap

    # Caching load & predict
    fixed_class_names = ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"]
    DATASET_DIR = r"C:\Users\romai\OneDrive\Tiedostot\Projet COVID\DATASET_V2V2"
    TEST_DIR = os.path.join(DATASET_DIR, "test")

    @st.cache_resource
    def load_and_predict(TEST_DIR: str, h: int, w: int, class_names: list):
        preproc = st.session_state.preproc
        model   = st.session_state.model

        # Chargement des images et labels
        raw_ds = image_dataset_from_directory(
            TEST_DIR, labels="inferred", class_names=class_names,
            image_size=(h, w), batch_size=32, shuffle=False
        )
        raw_images = np.concatenate([x.numpy() for x, _ in raw_ds])

        ds = image_dataset_from_directory(
            TEST_DIR, labels="inferred", class_names=class_names,
            label_mode="categorical", image_size=(h, w),
            batch_size=32, shuffle=False
        )
        test_ds = ds.map(lambda x, y: (preproc(x), y), tf.data.AUTOTUNE)

        # Vérités terrain et prédictions
        y_true = np.argmax(np.concatenate([y for _, y in test_ds]), axis=1)
        y_pred_probs = model.predict(test_ds)
        y_pred = np.argmax(y_pred_probs, axis=1)

        return raw_images, y_true, y_pred, y_pred_probs

    tab1, tab2 = st.tabs(["📊 Évaluation dataset", "🖼️ Classification image"])
    with tab1:
        st.header("Évaluation sur le jeu de test")
        # bouton, metrics, matrice de confusion...
        if st.button("▶️ Lancer l'évaluation du dataset"):
            with st.spinner("Chargement & prédictions…"):
                raw_images, y_true, y_pred, y_pred_probs = load_and_predict(
                    TEST_DIR, h, w, fixed_class_names
                )
            # stockage
            st.session_state.raw_images = raw_images
            st.session_state.y_true     = y_true
            st.session_state.y_pred     = y_pred
            st.session_state.y_pred_probs    = y_pred_probs 

        if "y_true" in st.session_state:
            raw_images = st.session_state.raw_images
            y_true     = st.session_state.y_true
            y_pred     = st.session_state.y_pred
            y_pred_probs = st.session_state.y_pred_probs
            class_names = fixed_class_names

            # KPI
            prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
            rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
            f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
            m1, m2, m3 = st.columns(3)
            m1.metric("Précision", f"{prec:.1%}")
            m2.metric("Recall", f"{rec:.1%}")
            m3.metric("F1", f"{f1:.1%}")

            # Mapping
            st.subheader("📑 Mapping indices → classes")
            for idx, name in enumerate(class_names):
                st.write(f"- **{idx}** : {name}")

            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                annot_kws={"size": 10},  # taille des chiffres
                ax=ax
            )
            ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(class_names, rotation=0, fontsize=8)
            ax.set_xlabel("Prédit")
            ax.set_ylabel("Vrai")

            st.subheader("Matrice de confusion")

            col1, _ = st.columns([1, 3])
            with col1:
                st.pyplot(fig)
            
            # Classification report
            report = classification_report(
                y_true, y_pred, target_names=class_names, output_dict=True
            )
            df = pd.DataFrame(report).transpose()
            st.subheader("Rapport de classification")
            st.dataframe(df, use_container_width=True)

            # Grad-CAM exemples
            st.subheader("Exemples Grad-CAM par classe")
            for i, cls in enumerate(class_names):
                st.markdown(f"**Classe : {cls}**")
                cols = st.columns(4)
                corr  = np.where((y_true==i)&(y_pred==i))[0]
                wrong = np.where((y_true==i)&(y_pred!=i))[0]
                idx_c  = random.choice(corr)  if corr.size>0  else None
                idx_w  = random.choice(wrong) if wrong.size>0 else None

                def show(idx, title, col):
                    if idx is None:
                        cols[col].write(f"Aucun {title.lower()}.")
                        return
                    img = raw_images[idx].astype(np.uint8)
                    cols[col].image(img, caption=f"{title} • Original", width=200)
                    arr = np.expand_dims(img,0)
                    prep = preproc(arr)
                    last_conv = next((l.name for l in reversed(model.layers) if "conv" in l.name), None)
                    if last_conv:
                        heat = make_gradcam_heatmap(prep, model, last_conv)
                        hm = tf.image.resize(heat[...,tf.newaxis], (h,w)).numpy().squeeze()
                        if cv2:
                            hm_u = np.uint8(255*hm)
                            hm_c = cv2.applyColorMap(hm_u, cv2.COLORMAP_JET)
                            ov = cv2.addWeighted(img,0.6,hm_c,0.4,0)
                            cols[col+1].image(ov, caption=f"{title} • Grad-CAM", width=200)
                        else:
                            fig2, ax2 = plt.subplots()
                            ax2.imshow(hm, cmap="jet"); ax2.axis("off")
                            cols[col+1].pyplot(fig2)

                show(idx_c, "Bien classée", 0)
                show(idx_w, "Mal classée", 2)
            
            # ─── Histogrammes des probabilités par classe ───
            st.subheader("🔢 Distribution des probabilités par classe")

            # Préparation du DataFrame
            correct = (y_true == y_pred)
            df = pd.DataFrame({
                **{f"Prob_{cls}": y_pred_probs[:, idx]
                for idx, cls in enumerate(class_names)},
                "Predicted prob": np.max(y_pred_probs, axis=1),
                "True class": [class_names[i] for i in y_true],
                "Pred class": [class_names[i] for i in y_pred],
                "Correct": correct
            })

            # Création de la figure
            fig = plt.figure(figsize=(16, 12))
            sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})
            colors = sns.color_palette()

            plot_idx = 1
            for cls in class_names:
                for is_correct in [True, False]:
                    ax = fig.add_subplot(len(class_names), 2, plot_idx)
                    sub = df[(df["True class"] == cls) & (df["Correct"] == is_correct)]
                    title = f"{cls} — {'correctes' if is_correct else 'incorrectes'}"
                    # Histogramme de la probabilité sur la bonne classe
                    sns.histplot(
                        sub[f"Prob_{cls}"], bins=20,
                        color=colors[2] if is_correct else colors[3],
                        label=title, kde=True, ax=ax, alpha=0.6
                    )
                    # Pour les incorrectes, superposer les autres prédictions
                    if not is_correct:
                        for other in class_names:
                            if other == cls: continue
                            sel = sub[sub["Pred class"] == other]
                            sns.histplot(
                                sel["Predicted prob"], bins=20,
                                label=f"prédit {other}", kde=True,
                                ax=ax, alpha=0.6,
                                color=colors[class_names.index(other)]
                            )
                    ax.set_xlim(0, 1)
                    if is_correct:
                        ax.set_ylim(0, 1200)
                    else:
                        ax.set_ylim(0, 20)
                    ax.set_title(title)
                    ax.legend(fontsize="small")
                    plot_idx += 1

            plt.tight_layout()
            # Affichage dans Streamlit
            st.pyplot(fig)
    with tab2:
        st.header("Classification d'une image unique")
        # uploader, predict, gradcam
        img_file = st.file_uploader("Téléversez une image (.jpg/.png)", type=['jpg','jpeg','png'])
        if img_file:
            img = Image.open(img_file).convert('RGB')
            st.image(img, caption="Votre image", width=300)
            if st.button("▶️ Classer l'image"):
                x = img.resize((w,h))
                x_arr = np.expand_dims(np.array(x), 0)
                x_prep = preproc(x_arr)
                with st.spinner("Prédiction en cours…"):
                    preds = model.predict(x_prep)[0]
                idx = np.argmax(preds)

                st.subheader("Résultat")
                st.write(f"**{fixed_class_names[idx]}** ({preds[idx]:.2%})")
                st.write("Probabilités détaillées :")
                for name, p in zip(fixed_class_names, preds):
                    st.write(f"- {name} : {p:.2%}")

                last_conv = next((l.name for l in reversed(model.layers) if "conv" in l.name), None)
                if last_conv:
                    heatmap = make_gradcam_heatmap(x_prep, model, last_conv)
                    hm_r = tf.image.resize(heatmap[...,tf.newaxis], (h,w)).numpy().squeeze()
                    if cv2:
                        hm_u = np.uint8(255*hm_r)
                        hm_c = cv2.applyColorMap(hm_u, cv2.COLORMAP_JET)
                        rgb = np.array(x)
                        ov = cv2.addWeighted(rgb,0.6,hm_c,0.4,0)
                        st.image(ov, caption="Grad-CAM", width=300)
                    else:
                        fig3, ax3 = plt.subplots()
                        ax3.imshow(hm_r, cmap="jet"); ax3.axis("off")
                        st.pyplot(fig3)
                else:
                    st.warning("Impossible de trouver la couche conv pour Grad-CAM.")

elif selected_page == "Conclusions & Perspectives":
    st.title("🧩 Conclusions & Perspectives")
    st.markdown("""
    ✅ Ce projet a permis de :
    - Mettre en œuvre un pipeline complet d’analyse d’images médicales
    - Tester empiriquement diverses approches de prétraitement
    - Optimiser l'entraînement d’un modèle de classification efficace

    🧭 **Perspectives** :
    - Ajout de feedback utilisateur (Human-in-the-loop)
    - Entraînement adaptatif (reinforcement learning)
    - Intégration à un outil clinique (API, interface web, export HL7/DICOM)
    """)

    st.image(
        "assets/questions.jpeg",
        caption="",
        width=300
    )