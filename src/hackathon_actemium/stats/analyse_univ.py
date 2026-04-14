import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import streamlit as st
from path.path import recup_csv
import os
import sys


def app_analyse():
    # --- Pour éviter l'erreur Unicode sous Windows ---
    #sys.stdout.reconfigure(encoding="utf-8") #pas besoin de cette ligne on le fait dans pd.readCSV

    st.set_page_config(page_title="WADI Dashboard", layout="wide")

    # --- Titre ---
    st.title("📊 Exploration des données du dataset WADI")

    # --- Chemin du CSV (relatif au script) ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "WADI_14days_new.csv")

    # ---- Charger les données (caché pour ne pas recharger à chaque interaction) ---
    @st.cache_data
    def load_data():
        return recup_csv(csv_path)
    df = load_data()

    # ------- Selection des valeurs nuériques pour les statistiques descriptives -------
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    # -------- Création d'une selection des colonnes ------
    selected_cols = st.multiselect(
        "Sélectionnez les colonnes numériques à analyser :",
        options=num_cols,
        default=num_cols[:5]  # Par défaut, on sélectionne les 5 premières colonnes numériques
    )

    # ------- Affichage des statistiques descriptives -------
    for col in selected_cols:
        st.subheader(f"Statistiques descriptives pour la colonne : {col}")
        st.write(df[col].describe())
        st.write(f"Nombre de valeurs manquantes : {df[col].isna().sum()}")

        # --- Visualisation de la distribution ---
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(x=df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Distribution de la colonne {col}")
        st.pyplot(fig)
        st.markdown("---")  # Séparateur entre les sections