import streamlit as st
from path.path import recup_csv
from skimpy import skim
import os
import sys

def app_stats():
    # --- Pour éviter l'erreur Unicode sous Windows ---
    sys.stdout.reconfigure(encoding="utf-8")

    st.set_page_config(page_title="WADI Dashboard", layout="wide")

    # --- Titre ---
    st.title("📊 Statistiques descriptives du dataset WADI")

    # --- Chemin du CSV (relatif au script) ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "WADI_14days_new.csv")

    # --- Charger les données (caché pour ne pas recharger à chaque interaction) ---
    @st.cache_data
    def load_data():
        return recup_csv(csv_path)

    df = load_data()

    # --- Affichage simple ---
    st.subheader("Aperçu des données (100 premières lignes)")
    st.dataframe(df.head(100))

    st.subheader("Dimensions du dataset")
    st.write(f"Lignes : {df.shape[0]}")
    st.write(f"Colonnes : {df.shape[1]}")

    # Statistiques
    if st.checkbox("Afficher statistiques descriptives"):
        st.dataframe(df.describe().T)  # stats numériques
        st.dataframe(df.isna().sum().to_frame(name="NAs"))  # valeurs manquantes