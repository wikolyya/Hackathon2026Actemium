import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from stats import recup_csv

# ----------- POur ACP -----------
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def app_analyse_biv():
    # --- Pour éviter l'erreur Unicode sous Windows ---
    sys.stdout.reconfigure(encoding="utf-8")

    st.set_page_config(page_title="WADI Dashboard", layout="wide")

    # ----- Titre -----
    st.title("📊 Analyse bivariée et ACP de la dataset")
    st.subheader("🔗 Analyse univariée des variables numériques 🔗")

    # --- Chemin du CSV (relatif au script) ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "WADI_14days_new.csv")

    # --- Charger les données (caché pour ne pas recharger à chaque interaction) ---
    @st.cache_data
    def load_data():
        return recup_csv(csv_path)

    df = load_data()

    # ------ Selection des valeurs nuériques pour les statistiques descriptives -------
    num_cols = df.select_dtypes(include=[np.number])

    # ------ Matrice de corrélation -------
    st.subheader("Matrice de corrélation")
    corr_matrix = num_cols.corr(method="pearson")   

    # ---- Miseà plat de la matrice pour affichage ----
    cor_long = (
        corr_matrix.stack()
        .reset_index()   
    )

    cor_long.columns = ["Variable 1", "Variable 2", "Corrélation"]

    # On filre moitié supérieur et corrélation forte
    cor_long = cor_long[cor_long["Variable 1"] < cor_long["Variable 2"]]
    cor_long = cor_long[cor_long["Corrélation"].abs() > 0.95]
    cor_long = cor_long.reindex(
        cor_long["Corrélation"].sort_values(ascending=False).index
    )

    # ------ Affichage de la matrice de corrélation -----
    st.dataframe(cor_long)


    # -------------------------------------- ACP -------------------------------------- #
    st.subheader("🔗 Analyse en Composantes Principales (ACP) 🔗")

    # Selection des données numériques pour l'ACP
    num_data = df.select_dtypes(include=[np.number]).copy()
    
    
    # ----------- Suppression des colonnes avec une variance nulle (ou quasi nulle) pour éviter les problèmes dans l'ACP -----------
    num_data =  num_data.loc[:, num_data.std(skipna=True) > 0] 

    # ------------ Suppression des lignes avec des valeurs manquantes pour l'ACP (car PCA ne gère pas les NaN) -----------
    num_data = num_data.dropna(axis=0)

    st.write(f"Dimension apres nettoyage : {num_data.shape}")
    

    # ------------ Centrage et réduction des données -----------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(num_data)

    # ------------ Application de l'ACP -----------
    pca = PCA()

    # VAriance expliquée par chaque composante
    vars_explained = pca.fit(X_scaled).explained_variance_ratio_
    cum_vars = np.cumsum(vars_explained)

    st.write("Variance expliquée par chaque composante :")
    pca_df = pd.DataFrame(
        pd.DataFrame(
            {"Composante": [f"PC{i+1}" for i in range(len(vars_explained))], "Variance expliquée": vars_explained, "Variance cumulée": cum_vars}
        )
    )

    # ----------- Resume de l'ACP -----------
    st.subheader("Résumé de l'ACP")
    st.dataframe(pca_df.head(20))

    st.write(f"PC1 : {vars_explained[0]*100:.1f} %")
    st.write(f"PC2 : {vars_explained[1]*100:.1f} %")
    st.write(f"Cumul PC1–PC2 : {cum_vars[1]*100:.1f} %")
    st.write(f"Cumul PC1–PC4 : {cum_vars[3]*100:.1f} %")
    st.write(f"Cumul PC1–PC6 : {cum_vars[5]*100:.1f} %")
    st.write(f"Cumul PC1–PC10 : {cum_vars[9]*100:.1f} %")

    # ----------- Graphique de la variance expliquée -----------
    st.subheader("Variance cumulée expliquée 📈 ")

    threshold = st.slider(
        "Seuil de variance cumulée",
        0.5, 0.99, 0.8, 0.01
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        range(1, len(cum_vars) + 1),
        cum_vars,
        marker="o"
    )

    ax.axhline(
        y=threshold,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"{threshold*100:.0f} % de variance"
    )

    ax.set_xlabel("Composantes principales")
    ax.set_ylabel("Variance cumulée")
    ax.set_title("Variance cumulée expliquée par l'ACP")
    ax.set_ylim(0, 1)
    ax.legend()

    st.pyplot(fig)

    # Nombre de composantes nécessaires
    n_comp = np.argmax(cum_vars >= threshold) + 1
    st.info(f"{n_comp} composantes nécessaires pour atteindre {threshold*100:.0f} % de variance")
            






