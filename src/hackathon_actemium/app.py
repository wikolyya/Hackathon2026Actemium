import streamlit as st
import stats.stats
from stats import analyse_univ 
from stats import analyse_bivariee

st.set_page_config(page_title="Dashboard WADI", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à :", ["Accueil", "Stats", "Analyse Univariée", "Analyse bivariée et ACP", "XGBOOST"])

if page == "Accueil":
    st.title("🏠 Bienvenue sur le Dashboard WADI")
    st.subheader("Projet réalisé dans le cadre du Hackathon Actemium")
    st.write("""
    Bienvenue sur notre projet réalisé conjoitement par les M1 et M2 MAS, et les M1 IA.
             Noms des membres du projet : 
    - M1 MAS : Stowe, Karen
    - M2 MAS : Merlin, 
    - M1 IA : Miroslav, Victoria
             

    Utilisez le menu à gauche pour naviguer :
    - **Stats** : statistiques descriptives du dataset
    - **Analyse** : visualisation interactive
    """)
elif page == "Stats":
    stats.stats.app_stats()  # fonction que l’on définira dans stats.py
elif page == "Analyse Univariée":
    analyse_univ.app_analyse()  
elif page == "Analyse bivariée et ACP":
    analyse_bivariee.app_analyse_biv()  
elif page == "XGBOOST":
    st.title("XGBOOST")
    st.write("Contenu à venir...")