import pandas as pd

path = r"WADI_14days_new.csv"

## ---------------------- FONCTIONS DE TRAITEMENT DE FICHIERS ------------------###
def recup_csv(path):
    """
    Docstring for recup_csv

    Fonction servant à récupérer les données d'un fichier csv
    
    :param path: chemin d'accès à notre fichier csv
    """
    df = pd.read_csv(path, low_memory=False, encoding="utf-8") # Permet de charger le fichier CSV en ignorant les avertissements liés à la mémoire
    return df

def entete_csv(path, n=5):
    """
    Docstring for entete_csv

    Fonction servant à récupérer les 5 premières lignes d'un fichier csv
    et à afficher les noms des colonnes et le nombre de colonnes
    
    :param path: chemin d'accès à notre fichier csv
    """
    df = pd.read_csv(path, low_memory=False)
    
    # Noms des colonnes 
    print("En-tête du fichier CSV :")
    print(list(df.columns))
    print(f"Nombre de colonnes : {len(df.columns)}")
    
    # Affichage des 5 premières lignes
    print(df.head(n)) # Affiche les n premières lignes du DataFrame


def count_ligns(path):
    # Nombre de lignes
    df = pd.read_csv(path, low_memory=False)
    num_rows = len(df)
    print(f"Nombre de lignes : {num_rows}")





