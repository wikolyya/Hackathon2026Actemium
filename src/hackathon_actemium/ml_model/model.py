from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from config_model import FIRST_SPLIT, FINAL_SPLIT, RANDOM_STATE
from path import read_csv

plt.style.use("ggplot")
import xgboost as xgb

# Redirection vers le bon directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "WADI_14days_new.csv")

# Importation dataset
data = read_csv(csv_path)

# A COMPLETER/MODIFIER pour adater à la target
def define_target(df, TARGET):
    """_summary_

    Args:
        TARGET (_type_): _description_
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    return X, y

# Séparation en données d'entrainements, de test et de validation
def separate(X, y, val):
    """
    Fonction permettant de séparer les données en ensemble
    d'apprentissage et de test (ou de test et de validation)
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, 
        y,
        test_size= val,
        random_state=RANDOM_STATE
    )
    
    return X_train, X_temp, y_train, y_temp
    
def train_data():
    """
    Fonction permettant d'entrainer le modèle
    """
    return print("En cours.")


