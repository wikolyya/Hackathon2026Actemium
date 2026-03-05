from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os 
from config_model import FIRST_SPLIT, FINAL_SPLIT, RANDOM_STATE
from path import read_csv

plt.style.use("ggplot")
import xgboost as xgb
import optuna 

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
    
## Idée avec XGBOOST : Optimisation des hyperparamètres : 
# 1. RandomizedSearch, tri aléatoire des hyperparamètres
# 2. Optuna
# 3. Early Stopping, permet d'éviter l'overfitting et d'accélrer la recherche

# Definition de la fonction avec objective trial
def objective(trial):
    """
    Fonction Optuna

    Args:
        trial (_type_): _description_
    """
    params={ 
        # Objectif du modèle (classification binaire)
        "objective": "binary:logistic",
        
        # Métrique d'eval de XGBOOST
        "eval_metric": "logloss",
        
        # Méthode d'apprentissage des arbres
        "tree_method": "hist",
        
        # ---- Hyperparam de régularisation ---- #
        
        #L2 regularization pour éviter l'overfitting
       "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
       
       #L1 regularization
       "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
       
       # ----- Paramètres des arbres ----------- #
       "max_depth": trial.suggest_int("max_depth",  3, 10),
       
       # Learning rate
       "eta": trial.suiggest_float("eta", 0.01, 0.3),
       
       # ------- Echantillonnage ---------- # 
       "subsample": trial.suggest_float("Subsample", 0.2, 1.0),
       
       # Fraction features utilisées par un arbre
       "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
       
       # --------- Paramètres des arbres ----------- # 
       "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
       
       "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True)
       
       #TODO : Faire l'entrainement du modèle
    }
    
    
    
    