# Hackathon2026Actemium
Projet conjoint entre M1/M2 MAS, M1 IA pour hackathon proposé dans le cadre universitaire

### Etapes clefs du développement du projet
1. Exploration des données en liste de dictionnaires
2. Nettoyage des données :
   - Remplacer les valeurs manquantes par None
   - Conversion des types de données appropriés
   - Supprimer si besoin valeurs aberrantes, doublons, anomlies
  
Ces deux étapes se font par itérations : Schéma optimal :
         - Données brutes
         - Nettoyage minimal
         - Exploration (u-ni / bi / ACP)
         - Hypothèses métier
         - Nettoyage informé
         -  Modélisation

3. Selection des variables utiles (donc LT notamment)
4. Baseline : se baser sur le naive bayes/autre méthode à discuter pour avoir des valeurs de bases demander des clarifications
5. Test de modèles 

  - RANDOM FOREST (avantage quand on a peu d'enregistrement)
  - XGBoost (plus avantageux que random forest sur des gros datasets)
  - LSTM (est-ce que facteur constant des flux d'informations)?
  - Transformer temporel #peut être pour booster le modèle vu le nomre de données


Lien utile:

Quand utiliser Random Forest:
https://www.minitab.com/fr-fr/solutions/analytics/statistical-analysis-predictive-analytics/random-forests/#:~:text=Random Forests est le seul,chaque enregistrement a son importance.


Pourquoi  utiliser GXBoost:
https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://translate.google.com/translate?u=https://www.sciencedirect.com/science/article/pii/S1877050925026092&hl=fr&sl=en&tl=fr&client=rq#:~:text=XGBoost%20consistently%20performed%20better%20achieving,particularly%20when%20using%20Pearson%20Correlation.&ved=2ahUKEwjC5ZnAw66SAxWnbKQEHYO3Ob0QFnoECC4QAw&usg=AOvVaw3mgjFK3RQXEXlpTPR9sOAK

Top 5 model d’analyse:
https://insightsoftware.com/fr/blog/top-5-predictive-analytics-models-and-algorithms/
