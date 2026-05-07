# Hackathon2026Actemium
Projet conjoint **M1/M2 MAS × M1 IA** réalisé dans le cadre d'un hackathon universitaire,
en partenariat avec Actemium.

### Structure du projet

```
notebooks/
├── EDA_WADI.ipynb
├── baselines.ipynb
└── WADI_XGBoost.ipynb
output-compare/
├── plots/
└── summary.json
src/hackathon_actemium/
├── dl_model/
├── ml_model/
└── models_tests/
README.md
Rapport_WADI.pdf
requirements.txt
```

---

## Pipeline de développement

1. **Exploration des données**
   - Analyse univariée, bivariée, ACP
   - Visualisation des distributions et corrélations

2. **Nettoyage des données**
   - Remplacement des valeurs manquantes
   - Conversion des types
   - Suppression des doublons et valeurs aberrantes

3. **Modélisation**
   - Baseline naïve (persistence, moyenne)
   - Comparaison de modèles :
     - `XGBoost` — performant sur grands datasets tabulaires
     - `Random Forest` — robuste sur petits volumes
     - `LSTM` — adapté aux séquences temporelles
     - `GRU` — alternative légère au LSTM
     - `TCN` — convolutions causales pour séries temporelles
     - `Temporal Transformer` — efficace avec variables exogènes

4. **Optimisation**
   - Tuning des hyperparamètres via **Optuna** (recherche bayésienne)
   - Early stopping pour limiter l'overfitting

---

## Ressources

| Sujet | Lien |
|---|---|
| Optuna + XGBoost | [optuna-examples](https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_simple.py) |
| Quand utiliser Random Forest | [Minitab](https://www.minitab.com/fr-fr/solutions/analytics/statistical-analysis-predictive-analytics/random-forests/) |
| Pourquoi XGBoost | [ScienceDirect](https://www.sciencedirect.com/article/pii/S1877050925026092) |
| Top 5 modèles prédictifs | [InsightSoftware](https://insightsoftware.com/fr/blog/top-5-predictive-analytics-models-and-algorithms/) |
| Comparaison GRU / LSTM / TCN / Transformer | [Preprints](https://www.preprints.org/manuscript/202601.1962) |
| Temporal Transformer (Informer) | [arXiv](https://arxiv.org/abs/1912.09363) |

---
