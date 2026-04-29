# Tests des modeles WADI

Ce dossier compare les modeles sur le probleme demande dans le guide projet:
prevoir `1_LT_001_PV` a horizon 10 minutes et mesurer `MAE`, `RMSE` et
`skill_vs_persistence`.

## Test rapide

Depuis la racine du projet:

```bash
source .venv/bin/activate
PYTHONPATH=src python -m hackathon_actemium.models_tests.main_compare \
  --nrows 15000 \
  --skip-deep
```

Ce test utilise l'horizon par defaut `600` pas, soit 10 minutes si les donnees
sont a 1 Hz. Il ecrit les resultats dans `outputs_compare/`.

## Sorties produites

- `outputs_compare/model_comparison.csv`: tableau final trie par RMSE.
- `outputs_compare/summary.json`: memes metriques en JSON.
- `outputs_compare/predictions_tabular.csv`: verite terrain, baseline,
  ARIMA, auto-ARIMA si active, XGBoost et modele local lineaire.
- `outputs_compare/predictions_kalman.csv`: verite terrain, baseline et Kalman.
- `outputs_compare/predictions_dl.csv`: cree seulement si les modeles deep
  learning sont lances.
- `outputs_compare/plots/`: graphiques PNG pour le rapport et le diaporama.

## Lancement complet

```bash
source .venv/bin/activate
PYTHONPATH=src python -m hackathon_actemium.models_tests.main_compare
```

Cette commande lance tous les modeles:

- baseline persistence
- ARIMA statsmodels
- auto-ARIMA pmdarima si active
- XGBoost
- modele local lineaire
- GRU
- LSTM
- TCN
- Transformer temporel
- Kalman

Les modeles deep learning peuvent prendre longtemps. Pour entrainer tous les
modeles sur un echantillon plus petit:

```bash
PYTHONPATH=src python -m hackathon_actemium.models_tests.main_compare \
  --nrows 20000 \
  --epochs 2 \
  --batch-size 128
```

Pour une comparaison rapide et reproductible pendant le developpement, garder
`--skip-deep`.

## Changer la variable cible

La cible par defaut est `1_LT_001_PV`, comme dans le guide projet. Pour predire
une autre variable, utiliser `--target`.

Exemple avec `3_LT_001_PV`:

```bash
PYTHONPATH=src python -m hackathon_actemium.models_tests.main_compare \
  --target 3_LT_001_PV \
  --horizon 600 \
  --skip-deep
```

Pour afficher toutes les colonnes disponibles dans le CSV:

```bash
python -c "import pandas as pd; df=pd.read_csv('src/hackathon_actemium/stats/WADI_14days_new.csv', nrows=1); print('\n'.join(df.columns))"
```

`--horizon 600` correspond a une prediction a +10 minutes si la frequence est
bien de 1 seconde. Utiliser `--horizon 300` pour 5 minutes et `--horizon 1800`
pour 30 minutes.

## Options utiles

- `--target 1_LT_001_PV`: cible conseillee par le guide.
- `--horizon 600`: prediction a +10 minutes.
- `--horizon 300`: prediction a +5 minutes.
- `--horizon 1800`: prediction a +30 minutes.
- `--nrows 20000`: limite les donnees pour tester vite.
- `--save-models`: sauvegarde les modeles tabulaires dans
  `outputs_compare/saved_models/`.

## Baselines ARIMA

Le script supporte maintenant un vrai modele ARIMA avec `statsmodels`:

```bash
PYTHONPATH=src python -m hackathon_actemium.models_tests.main_compare \
  --target 3_LT_001_PV \
  --horizon 600 \
  --nrows 15000 \
  --skip-deep
```

Par defaut, l'ordre ARIMA est `(5, 1, 0)`. Pour le changer:

```bash
PYTHONPATH=src python -m hackathon_actemium.models_tests.main_compare \
  --target 3_LT_001_PV \
  --horizon 600 \
  --arima-order 3 1 2 \
  --skip-deep
```

Pour lancer aussi `pmdarima.auto_arima`:

```bash
PYTHONPATH=src python -m hackathon_actemium.models_tests.main_compare \
  --target 3_LT_001_PV \
  --horizon 600 \
  --nrows 15000 \
  --include-auto-arima \
  --skip-deep
```

Attention: `auto_arima` peut etre lent sur le dataset complet. Le tester d'abord
avec `--nrows 15000` ou `--nrows 20000`.

## Commandes de test ARIMA / auto-ARIMA

Depuis la racine du projet:

```bash
cd "/Users/miroslavpenkov/Documents/MASTER IA 2025-2027/MASTER IA/S2/Projet IA/Hackathon2026Actemium"
source .venv/bin/activate
```

Test rapide avec ARIMA, sans deep learning:

```bash
PYTHONPATH=src python -m hackathon_actemium.models_tests.main_compare \
  --target 3_LT_001_PV \
  --horizon 600 \
  --nrows 15000 \
  --skip-deep
```

Test avec ARIMA + auto-ARIMA:

```bash
PYTHONPATH=src python -m hackathon_actemium.models_tests.main_compare \
  --target 3_LT_001_PV \
  --horizon 600 \
  --nrows 15000 \
  --include-auto-arima \
  --skip-deep
```

Test ultra-rapide de debug:

```bash
PYTHONPATH=src python -m hackathon_actemium.models_tests.main_compare \
  --target 3_LT_001_PV \
  --horizon 60 \
  --nrows 3000 \
  --include-auto-arima \
  --skip-deep \
  --skip-kalman \
  --arima-max-train-samples 1000
```

Lire le tableau final:

```bash
cat outputs_compare/model_comparison.csv
```

Les graphiques sont generes automatiquement dans:

```bash
outputs_compare/plots/
```

Pour les lister:

```bash
ls outputs_compare/plots
```

Options utiles:

- `--skip-arima`: ignore le modele ARIMA statsmodels.
- `--include-auto-arima`: ajoute auto-ARIMA.
- `--arima-order 5 1 0`: change l'ordre `(p, d, q)`.
- `--arima-max-train-samples 30000`: limite les points utilises par ARIMA.
- `--arima-max-train-samples 0`: utilise tout le train.
- `--arima-mode dynamic`: forecast rapide depuis le train.
- `--arima-mode rolling`: mise a jour avec observations, plus juste mais tres lent.

## Graphiques generes

A la fin de chaque run, le script cree automatiquement des graphiques dans
`outputs_compare/plots/`:

- `metrics_rmse_mae.png`: comparaison MAE/RMSE par modele.
- `skill_vs_persistence.png`: gain ou perte de chaque modele vs persistence.
- `modeles_tabular_predictions_vs_true.png`: courbes prediction vs verite terrain.
- `modeles_tabular_residuals.png`: erreurs residuelles dans le temps + histogramme.
- `modeles_tabular_predicted_vs_actual.png`: scatter reel vs predit.
- `modeles_tabular_top_errors.png`: plus grosses erreurs absolues.

Si Kalman ou les modeles deep learning sont lances, des graphiques equivalents
sont aussi produits pour `predictions_kalman.csv` et `predictions_dl.csv`.

Options utiles:

- `--skip-plots`: ne genere pas les PNG.
- `--plot-samples 3000`: change le nombre de points affiches dans les courbes.
