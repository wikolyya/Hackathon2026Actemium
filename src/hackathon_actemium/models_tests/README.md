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
  XGBoost et modele local lineaire.
- `outputs_compare/predictions_kalman.csv`: verite terrain, baseline et Kalman.
- `outputs_compare/predictions_dl.csv`: cree seulement si les modeles deep
  learning sont lances.

## Lancement complet

```bash
source .venv/bin/activate
PYTHONPATH=src python -m hackathon_actemium.models_tests.main_compare
```

Cette commande lance tous les modeles:

- baseline persistence
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
