MODEL_TYPES = [
    "arima",
    "sarimax",
    "stl_arima",
    "stl_sarimax"
]

DEFAULT_MODEL = "arima"

SEASONALITY = 12   # fixe comme tu veux
S_WINDOW = 7       # lissage STL (optionnel)

TYPE_PRODUIT = False
VERBOSE = 0