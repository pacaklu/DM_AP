"""Config file for lgbm model."""

params = {
    "num_boost_round": 100000,
    "early_stopping_rounds": 200,
    "learning_rate": 0.01,
    "metric": "auc",
    "objective": "binary",
    "seed": 1234,
    "verbose": 1,
}
