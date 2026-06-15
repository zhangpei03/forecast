from src.services.autogluon_service import _lightweight_hyperparameters


def test_lightweight_hyperparameters_excludes_heavy_deep_models() -> None:
    hyperparameters = _lightweight_hyperparameters()

    assert set(hyperparameters) == {
        "SeasonalNaive",
        "RecursiveTabular",
        "DirectTabular",
        "ETS",
        "Theta",
    }
    assert "Chronos2" not in hyperparameters
    assert "TemporalFusionTransformer" not in hyperparameters
