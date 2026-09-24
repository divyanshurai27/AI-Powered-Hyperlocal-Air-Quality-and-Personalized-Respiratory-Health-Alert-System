import json

import numpy as np
import pandas as pd
import pytest

from app.domain.pollutants import Pollutant
from app.ml.features import (
    HORIZONS,
    LAGS_HOURS,
    build_forecasting_matrix,
    build_station_frames,
    build_targets,
    feature_lookback_hours,
    feature_names,
    generate_features,
)
from app.ml.forecasting import (
    ArtifactContractError,
    DirectGBMForecaster,
    ForecastArtifactMetadata,
    GBMConfig,
    PersistenceForecaster,
    SeasonalNaiveForecaster,
    compare_forecast_models,
    dataset_fingerprint,
    load_model_artifact,
    save_model_artifact,
    validate_forecast,
)

START = pd.Timestamp("2026-01-01T00:30Z")
P = Pollutant.PM25


def synthetic_obs(hours: int = 24 * 20, stations=("s1", "s2"), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, periods=hours, freq="h", tz="UTC")
    rows = []
    for k, sid in enumerate(stations):
        daily = 40 + 15 * np.sin(2 * np.pi * (idx.hour + 5.5) / 24) + k * 5
        for pol, scale in (("pm25", 1.0), ("pm10", 1.8), ("no2", 0.6)):
            vals = daily * scale + rng.normal(0, 2, hours)
            rows += [
                {
                    "station_id": sid,
                    "pollutant": pol,
                    "timestamp": t,
                    "concentration": max(v, 0),
                    "quality_flag": "valid",
                }
                for t, v in zip(idx, vals, strict=True)
            ]
    return pd.DataFrame(rows)


def synthetic_weather(hours: int = 24 * 21) -> pd.DataFrame:
    idx = pd.date_range(
        START.floor("h") - pd.Timedelta(hours=24), periods=hours, freq="h", tz="UTC"
    )
    return pd.DataFrame(
        {
            "temperature": 20 + 5 * np.sin(np.arange(hours) / 24 * 2 * np.pi),
            "relative_humidity": 70.0,
            "precipitation": 0.0,
            "wind_speed": 2.0,
            "wind_direction": 180.0,
            "pressure": 912.0,
        },
        index=idx,
    )


COORDS = {"s1": (12.91, 77.59), "s2": (13.03, 77.58)}


@pytest.fixture(scope="module")
def frames():
    return build_station_frames(synthetic_obs())


# --- leakage -----------------------------------------------------------------------------


def test_declared_lookbacks_never_reach_into_the_future() -> None:
    for pollutant in (Pollutant.PM25, Pollutant.NO2):
        spec = feature_lookback_hours(pollutant)
        assert set(spec) == set(feature_names(pollutant)), "every feature must declare its lookback"
        assert all(offset <= 0 for offset in spec.values())


def test_features_at_t_are_unchanged_when_the_future_changes(frames) -> None:
    """Perturb every value after t; features at t must be bit-identical (PRD §29)."""
    frame = frames["s1"]
    weather = synthetic_weather()
    t = frame.index[24 * 10]
    base = generate_features(frame, weather, P, *COORDS["s1"]).loc[t]

    tampered = frame.copy()
    future = tampered.index > t
    for col in [c for c in tampered.columns if "__" not in c]:
        tampered.loc[future, col] = 9999.0
    w2 = weather.copy()
    w2.loc[w2.index > t, :] = -50.0
    after = generate_features(tampered, w2, P, *COORDS["s1"]).loc[t]

    pd.testing.assert_series_equal(base, after)


def test_lag_features_read_exactly_k_hours_back(frames) -> None:
    frame = frames["s1"]
    X = generate_features(frame, synthetic_weather(), P, *COORDS["s1"])
    t = frame.index[100]
    for k in LAGS_HOURS:
        assert X.loc[t, f"pm25_lag_{k}h"] == frame.loc[t - pd.Timedelta(hours=k), "pm25"]


def test_weather_is_latest_row_at_or_before_t(frames) -> None:
    frame = frames["s1"]
    weather = synthetic_weather()
    X = generate_features(frame, weather, P, *COORDS["s1"])
    t = frame.index[50]  # hh:30
    assert X.loc[t, "wx_temperature"] == weather.loc[t.floor("h"), "temperature"]


def test_stale_weather_becomes_missing_not_carried_forever(frames) -> None:
    frame = frames["s1"]
    weather = synthetic_weather().iloc[:48]  # weather stops after 2 days
    X = generate_features(frame, weather, P, *COORDS["s1"])
    assert X["wx_temperature"].iloc[-1] != X["wx_temperature"].iloc[-1]  # NaN


def test_rolling_stats_with_insufficient_history_are_nan(frames) -> None:
    X = generate_features(frames["s1"], synthetic_weather(), P, *COORDS["s1"])
    assert np.isnan(X["pm25_roll24_mean"].iloc[0])
    assert not np.isnan(X["pm25_roll24_mean"].iloc[30])


def test_calendar_uses_ist_hour() -> None:
    from app.ml.features import build_calendar_features

    idx = pd.DatetimeIndex([pd.Timestamp("2026-01-01T00:30Z")])  # = 06:00 IST
    cal = build_calendar_features(idx)
    assert cal["cal_hour_sin"].iloc[0] == pytest.approx(np.sin(2 * np.pi * 6 / 24))


# --- targets & matrix --------------------------------------------------------------------


def test_targets_are_observed_future_values_only() -> None:
    obs = synthetic_obs(hours=72, stations=("s1",))
    gap_time = START + pd.Timedelta(hours=30)
    obs = obs[~((obs["pollutant"] == "pm25") & (obs["timestamp"] == gap_time))]
    frame = build_station_frames(obs)["s1"]
    Y = build_targets(frame, P)
    t = START + pd.Timedelta(hours=29)
    assert np.isnan(Y.loc[t, "h1"])  # the gap hour was forward-filled as input, never a target
    assert frame.loc[gap_time, "pm25"] == frame.loc[t, "pm25"]
    assert Y.loc[t, "h2"] == frame.loc[t + pd.Timedelta(hours=2), "pm25"]


def test_matrix_shape_order_and_horizons(frames) -> None:
    X, Y, S, _ = build_forecasting_matrix(frames, synthetic_weather(), P, COORDS)
    assert list(X.columns) == feature_names(P)
    assert list(Y.columns) == [f"h{h}" for h in HORIZONS] == list(S.columns)
    t = X.index.get_level_values("t")
    assert t.is_monotonic_increasing
    assert set(X.index.get_level_values("station_id")) == {"s1", "s2"}


def test_implausible_readings_are_neither_input_nor_target() -> None:
    obs = synthetic_obs(hours=48, stations=("s1",))
    bad = (obs["pollutant"] == "pm25") & (obs["timestamp"] == START + pd.Timedelta(hours=10))
    obs.loc[bad, ["concentration", "quality_flag"]] = [5000.0, "implausible"]
    frame = build_station_frames(obs)["s1"]
    assert frame["pm25"].max() < 1000
    assert not frame.loc[START + pd.Timedelta(hours=10), "pm25__observed"]


# --- models & evaluation -----------------------------------------------------------------


def test_persistence_repeats_current_for_every_horizon(frames) -> None:
    X, _, _, _ = build_forecasting_matrix(frames, synthetic_weather(), P, COORDS)
    pred = PersistenceForecaster("pm25_current").predict(X)
    assert pred.shape == (len(X), 24)
    assert (pred["h24"] == X["pm25_current"]).all()


def test_seasonal_naive_is_value_24h_before_target(frames) -> None:
    X, _, S, _ = build_forecasting_matrix(frames, synthetic_weather(), P, COORDS)
    frame = frames["s1"]
    sid, t = X.index[200]
    pred = SeasonalNaiveForecaster().predict(S)
    h = 5
    expected = frames[sid].loc[t + pd.Timedelta(hours=h - 24), "pm25"]
    assert pred.loc[(sid, t), f"h{h}"] == expected
    assert frame is not None


def test_validate_forecast_known_values() -> None:
    idx = pd.RangeIndex(2)
    obs = pd.DataFrame({f"h{h}": [10.0, 20.0] for h in HORIZONS}, index=idx)
    pred = obs + 2.0
    m = validate_forecast(pred, obs)["overall"]
    assert (
        m["mae"] == pytest.approx(2.0)
        and m["rmse"] == pytest.approx(2.0)
        and m["bias"] == pytest.approx(2.0)
    )
    perfect = validate_forecast(obs, obs)["overall"]
    assert perfect["r2"] == pytest.approx(1.0) and perfect["mae"] == 0


def test_validate_forecast_skips_missing_pairs() -> None:
    obs = pd.DataFrame({f"h{h}": [10.0, np.nan] for h in HORIZONS})
    pred = pd.DataFrame({f"h{h}": [12.0, 50.0] for h in HORIZONS})
    assert validate_forecast(pred, obs)["overall"]["n"] == 24


def test_compare_models_reports_skill_vs_persistence() -> None:
    obs = pd.DataFrame({f"h{h}": [10.0, 20.0] for h in HORIZONS})
    table = compare_forecast_models(
        {"persistence": validate_forecast(obs + 4, obs), "better": validate_forecast(obs + 1, obs)}
    )
    assert table[0]["model"] == "better"
    assert table[0]["skill_vs_persistence"] == pytest.approx(0.75)


@pytest.fixture(scope="module")
def trained(frames):
    X, Y, _, _ = build_forecasting_matrix(frames, synthetic_weather(), P, COORDS)
    t = X.index.get_level_values("t")
    cut = t.min() + pd.Timedelta(days=12)
    tr, va = t < cut - pd.Timedelta(hours=24), t >= cut
    model = DirectGBMForecaster(feature_names(P), GBMConfig(n_estimators=40, min_child_samples=5))
    model.fit(X[tr], Y[tr], X[va], Y[va])
    return model, X[va]


def test_gbm_predicts_24_finite_nonnegative_horizons(trained) -> None:
    model, Xva = trained
    pred = model.predict(Xva)
    assert list(pred.columns) == [f"h{h}" for h in HORIZONS]
    assert np.isfinite(pred.to_numpy()).all()
    assert (pred.to_numpy() >= 0).all()


def test_gbm_same_input_same_output(trained) -> None:
    model, Xva = trained
    pd.testing.assert_frame_equal(model.predict(Xva), model.predict(Xva))


# --- artifacts ---------------------------------------------------------------------------


def _meta(**kw) -> ForecastArtifactMetadata:
    base = dict(
        model_name="lightgbm_direct",
        model_version="aq_pm25_vtest",
        pollutant="pm25",
        feature_version="aq_features_v1.0.0",
        feature_names=feature_names(P),
        horizons=list(HORIZONS),
        training_dataset="aq_x",
        training_start="",
        training_end="",
        validation_start="",
        validation_end="",
        test_start="",
        test_end="",
        hyperparameters={},
        validation_metrics={},
        test_metrics={},
    )
    return ForecastArtifactMetadata(**(base | kw))


def test_artifact_round_trip(trained, tmp_path) -> None:
    model, Xva = trained
    save_model_artifact(model, _meta(), tmp_path)
    loaded, meta = load_model_artifact(tmp_path, feature_names(P))
    assert meta.model_version == "aq_pm25_vtest" and len(meta.model_sha256) == 64
    pd.testing.assert_frame_equal(loaded.predict(Xva), model.predict(Xva))


def test_artifact_rejected_on_feature_contract_mismatch(trained, tmp_path) -> None:
    save_model_artifact(trained[0], _meta(), tmp_path)
    with pytest.raises(ArtifactContractError, match="feature list"):
        load_model_artifact(tmp_path, feature_names(P)[:-1])


def test_artifact_rejected_on_feature_version_mismatch(trained, tmp_path) -> None:
    save_model_artifact(trained[0], _meta(feature_version="aq_features_v0.9.0"), tmp_path)
    with pytest.raises(ArtifactContractError, match="trained on"):
        load_model_artifact(tmp_path, feature_names(P))


def test_artifact_rejected_when_model_file_tampered(trained, tmp_path) -> None:
    save_model_artifact(trained[0], _meta(), tmp_path)
    with open(tmp_path / "model.joblib", "ab") as f:
        f.write(b"x")
    with pytest.raises(ArtifactContractError, match="checksum"):
        load_model_artifact(tmp_path, feature_names(P))


def test_artifact_metadata_has_prd_39_fields(trained, tmp_path) -> None:
    save_model_artifact(trained[0], _meta(), tmp_path)
    meta = json.loads((tmp_path / "metadata.json").read_text())
    for key in (
        "model_name",
        "model_version",
        "feature_version",
        "training_dataset",
        "training_start",
        "training_end",
        "validation_metrics",
        "test_metrics",
    ):
        assert key in meta


def test_dataset_fingerprint_is_deterministic_and_sensitive() -> None:
    obs = synthetic_obs(hours=24, stations=("s1",))
    assert dataset_fingerprint(obs) == dataset_fingerprint(obs.sample(frac=1, random_state=1))
    changed = obs.copy()
    changed.loc[0, "concentration"] += 1
    assert dataset_fingerprint(changed) != dataset_fingerprint(obs)
