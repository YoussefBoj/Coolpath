from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from coolpath.thermal.comfypack import load_comfypack, quality_control
from coolpath.thermal.comfort import tmrt_iso7726, wind_to_10m, compute_utci
from coolpath.thermal.history import (
    exponential_history,
    exponential_history_variable_dt,
    apply_lag,
    calibrate_globe_inertia,
    unreliable_history_windows,
    reliable_after_gaps,
)
from coolpath.analysis.statistics import (
    robust_correlation,
    fdr_bh,
    partial_spearman_block_permutation,
    morans_i,
    ols_hac,
)

DEFAULT_REGEX = r"(?P<chapitre>GS\d+).*__f(?P<frame>\d+)"
MATERIAL_COLUMNS = [
    "pct_Asphalt",
    "pct_Cement_Concrete",
    "pct_Soil_Mud",
    "pct_Glass",
    "pct_Metal",
    "pct_Paint",
]


def timestamp_images(
    ind: pd.DataFrame,
    t0: dict,
    fps: dict,
    day="2026-06-11",
    regex=DEFAULT_REGEX,
    default_fps=59.94,
):
    """Reconstruct panorama timestamps from chapter start time and frame index."""
    ind = ind.copy()
    timestamps = []
    for name in ind.image.astype(str):
        match = re.search(regex, name)
        if not match:
            timestamps.append(pd.NaT)
            continue
        chapter = match.group("chapitre")
        frame = int(match.group("frame"))
        if chapter not in t0:
            timestamps.append(pd.NaT)
            continue
        start = pd.Timestamp(f"{day} {t0[chapter]}")
        timestamps.append(start + pd.to_timedelta(frame / fps.get(chapter, default_fps), unit="s"))
    ind["ts"] = pd.to_datetime(timestamps)
    return ind


def retain_frames_with_sensor_support(ind: pd.DataFrame, cp: pd.DataFrame, tolerance_s=2.0):
    """Keep only frames having a Comfy'Pack measurement within the notebook tolerance."""
    left = ind.dropna(subset=["ts"]).sort_values("ts").copy()
    right = cp[["ts"]].assign(_sensor=True).sort_values("ts")
    # merge_asof requires exactly matching datetime precisions.
    left["ts"] = left["ts"].astype("datetime64[ns]")
    right["ts"] = right["ts"].astype("datetime64[ns]")
    out = pd.merge_asof(
        left,
        right,
        on="ts",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance_s),
    )
    orphan_count = int(out["_sensor"].isna().sum())
    out = out[out["_sensor"].notna()].drop(columns="_sensor").reset_index(drop=True)
    return out, orphan_count


def aggregate(cp, ind, window_s, min_sensor=2, min_images=1):
    """Aggregate sensor means and morphology medians in common time windows."""
    origin = min(cp.ts.min(), ind.ts.min())
    cp = cp.copy()
    ind = ind.copy()
    cp["bin"] = ((cp.ts - origin).dt.total_seconds() // window_s).astype(int)
    ind["bin"] = ((ind.ts - origin).dt.total_seconds() // window_s).astype(int)

    sensor_cols = [
        c
        for c in ["Kdown", "Tg", "Ta", "RH", "WS", "v10", "Tmrt", "UTCI", "Prt_Mid", "Kdown_hist"]
        if c in cp
    ]
    indicator_cols = [
        c
        for c in ind.columns
        if c not in {"image", "ts", "bin"} and pd.api.types.is_numeric_dtype(ind[c])
    ]

    sensors = cp.groupby("bin")[sensor_cols].mean()
    sensors["n_mesures"] = cp.groupby("bin").size()
    morphology = ind.groupby("bin")[indicator_cols].median()
    morphology["n_images"] = ind.groupby("bin").size()
    table = sensors.join(morphology, how="inner").reset_index()
    table = table[(table.n_mesures >= min_sensor) & (table.n_images >= min_images)].copy()
    table["t_mid_s"] = (table["bin"] + 0.5) * window_s
    return table.sort_values("t_mid_s").reset_index(drop=True)


def _build_current_history_comparison(table, bases, targets):
    rows = []
    for base in bases:
        historical = f"{base}_hist"
        if base not in table or historical not in table:
            continue
        for target in targets:
            if target not in table:
                continue
            common = (
                np.isfinite(table[base])
                & np.isfinite(table[historical])
                & np.isfinite(table[target])
            )
            current = robust_correlation(table.loc[common, base], table.loc[common, target])
            hist = robust_correlation(table.loc[common, historical], table.loc[common, target])
            rows.append(
                {
                    "indicateur": base,
                    "cible": target,
                    "r_courant": current["r"],
                    "p_courant": current["p_corrigee"],
                    "r_hist": hist["r"],
                    "p_hist": hist["p_corrigee"],
                    "gain_abs": abs(hist["r"]) - abs(current["r"]),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["hist_meilleur"] = np.where(result.gain_abs > 0, "oui", "non")
    return result


def run_analysis(
    indicators_csv,
    fusion_csv,
    comfypack_csv,
    output_dir,
    t0,
    fps,
    day="2026-06-11",
    windows=(30, 60, 120),
    reference_window=60,
    ta_source="HMP_Temp",
    tau=None,
    lag=None,
    sensor_tolerance_s=2.0,
    gap_max_s=20.0,
    fabricated_weight_limit=0.20,
):
    """Run the scientific analysis refactored from notebook 30.

    Outputs keep the notebook's main tables: QC, inertia calibration, temporal
    segments, all Pearson/Spearman correlations, current-vs-history comparison,
    partial Spearman block permutations, global FDR, Moran I, and reduced HAC
    models used for the radiative-mediation interpretation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    indicators = pd.read_csv(indicators_csv)
    fusion = pd.read_csv(fusion_csv)
    ind = indicators.merge(fusion, on="image", how="left", suffixes=("", "_fusion"))
    ind = timestamp_images(ind, t0, fps, day)

    cp = load_comfypack(comfypack_csv, ta_source)
    cp, qc = quality_control(cp)
    qc.to_csv(output_dir / "controle_qualite.csv", index=False)

    ind, orphan_count = retain_frames_with_sensor_support(ind, cp, sensor_tolerance_s)

    cp["v10"] = wind_to_10m(cp["WS"])
    cp["Tmrt"] = tmrt_iso7726(cp["Tg"], cp["Ta"], cp["WS"])
    cp["UTCI"] = compute_utci(cp["Ta"], cp["Tmrt"], cp["RH"], cp["v10"])

    dt = float(cp.ts.diff().dt.total_seconds().median())
    best, grid = calibrate_globe_inertia(cp.Kdown.to_numpy(), cp.Tg.to_numpy(), dt)
    pd.DataFrame(grid).to_csv(output_dir / "calibration_inertie_globe.csv", index=False)
    tau = float(tau if tau is not None else best["tau_s"])
    lag = float(lag if lag is not None else best["lag_s"])

    sensor_t = (cp.ts - cp.ts.iloc[0]).dt.total_seconds().to_numpy()
    cp["Kdown_hist"] = apply_lag(
        exponential_history(cp.Kdown.to_numpy(), dt, tau),
        sensor_t,
        lag,
    )

    current_indicators = [
        c
        for c in [
            "SVF",
            "GVI_haute_pct",
            "GVI_basse_pct",
            "taux_impermeable_pct",
            "albedo_moyen",
            "canopee_sol_non_observable_pct",
            "taux_bati_pct",
        ] + MATERIAL_COLUMNS
        if c in ind
    ]

    ind = ind.sort_values("ts").reset_index(drop=True)
    image_t = (ind.ts - ind.ts.min()).dt.total_seconds().to_numpy()
    historized = []
    for column in current_indicators:
        values = pd.to_numeric(ind[column], errors="coerce").to_numpy(float)
        if np.isfinite(values).sum() < 5 or np.nanstd(values) <= 1e-12:
            continue
        hist = exponential_history_variable_dt(values, image_t, tau)
        ind[f"{column}_hist"] = apply_lag(hist, image_t, lag)
        historized.append(column)

    # Notebook-30 reliability correction around large gaps between video chapters.
    windows_unreliable = unreliable_history_windows(
        image_t,
        tau,
        gap_max_s=gap_max_s,
        fabricated_weight_limit=fabricated_weight_limit,
    )
    reliable = reliable_after_gaps(image_t, windows_unreliable, lag_s=lag)
    for column in historized:
        ind.loc[~reliable, f"{column}_hist"] = np.nan

    tables = {int(w): aggregate(cp, ind, int(w)) for w in windows}
    for window_s, table in tables.items():
        table.to_csv(output_dir / f"table_segments_{window_s}s.csv", index=False)

    ref = tables[int(reference_window)]
    targets = [
        c
        for c in ["Kdown", "Kdown_hist", "Tg", "Ta", "RH", "v10", "Tmrt", "UTCI", "Prt_Mid"]
        if c in ref
    ]
    predictors = current_indicators + [f"{c}_hist" for c in historized if f"{c}_hist" in ref]

    rows = []
    for window_s, table in tables.items():
        for predictor in predictors:
            if predictor not in table:
                continue
            for target in targets:
                if target not in table:
                    continue
                rec = robust_correlation(table[predictor], table[target])
                rec.update({"fenetre_s": window_s, "indicateur": predictor, "cible": target})
                rows.append(rec)
    matrix = pd.DataFrame(rows)
    matrix["rejet_fdr"], matrix["q_value"] = fdr_bh(matrix.p_corrigee.to_numpy())
    matrix.to_csv(output_dir / "matrice_correlations.csv", index=False)

    comparison_targets = [c for c in ["Kdown", "Tg", "Ta", "RH", "v10", "Tmrt", "UTCI", "Prt_Mid"] if c in ref]
    comparison = _build_current_history_comparison(ref, historized, comparison_targets)
    comparison.to_csv(output_dir / "comparaison_courant_vs_historique.csv", index=False)

    partial = []
    for base in ["GVI_haute_pct", "SVF"]:
        for predictor in [base, f"{base}_hist"]:
            if predictor not in ref:
                continue
            for target in [c for c in ["Tmrt", "UTCI"] if c in ref]:
                control_sets = [
                    ("temps", ["t_mid_s"]),
                    ("temps_Ta", ["t_mid_s", "Ta"]),
                    ("temps_Ta_Kdown", ["t_mid_s", "Ta", "Kdown_hist"]),
                    ("complet", ["t_mid_s", "Ta", "Kdown_hist", "RH", "v10"]),
                ]
                for label, control_names in control_sets:
                    controls = [ref[c] for c in control_names if c in ref]
                    rho, p, n = partial_spearman_block_permutation(
                        ref[predictor],
                        ref[target],
                        controls,
                        n_perm=999,
                        block_size=max(2, int(round(tau / reference_window)) + 1),
                    )
                    partial.append(
                        {
                            "indicateur": predictor,
                            "cible": target,
                            "controle": label,
                            "rho": rho,
                            "p_perm": p,
                            "n": n,
                        }
                    )
    pd.DataFrame(partial).to_csv(output_dir / "correlations_partielles.csv", index=False)

    moran = []
    for column in targets:
        I, p, n = morans_i(ref[column], ref.t_mid_s)
        moran.append({"variable": column, "I": I, "p_perm": p, "n": n})
    pd.DataFrame(moran).to_csv(output_dir / "autocorrelation_moran.csv", index=False)

    # Reduced HAC models: A (morphology history), B (+ radiation history), A' (current morphology).
    target = "Tmrt" if "Tmrt" in ref and ref.Tmrt.notna().sum() > 20 else "UTCI"
    model_rows = []
    models = {
        "A_hist": ["GVI_haute_pct_hist", "SVF_hist", "t_mid_s"],
        "B_hist_Kdown": ["GVI_haute_pct_hist", "SVF_hist", "t_mid_s", "Kdown_hist"],
        "A_current": ["GVI_haute_pct", "SVF", "t_mid_s"],
    }
    for model_label, names in models.items():
        names = [name for name in names if name in ref]
        if len(names) < 2:
            continue
        fit = ols_hac(ref[target], ref[names].to_numpy(), names)
        for name, beta, se, p in zip(fit["names"], fit["beta"], fit["se_hac"], fit["p_hac"]):
            model_rows.append(
                {
                    "modele": model_label,
                    "cible": target,
                    "variable": name,
                    "beta": beta,
                    "se_hac": se,
                    "p_hac": p,
                    "n": fit["n"],
                    "hac_lag": fit["lag"],
                }
            )
    pd.DataFrame(model_rows).to_csv(output_dir / "modeles_reduits.csv", index=False)

    summary = {
        "frames_with_sensor_support": int(len(ind)),
        "orphan_frames": orphan_count,
        "tau_s": tau,
        "lag_s": lag,
        "calibration_r": float(best["r"]),
        "history_unreliable_frames": int((~reliable).sum()),
        "n_tests": int(len(matrix)),
        "n_fdr": int(matrix.rejet_fdr.sum()),
    }
    pd.DataFrame([summary]).to_csv(output_dir / "resume_analyse.csv", index=False)
    return summary
