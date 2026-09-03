#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TrueIQ Small-N v3
=================
A deterministic, small-data-first DeepSWE leaderboard engine.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold, cross_val_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class Config:
    effort_order: Dict[str, int] = field(default_factory=lambda: {
        "low": 0,
        "medium": 1,
        "high": 2,
        "xhigh": 3,
        "max": 4,
        "default": 2,
        "none": 2,
        "": 2,
    })

    profile_peak_weights: Tuple[float, ...] = (0.35, 0.45, 0.55)
    upper_weight_shapes: Tuple[Tuple[float, float, float], ...] = (
        (0.20, 0.30, 0.50),
        (0.25, 0.35, 0.40),
    )
    breadth_weights: Tuple[float, ...] = (0.00, 0.05, 0.10)
    spam_tax_caps: Tuple[float, ...] = (0.20, 0.24, 0.28)
    spam_power: float = 1.25

    spam_pct_lo: float = 0.70
    spam_pct_hi: float = 0.95
    spam_run_gate: float = 0.25
    spam_min_efforts: int = 2
    spam_persistence_min: float = 0.50

    hunger_top_gain_pp: float = 5.0
    hunger_relative_to_max_gain: float = 0.60
    saturation_top_gain_pp: float = 3.0
    threshold_spike_ratio: float = 2.2
    flat_gain_pp: float = 2.5

    overthink_work_log_gain: float = 0.22
    overthink_gain_pp: float = 1.5
    efficient_gain_per_logwork: float = 8.0

    stable_top3_range_pp: float = 4.0
    variable_top3_range_pp: float = 8.0
    stable_ci_half_pp: float = 4.0

    surgeon_capability_pct: float = 0.70
    surgeon_steps_pct_max: float = 0.40
    surgeon_depth_pct_min: float = 0.60
    surgeon_delib_floor_pct: float = 0.25
    compute_heavy_steps_pct: float = 0.75
    compute_heavy_delib_pct: float = 0.55

    bootstrap_samples: int = 0
    bootstrap_seed: int = 1337

    diagnostic_rf_estimators: int = 500
    random_state: int = 42


BUILTIN_MODEL_MODES = {
    "kimi-k3": "fixed",
}


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def fnum(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def safe_div(a, b, default=np.nan) -> float:
    a, b = fnum(a), fnum(b)
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return default
    return a / b


def norm_model(s: str) -> str:
    return str(s).strip().lower()


def norm_effort(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "none"
    return str(x).strip().lower()


def ci_sigma_from_half(ci_half_fraction: float) -> float:
    h = fnum(ci_half_fraction)
    if not np.isfinite(h):
        return np.nan
    return h * 100.0 / 1.96


def percentile_rank(v: float, arr: Iterable[float]) -> float:
    vals = np.asarray([fnum(x) for x in arr], dtype=float)
    vals = vals[np.isfinite(vals)]
    v = fnum(v)
    if not np.isfinite(v) or len(vals) == 0:
        return 0.5
    return float((np.sum(vals < v) + 0.5 * np.sum(vals == v)) / len(vals))


def scale_percentile(p: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((p - lo) / (hi - lo), 0.0, 1.0))


def geometric_mean(vals: Sequence[float]) -> float:
    a = np.asarray([max(fnum(v, 0.0), 0.0) for v in vals], dtype=float)
    if len(a) == 0 or np.any(a <= 0):
        return 0.0
    return float(np.exp(np.mean(np.log(a))))


def weighted_mean(vals: Sequence[float], weights: Sequence[float]) -> float:
    v = np.asarray(vals, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    if not np.any(ok) or np.sum(w[ok]) <= 0:
        return np.nan
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))


def robust_mad(vals: Iterable[float], floor=1e-6) -> float:
    a = np.asarray([fnum(x) for x in vals], dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return floor
    med = np.median(a)
    return float(max(np.median(np.abs(a - med)) * 1.4826, floor))


def spearman_safe(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if np.sum(ok) < 3:
        return np.nan
    if spearmanr is None:
        return float(pd.Series(x[ok]).rank().corr(pd.Series(y[ok]).rank()))
    return float(spearmanr(x[ok], y[ok]).statistic)


def provider_family(model: str) -> str:
    m = norm_model(model)
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gpt"):
        return "openai"
    if m.startswith("gemini"):
        return "google"
    if m.startswith("glm"):
        return "zhipu"
    if m.startswith("kimi"):
        return "moonshot"
    if m.startswith("qwen"):
        return "alibaba"
    if m.startswith("deepseek"):
        return "deepseek"
    if m.startswith("grok"):
        return "xai"
    if m.startswith("muse"):
        return "muse"
    return "other"


# -----------------------------------------------------------------------------
# Loading and run feature engineering
# -----------------------------------------------------------------------------

class Loader:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load(self, path: str) -> Tuple[pd.DataFrame, dict]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        rows = raw.get("rows", raw if isinstance(raw, list) else [])
        if not isinstance(rows, list) or not rows:
            raise ValueError("Expected a JSON object with non-empty `rows` array.")
        df = pd.DataFrame(rows).copy()
        required = ["model", "pass_at_1", "pass_at_4", "mean_agent_steps", "mean_output_tokens"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        df["model"] = df["model"].map(norm_model)
        df["effort"] = df.get("reasoning_effort", pd.Series([None] * len(df))).map(norm_effort)
        df["effort_num"] = df["effort"].map(lambda x: self.cfg.effort_order.get(x, 2)).astype(int)
        return df, raw if isinstance(raw, dict) else {"rows": raw}


class FeatureEngineer:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        x = df.copy()
        x["pass1"] = pd.to_numeric(x["pass_at_1"], errors="coerce") * 100.0
        x["pass4"] = pd.to_numeric(x["pass_at_4"], errors="coerce") * 100.0
        x["steps"] = pd.to_numeric(x["mean_agent_steps"], errors="coerce")
        x["out_tokens"] = pd.to_numeric(x["mean_output_tokens"], errors="coerce")
        x["input_tokens"] = pd.to_numeric(x.get("mean_input_tokens", np.nan), errors="coerce")
        x["cache_tokens"] = pd.to_numeric(x.get("mean_cache_tokens", np.nan), errors="coerce")
        x["duration"] = pd.to_numeric(x.get("mean_duration_seconds", np.nan), errors="coerce")
        x["peak_context"] = pd.to_numeric(x.get("median_peak_context_tokens", np.nan), errors="coerce")
        x["cost"] = pd.to_numeric(x.get("mean_cost_usd", np.nan), errors="coerce")
        x["ci_half_pp"] = pd.to_numeric(x.get("ci_half", np.nan), errors="coerce") * 100.0
        x["pass1_sigma"] = x.get("ci_half", pd.Series(np.nan, index=x.index)).map(ci_sigma_from_half)

        x["tokens_per_step"] = x["out_tokens"] / x["steps"].replace(0, np.nan)
        x["steps_per_100k_out"] = x["steps"] / x["out_tokens"].replace(0, np.nan) * 100000.0
        x["sec_per_1k_out"] = x["duration"] / (x["out_tokens"].replace(0, np.nan) / 1000.0)
        x["sec_per_step"] = x["duration"] / x["steps"].replace(0, np.nan)
        x["input_per_step"] = x["input_tokens"] / x["steps"].replace(0, np.nan)
        x["context_per_step"] = x["peak_context"] / x["steps"].replace(0, np.nan)
        x["cache_ratio"] = x["cache_tokens"] / x["input_tokens"].replace(0, np.nan)

        work_cols = ["steps", "out_tokens", "input_tokens", "peak_context"]
        logs = []
        for c in work_cols:
            v = np.log1p(x[c].clip(lower=0))
            med = np.nanmedian(v)
            mad = robust_mad(v)
            logs.append((v - med) / mad)
        x["work_z"] = np.nanmean(np.column_stack(logs), axis=1)
        return x


# -----------------------------------------------------------------------------
# Breadth: Pass@4 residual relative to Pass@1
# -----------------------------------------------------------------------------

class BreadthModel:
    def fit_transform(self, runs: pd.DataFrame) -> pd.DataFrame:
        out = runs.copy()
        p1 = out["pass1"].to_numpy(float)
        p4 = out["pass4"].to_numpy(float)
        ok = np.isfinite(p1) & np.isfinite(p4)
        expected = np.full(len(out), np.nan)
        if SKLEARN_AVAILABLE and np.sum(ok) >= 8:
            iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
            iso.fit(p1[ok], p4[ok])
            expected[ok] = iso.predict(p1[ok])
        else:
            if np.sum(ok) >= 3:
                coef = np.polyfit(p1[ok], p4[ok], 1)
                expected[ok] = np.clip(np.polyval(coef, p1[ok]), 0, 100)
        out["pass4_expected"] = expected
        out["breadth_residual"] = (out["pass4"] - out["pass4_expected"]).clip(-12.0, 12.0)
        return out


# -----------------------------------------------------------------------------
# Leave-one-model-out peer percentiles
# -----------------------------------------------------------------------------

class PeerNormalizer:
    METRICS = [
        "steps", "out_tokens", "tokens_per_step", "steps_per_100k_out",
        "sec_per_1k_out", "sec_per_step", "input_per_step", "context_per_step",
    ]

    def transform(self, runs: pd.DataFrame, cfg: Config) -> pd.DataFrame:
        out = runs.copy()
        for metric in self.METRICS:
            vals = []
            for _, r in out.iterrows():
                peers = out[(out["effort"] == r["effort"]) & (out["model"] != r["model"])][metric]
                peers = peers[np.isfinite(peers)]
                if len(peers) < 4:
                    peers = out[out["model"] != r["model"]][metric]
                    peers = peers[np.isfinite(peers)]
                vals.append(percentile_rank(r[metric], peers))
            out[f"pct_{metric}"] = vals

        out["pct_fast_tokens"] = 1.0 - out["pct_sec_per_1k_out"]
        out["pct_fast_steps"] = 1.0 - out["pct_sec_per_step"]

        spam_runs = []
        for _, r in out.iterrows():
            s = scale_percentile(r["pct_steps"], cfg.spam_pct_lo, cfg.spam_pct_hi)
            o = scale_percentile(r["pct_out_tokens"], cfg.spam_pct_lo, cfg.spam_pct_hi)
            f = scale_percentile(r["pct_fast_tokens"], cfg.spam_pct_lo, cfg.spam_pct_hi)
            spam_runs.append(geometric_mean([s, o, f]))
        out["spam_run_strength"] = spam_runs
        return out


# -----------------------------------------------------------------------------
# Model modes
# -----------------------------------------------------------------------------

def load_model_modes(path: Optional[str]) -> Dict[str, str]:
    modes = dict(BUILTIN_MODEL_MODES)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for k, v in raw.items():
            vv = str(v).strip().lower()
            if vv not in {"fixed", "tunable", "incomplete", "auto"}:
                raise ValueError(f"Invalid model mode {v!r} for {k!r}")
            modes[norm_model(k)] = vv
    return modes


def resolve_mode(model: str, n_efforts: int, modes: Dict[str, str]) -> str:
    explicit = modes.get(model, "auto")
    if explicit == "fixed":
        return "fixed"
    if explicit == "tunable":
        return "tunable"
    if explicit == "incomplete":
        return "incomplete"
    return "tunable" if n_efforts >= 2 else "single"


# -----------------------------------------------------------------------------
# Effort response and top-end efficiency
# -----------------------------------------------------------------------------

class EffortAnalyzer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def analyze(self, g: pd.DataFrame, mode: str) -> dict:
        gs = g.sort_values(["effort_num", "effort"]).copy()
        n = len(gs)
        if n <= 1:
            response = "FIXED-EFFORT" if mode == "fixed" else (
                "UNMEASURED-CURVE" if mode in {"tunable", "incomplete"} else "SINGLE-SETTING"
            )
            return {
                "effort_response": response,
                "hunger_strength": 0.0,
                "top_efficiency": "FIXED-RUN" if mode == "fixed" else "SINGLE-RUN",
                "top_gain": np.nan,
                "top_gain_z": np.nan,
                "top_dependency": np.nan,
                "top3_range": 0.0,
                "transition_notes": "",
            }

        p = gs["pass1"].to_numpy(float)
        sig = gs["pass1_sigma"].fillna(1.5).to_numpy(float)
        work = gs["work_z"].to_numpy(float)
        gains = np.diff(p)
        sigmas = np.sqrt(sig[:-1] ** 2 + sig[1:] ** 2)
        gain_z = gains / np.maximum(sigmas, 0.75)
        dwork = np.diff(work)
        top_gain = float(gains[-1])
        top_z = float(gain_z[-1])
        max_gain = float(np.max(gains)) if len(gains) else 0.0
        max_gain_idx = int(np.argmax(gains)) if len(gains) else 0
        top_dependency = safe_div(max(top_gain, 0.0), max(np.max(p), 1e-6), 0.0)

        hunger_abs = np.clip((top_gain - self.cfg.hunger_top_gain_pp) / 7.0, 0.0, 1.0)
        hunger_sig = np.clip((top_z - 1.0) / 2.0, 0.0, 1.0)
        hunger_rel = np.clip(safe_div(top_gain, max(max_gain, 1e-6), 0.0) /
                             max(self.cfg.hunger_relative_to_max_gain, 1e-6), 0.0, 1.0)
        hunger = float((0.45 * hunger_abs + 0.35 * hunger_sig + 0.20 * hunger_rel))

        earlier = gains[:-1] if len(gains) > 1 else np.array([])
        earlier_max = float(np.max(earlier)) if len(earlier) else 0.0
        positive = gains[gains > 0]
        typical_pos = float(np.median(positive)) if len(positive) else 0.0
        spike_ratio = safe_div(max_gain, max(typical_pos, 1e-6), 0.0)

        if top_gain >= self.cfg.hunger_top_gain_pp and top_z >= 1.0 and top_gain >= self.cfg.hunger_relative_to_max_gain * max(max_gain, 1e-6):
            response = "EFFORT-HUNGRY"
        elif max_gain_idx < len(gains) - 1 and max_gain >= max(6.0, 1.6 * max(typical_pos, 1.0)) and spike_ratio >= self.cfg.threshold_spike_ratio:
            response = "THRESHOLDED"
        elif earlier_max >= 5.0 and top_gain <= self.cfg.saturation_top_gain_pp:
            response = "SATURATING"
        elif np.max(np.abs(gains)) <= self.cfg.flat_gain_pp:
            response = "FLAT"
        else:
            response = "STEADY-SCALER"

        over_flags = []
        effs = []
        for j in range(max(0, len(gains) - 2), len(gains)):
            wg = max(float(dwork[j]), 0.0)
            qg = float(gains[j])
            z = float(gain_z[j])
            over = wg >= self.cfg.overthink_work_log_gain and (qg <= self.cfg.overthink_gain_pp or z < 0.75)
            over_flags.append(over)
            effs.append(safe_div(qg, max(wg, 1e-6), np.nan))
        if any(over_flags):
            top_eff = "OVERTHINKING"
        elif top_gain >= 5.0 and top_z >= 1.0:
            top_eff = "SCALING"
        elif np.isfinite(np.nanmedian(effs)) and np.nanmedian(effs) >= self.cfg.efficient_gain_per_logwork:
            top_eff = "EFFICIENT"
        else:
            top_eff = "DIMINISHING"

        notes = "; ".join(
            f"{gs.iloc[i]['effort']}->{gs.iloc[i+1]['effort']}: {gains[i]:+.2f}pp"
            for i in range(len(gains))
        )
        top3 = gs.tail(min(3, n))["pass1"].to_numpy(float)
        return {
            "effort_response": response,
            "hunger_strength": hunger,
            "top_efficiency": top_eff,
            "top_gain": top_gain,
            "top_gain_z": top_z,
            "top_dependency": top_dependency,
            "top3_range": float(np.max(top3) - np.min(top3)) if len(top3) else np.nan,
            "transition_notes": notes,
        }


# -----------------------------------------------------------------------------
# Model aggregation and phenotypes
# -----------------------------------------------------------------------------

class ModelBuilder:
    def __init__(self, cfg: Config, modes: Dict[str, str]):
        self.cfg = cfg
        self.modes = modes
        self.effort = EffortAnalyzer(cfg)

    def build(self, runs: pd.DataFrame) -> pd.DataFrame:
        rows = []
        prelim = {}
        for m, g in runs.groupby("model"):
            gs = g.sort_values(["effort_num", "effort"])
            top = gs.tail(min(3, len(gs)))["pass1"].to_numpy(float)
            prelim[m] = float(np.max(top)) if len(top) else np.nan
        prelim_vals = list(prelim.values())

        for m, g in runs.groupby("model"):
            gs = g.sort_values(["effort_num", "effort"]).copy()
            n = len(gs)
            mode = resolve_mode(m, n, self.modes)
            effort_info = self.effort.analyze(gs, mode)

            spam_vals = gs["spam_run_strength"].to_numpy(float)
            spam_gate = spam_vals >= self.cfg.spam_run_gate
            persistence = float(np.mean(spam_gate)) if len(spam_vals) else 0.0
            if n >= self.cfg.spam_min_efforts and persistence >= self.cfg.spam_persistence_min:
                spam_strength = float(np.median(spam_vals[spam_gate]) * math.sqrt(persistence)) if np.any(spam_gate) else 0.0
            else:
                spam_strength = 0.0
            local_mechanical = float(np.max(spam_vals)) if len(spam_vals) else 0.0

            upper = gs.tail(min(3, n))
            steps_pct = float(np.median(upper["pct_steps"]))
            depth_pct = float(np.median(upper["pct_tokens_per_step"]))
            delib_pct = float(np.median(upper["pct_sec_per_1k_out"]))
            cap_pct = percentile_rank(prelim[m], prelim_vals)

            surgical_parts = [
                scale_percentile(cap_pct, self.cfg.surgeon_capability_pct, 0.98),
                scale_percentile(1.0 - steps_pct, 1.0 - self.cfg.surgeon_steps_pct_max, 0.98),
                scale_percentile(depth_pct, self.cfg.surgeon_depth_pct_min, 0.98),
                max(scale_percentile(delib_pct, self.cfg.surgeon_delib_floor_pct, 0.90), 0.20),
            ]
            surgical_strength = geometric_mean(surgical_parts)
            if spam_strength >= 0.45:
                interaction = "SPAMMER"
            elif surgical_strength >= 0.55:
                interaction = "SURGICAL"
            elif steps_pct >= self.cfg.compute_heavy_steps_pct and delib_pct >= self.cfg.compute_heavy_delib_pct:
                interaction = "COMPUTE-HEAVY"
            elif steps_pct >= self.cfg.compute_heavy_steps_pct or local_mechanical >= 0.45:
                interaction = "CHURNING"
            else:
                interaction = "CLEAN"

            mean_ci = float(np.nanmean(gs["ci_half_pp"])) if np.any(np.isfinite(gs["ci_half_pp"])) else np.nan
            if mode == "fixed":
                evidence = "FIXED-EFFORT"
            elif n == 1:
                evidence = "SINGLE-SETTING"
            elif mode == "incomplete":
                evidence = "PARTIAL-CURVE"
            elif n >= 3 and effort_info["top3_range"] <= self.cfg.stable_top3_range_pp and (not np.isfinite(mean_ci) or mean_ci <= self.cfg.stable_ci_half_pp):
                evidence = "STABLE"
            elif n >= 3 and effort_info["top3_range"] >= self.cfg.variable_top3_range_pp:
                evidence = "VARIABLE"
            else:
                evidence = "SUPPORTED"

            rows.append({
                "model": m,
                "provider_family": provider_family(m),
                "mode": mode,
                "n_efforts": n,
                "efforts": ",".join(gs["effort"].tolist()),
                "peak_pass1": float(gs["pass1"].max()),
                "upper_pass1_mean": float(upper["pass1"].mean()),
                "breadth_residual_median": float(np.nanmedian(upper["breadth_residual"])),
                "spam_strength": float(np.clip(spam_strength, 0.0, 1.0)),
                "local_mechanical_strength": local_mechanical,
                "spam_persistence": persistence,
                "surgical_strength": float(np.clip(surgical_strength, 0.0, 1.0)),
                "upper_steps_pct": steps_pct,
                "upper_depth_pct": depth_pct,
                "upper_delib_pct": delib_pct,
                "interaction_style": interaction,
                "evidence": evidence,
                "mean_ci_half_pp": mean_ci,
                **effort_info,
            })
        out = pd.DataFrame(rows)
        out["phenotype"] = (
            out["interaction_style"] + " · " +
            out["effort_response"] + " · " +
            out["top_efficiency"] + " · " +
            out["evidence"]
        )
        return out


# -----------------------------------------------------------------------------
# Deterministic specification-curve scoring
# -----------------------------------------------------------------------------

class SpecificationScorer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _profile(self, g: pd.DataFrame, peak_weight: float, shape: Tuple[float, float, float], breadth_weight: float) -> float:
        gs = g.sort_values(["effort_num", "effort"]).copy()
        q = gs["pass1"].to_numpy(float) + breadth_weight * gs["breadth_residual"].fillna(0.0).to_numpy(float)
        n = len(q)
        if n == 1:
            return float(q[0])
        k = min(3, n)
        upper = q[-k:]
        if k == 2:
            w = np.array([0.40, 0.60])
        else:
            w = np.asarray(shape, dtype=float)
        upper_mean = weighted_mean(upper, w)
        peak = float(np.max(upper))
        return float(peak_weight * peak + (1.0 - peak_weight) * upper_mean)

    def score(self, runs: pd.DataFrame, models: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        m_info = models.set_index("model")
        spec_rows = []
        spec_id = 0
        for peak_w, shape, breadth_w, spam_cap in itertools.product(
            self.cfg.profile_peak_weights,
            self.cfg.upper_weight_shapes,
            self.cfg.breadth_weights,
            self.cfg.spam_tax_caps,
        ):
            spec_id += 1
            for m, g in runs.groupby("model"):
                base = self._profile(g, peak_w, shape, breadth_w)
                spam = fnum(m_info.loc[m, "spam_strength"], 0.0)
                factor = 1.0 - spam_cap * (max(spam, 0.0) ** self.cfg.spam_power)
                score = float(np.clip(base * factor, 0.0, 100.0))
                spec_rows.append({
                    "spec_id": spec_id,
                    "model": m,
                    "peak_weight": peak_w,
                    "upper_shape": "/".join(f"{x:.2f}" for x in shape),
                    "breadth_weight": breadth_w,
                    "spam_tax_cap": spam_cap,
                    "profile_before_benchmax": base,
                    "benchmax_factor": factor,
                    "score": score,
                })
        specs = pd.DataFrame(spec_rows)

        specs["rank"] = specs.groupby("spec_id")["score"].rank(ascending=False, method="average")
        agg = specs.groupby("model").agg(
            trueiq=("score", "median"),
            spec_score_p10=("score", lambda s: float(np.quantile(s, 0.10))),
            spec_score_p90=("score", lambda s: float(np.quantile(s, 0.90))),
            spec_score_min=("score", "min"),
            spec_score_max=("score", "max"),
            spec_rank_median=("rank", "median"),
            spec_rank_best=("rank", "min"),
            spec_rank_worst=("rank", "max"),
            profile_before_benchmax=("profile_before_benchmax", "median"),
            benchmax_factor=("benchmax_factor", "median"),
        ).reset_index()
        out = models.merge(agg, on="model", how="left")
        out["spec_width"] = out["spec_score_p90"] - out["spec_score_p10"]
        out["rank_sensitivity"] = out["spec_rank_worst"] - out["spec_rank_best"]
        out = out.sort_values(["trueiq", "peak_pass1"], ascending=[False, False]).reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)
        return out, specs


# -----------------------------------------------------------------------------
# Bootstrap uncertainty (optional, score-independent)
# -----------------------------------------------------------------------------

class BootstrapLab:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, runs: pd.DataFrame, models: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        B = int(self.cfg.bootstrap_samples)
        if B <= 0:
            return pd.DataFrame(), pd.DataFrame()
        rng = np.random.default_rng(self.cfg.bootstrap_seed)
        minfo = models.set_index("model")
        names = models["model"].tolist()
        samples = {m: np.zeros(B) for m in names}

        for m, g in runs.groupby("model"):
            base_score = fnum(minfo.loc[m, "trueiq"])
            upper = g.sort_values(["effort_num", "effort"]).tail(min(3, len(g)))
            sigs = upper["pass1_sigma"].to_numpy(float)
            sigs = sigs[np.isfinite(sigs)]
            sigma = float(np.sqrt(np.mean(sigs ** 2)) / max(math.sqrt(len(sigs)), 1.0)) if len(sigs) else 1.5
            spec_sigma = fnum(minfo.loc[m, "spec_width"], 0.0) / 2.56
            total_sigma = math.sqrt(max(sigma, 0.0) ** 2 + max(spec_sigma, 0.0) ** 2)
            samples[m] = np.clip(rng.normal(base_score, total_sigma, size=B), 0, 100)

        score_matrix = np.column_stack([samples[m] for m in names])
        ranks = np.empty_like(score_matrix)
        for b in range(B):
            order = np.argsort(-score_matrix[b])
            rr = np.empty(len(names), float)
            rr[order] = np.arange(1, len(names) + 1)
            ranks[b] = rr

        rows = []
        pairs = []
        for j, m in enumerate(names):
            s = score_matrix[:, j]
            r = ranks[:, j]
            rows.append({
                "model": m,
                "score_p05": float(np.quantile(s, 0.05)),
                "score_p50": float(np.quantile(s, 0.50)),
                "score_p95": float(np.quantile(s, 0.95)),
                "rank_mean": float(np.mean(r)),
                "rank_p05": float(np.quantile(r, 0.05)),
                "rank_p95": float(np.quantile(r, 0.95)),
                "p_top3": float(np.mean(r <= 3)),
                "p_top5": float(np.mean(r <= 5)),
                "p_top10": float(np.mean(r <= 10)),
            })
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                gap = score_matrix[:, i] - score_matrix[:, j]
                pairs.append({
                    "model_a": names[i],
                    "model_b": names[j],
                    "p_a_above_b": float(np.mean(gap > 0) + 0.5 * np.mean(gap == 0)),
                    "median_gap": float(np.median(gap)),
                    "gap_p05": float(np.quantile(gap, 0.05)),
                    "gap_p95": float(np.quantile(gap, 0.95)),
                })
        return pd.DataFrame(rows), pd.DataFrame(pairs)


# -----------------------------------------------------------------------------
# Diagnostics: correlations and optional small-N ML (never used in score)
# -----------------------------------------------------------------------------

class Diagnostics:
    RUN_FEATURES = [
        "steps", "out_tokens", "input_tokens", "duration", "peak_context", "cost",
        "tokens_per_step", "steps_per_100k_out", "sec_per_1k_out", "sec_per_step",
        "input_per_step", "context_per_step", "cache_ratio",
    ]

    def correlations(self, runs: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for f in self.RUN_FEATURES:
            if f not in runs:
                continue
            rho = spearman_safe(runs[f], runs["pass1"])
            by_eff = []
            for _, g in runs.groupby("effort"):
                r = spearman_safe(g[f], g["pass1"])
                if np.isfinite(r):
                    by_eff.append(r)
            rows.append({
                "feature": f,
                "spearman_global": rho,
                "spearman_within_effort_median": float(np.median(by_eff)) if by_eff else np.nan,
                "n_efforts_with_corr": len(by_eff),
            })
        return pd.DataFrame(rows).sort_values("spearman_global", key=lambda s: s.abs(), ascending=False)

    def ml_diagnostics(self, runs: pd.DataFrame, cfg: Config) -> dict:
        meta = {"available": False, "used_in_score": False}
        if not SKLEARN_AVAILABLE:
            return meta
        feats = [f for f in self.RUN_FEATURES if f in runs.columns]
        X = runs[feats].replace([np.inf, -np.inf], np.nan)
        feats = [f for f in feats if X[f].notna().sum() >= max(8, len(runs) // 3)]
        if len(feats) < 3 or runs["model"].nunique() < 8:
            return meta
        X = runs[feats].copy()
        for c in feats:
            X[c] = X[c].fillna(X[c].median())
        scaler = RobustScaler()
        Xs = scaler.fit_transform(X)
        y = runs["pass1"].to_numpy(float)
        groups = runs["model"].to_numpy()

        rf = RandomForestRegressor(
            n_estimators=cfg.diagnostic_rf_estimators,
            random_state=cfg.random_state,
            min_samples_leaf=3,
            max_features=0.7,
        )
        n_splits = min(5, len(np.unique(groups)))
        cv = GroupKFold(n_splits=n_splits)
        try:
            mae = -cross_val_score(rf, Xs, y, groups=groups, cv=cv, scoring="neg_mean_absolute_error").mean()
            r2 = cross_val_score(rf, Xs, y, groups=groups, cv=cv, scoring="r2").mean()
        except Exception:
            mae, r2 = np.nan, np.nan
        rf.fit(Xs, y)
        importance = sorted(zip(feats, rf.feature_importances_), key=lambda z: z[1], reverse=True)

        pca = PCA(n_components=min(5, Xs.shape[1], Xs.shape[0]))
        pca.fit(Xs)
        meta = {
            "available": True,
            "used_in_score": False,
            "warning": "Diagnostic only. Small-N ML is not trusted as a scorer.",
            "group_cv_mae": float(mae) if np.isfinite(mae) else None,
            "group_cv_r2": float(r2) if np.isfinite(r2) else None,
            "rf_importance": [{"feature": f, "importance": float(v)} for f, v in importance],
            "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        }
        return meta


# -----------------------------------------------------------------------------
# Run notes
# -----------------------------------------------------------------------------

class RunNoteEngine:
    def apply(self, runs: pd.DataFrame, models: pd.DataFrame) -> pd.DataFrame:
        out = runs.copy()
        mi = models.set_index("model")
        notes = []
        for _, r in out.iterrows():
            parts = []
            if r["spam_run_strength"] >= 0.60:
                parts.append("MECHANICAL-CHURN")
            elif r["pct_steps"] >= 0.80:
                parts.append("STEP-HEAVY")
            if r["pct_tokens_per_step"] >= 0.80:
                parts.append("DEEP-STEPS")
            if r["pct_sec_per_1k_out"] >= 0.80:
                parts.append("DELIBERATE")
            elif r["pct_fast_tokens"] >= 0.90:
                parts.append("FAST-LOOP")
            if not parts:
                parts.append("NORMAL-RUN")
            notes.append("+".join(parts))
        out["run_note"] = notes
        out["model_trueiq"] = out["model"].map(mi["trueiq"])
        out["model_phenotype"] = out["model"].map(mi["phenotype"])
        return out


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------

class Reporter:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def audit(self, models: pd.DataFrame, raw_meta: dict, diag_meta: dict) -> str:
        lines = []
        lines.append("# TrueIQ Small-N v3 audit")
        lines.append("")
        lines.append("## Dataset")
        lines.append(f"- Scope: {raw_meta.get('scope', 'unknown')}")
        lines.append(f"- Generated at: {raw_meta.get('generated_at', 'unknown')}")
        lines.append(f"- Task set: {raw_meta.get('n_tasks_in_set', 'unknown')}")
        lines.append("")
        lines.append("## Top models")
        lines.append("")
        lines.append("| rank | model | TrueIQ | spec 10–90 | profile | phenotype |")
        lines.append("|---:|---|---:|---:|---:|---|")
        for _, r in models.head(30).iterrows():
            lines.append(
                f"| {int(r['rank'])} | {r['model']} | {r['trueiq']:.2f} | "
                f"{r['spec_score_p10']:.2f}–{r['spec_score_p90']:.2f} | "
                f"{r['profile_before_benchmax']:.2f} | {r['phenotype']} |"
            )
        lines.append("")
        return "\n".join(lines)

    def write_frontend_json(self, path: Path, models: pd.DataFrame, runs: pd.DataFrame, raw_meta: dict):
        frontend_data = {
            "meta": {
                "generated_at": raw_meta.get("generated_at", ""),
                "n_tasks_in_set": raw_meta.get("n_tasks_in_set", 113),
                "n_models": int(models["model"].nunique()),
                "n_runs": int(len(runs)),
            },
            "models": []
        }

        m_info = models.set_index("model")
        
        for model_name, g in runs.groupby("model"):
            m = m_info.loc[model_name]
            
            model_runs = []
            for _, r in g.sort_values("effort_num").iterrows():
                model_runs.append({
                    "effort": r["effort"],
                    "steps": fnum(r["steps"]),
                    "out_tokens": fnum(r["out_tokens"]),
                    "cost": fnum(r["cost"]),
                    "duration": fnum(r["duration"]),
                    "run_q": fnum(r["pass1"]),
                    "pass1": fnum(r["pass1"]),
                    "run_note": r.get("run_note", "")
                })

            frontend_data["models"].append({
                "model_id": model_name,
                "trueiq": fnum(m["trueiq"]),
                "conservative_score": fnum(m["spec_score_p10"]),
                "profile_q": fnum(m["profile_before_benchmax"]),
                "peak_p1": fnum(m["peak_pass1"]),
                "n_efforts": int(m["n_efforts"]),
                "spam_strength": fnum(m["spam_strength"]),
                "interaction_style": str(m["interaction_style"]),
                "effort_response": str(m["effort_response"]),
                "top_efficiency": str(m["top_efficiency"]),
                "evidence": str(m["evidence"]),
                "runs": model_runs
            })

        frontend_data["models"].sort(key=lambda x: x["trueiq"], reverse=True)
        path.write_text(json.dumps(frontend_data, indent=2, ensure_ascii=False), encoding="utf-8")


# -----------------------------------------------------------------------------
# Main engine
# -----------------------------------------------------------------------------

class TrueIQSmallN:
    def __init__(self, cfg: Config, model_modes: Optional[Dict[str, str]] = None):
        self.cfg = cfg
        self.model_modes = model_modes or dict(BUILTIN_MODEL_MODES)

    def run(self, input_path: str, outdir: str) -> dict:
        outp = Path(outdir)
        outp.mkdir(parents=True, exist_ok=True)

        runs, raw = Loader(self.cfg).load(input_path)
        runs = FeatureEngineer().transform(runs)
        runs = BreadthModel().fit_transform(runs)
        runs = PeerNormalizer().transform(runs, self.cfg)

        models = ModelBuilder(self.cfg, self.model_modes).build(runs)
        models, specs = SpecificationScorer(self.cfg).score(runs, models)
        runs = RunNoteEngine().apply(runs, models)

        boot, pairs = BootstrapLab(self.cfg).run(runs, models)
        if len(boot):
            models = models.merge(boot, on="model", how="left")

        diag = Diagnostics()
        corr = diag.correlations(runs)
        diag_meta = diag.ml_diagnostics(runs, self.cfg)

        reporter = Reporter(self.cfg)
        audit = reporter.audit(models, raw, diag_meta)
        reporter.write_frontend_json(outp / "trueiq_frontend.json", models, runs, raw)

        models.to_csv(outp / "models_scored.csv", index=False)
        runs.to_csv(outp / "runs_scored.csv", index=False)
        specs.to_csv(outp / "specification_scores.csv", index=False)
        corr.to_csv(outp / "correlations.csv", index=False)
        if len(boot):
            boot.to_csv(outp / "bootstrap_uncertainty.csv", index=False)
        if len(pairs):
            pairs.to_csv(outp / "pairwise_probabilities.csv", index=False)
        (outp / "audit_report.md").write_text(audit, encoding="utf-8")

        metadata = {
            "input": str(input_path),
            "outdir": str(outdir),
            "config": asdict(self.cfg),
            "model_modes": self.model_modes,
            "n_runs": int(len(runs)),
            "n_models": int(models["model"].nunique()),
            "diagnostic_ml": diag_meta,
            "score_used_ml": False,
            "point_score_random": False,
            "bootstrap_affects_point_rank": False,
        }
        (outp / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return {"models": models, "runs": runs, "specs": specs, "metadata": metadata}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TrueIQ Small-N v3 deterministic DeepSWE scorer")
    p.add_argument("--input", required=True, help="leaderboard-live.json / deepswe.json")
    p.add_argument("--outdir", default="data")
    p.add_argument("--model-modes", default=None, help="optional JSON mapping model -> fixed/tunable/incomplete/auto")
    p.add_argument("--bootstrap", type=int, default=0, help="optional uncertainty draws")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = Config(bootstrap_samples=max(0, args.bootstrap))
    modes = load_model_modes(args.model_modes)
    engine = TrueIQSmallN(cfg, modes)
    result = engine.run(args.input, args.outdir)
    models = result["models"]
    print(models[["rank", "model", "trueiq", "phenotype"]].head(30).to_string(index=False))
    print(f"\nWrote outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
