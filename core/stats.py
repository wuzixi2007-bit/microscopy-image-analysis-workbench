"""Aggregate per-target measurements into summary stats + the distribution
charts. Independent of the detection models, implemented now."""

from dataclasses import dataclass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .scale import Scale, column_names


def format_value(value: float) -> str:
    """4 significant digits, not fixed decimals: once a scale is applied the
    same quantity can be ~40 px or ~0.18 µm, and a fixed '.2f' would show
    a 0.012 µm standard deviation as '0.01'."""
    return f"{value:.4g}"


def summarize(properties: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(properties)


@dataclass(frozen=True)
class Metric:
    key: str  # column_names() key; the unit-bearing column name depends on the scale
    label: str  # the per-target quantity: axis title, std/min/max rows, histogram heading
    mean_label: str  # label for the across-targets mean row
    plot_title: str


# What gets summarized and charted per target kind. "particle" is the
# equivalent diameter of a compact blob; "crack" is length and mean width,
# since equivalent diameter is meaningless for a thin elongated object (see
# core/measurement.py's compute_crack_properties). First entry = headline.
METRICS = {
    "particle": [Metric("equivalent_diameter", "等效直径", "平均等效直径", "粒径分布 (Particle Size Distribution)")],
    "crack": [
        Metric("length", "长度", "平均长度", "裂纹长度分布"),
        Metric("mean_width", "平均宽度", "平均宽度（各条裂纹均值）", "裂纹平均宽度分布"),
    ],
}


def summary_stats(df: pd.DataFrame, column: str) -> dict:
    return {
        "count": int(df[column].count()),
        "mean": float(df[column].mean()),
        "std": float(df[column].std()) if len(df) > 1 else 0.0,
        "min": float(df[column].min()),
        "max": float(df[column].max()),
    }


def measure_stats(df: pd.DataFrame, scale: Scale, kind: str) -> dict[str, dict]:
    """summary_stats for each of `kind`'s metrics, keyed by Metric.key.
    The column name carries the unit, so it's asked of the Scale rather than
    hardcoded."""
    names = column_names(scale.calibrated)
    return {m.key: summary_stats(df, names[m.key]) for m in METRICS[kind]}


def plot_distribution(df: pd.DataFrame, column: str, unit: str, label: str, title: str) -> go.Figure:
    fig = px.histogram(df, x=column, nbins=30, title=title)
    fig.update_layout(xaxis_title=f"{label} ({unit})", yaxis_title="count")
    return fig


def distribution_figures(df: pd.DataFrame, scale: Scale, kind: str) -> list[tuple[str, go.Figure]]:
    """(label, figure) per metric of `kind`, in METRICS order."""
    names = column_names(scale.calibrated)
    return [
        (m.label, plot_distribution(df, names[m.key], scale.length_unit, m.label, m.plot_title)) for m in METRICS[kind]
    ]
