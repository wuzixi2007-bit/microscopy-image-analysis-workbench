"""Generate a .docx experiment report: annotated result image, summary
stats, distribution histogram(s), and the full per-target data table."""

import io
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: the report is built on a server thread, never a GUI
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from docx import Document
from docx.shared import Inches
from PIL import Image

from .scale import Scale
from .stats import METRICS, format_value


logger = logging.getLogger(__name__)

_CJK_FONTS = ["PingFang SC", "Heiti SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
# ASCII stand-ins for when no CJK font is installed (matplotlib would draw empty boxes)
_ASCII_LABELS = {"等效直径": "Equivalent diameter", "长度": "Length", "平均宽度": "Mean width"}


def _histogram_png_without_chrome(fig: go.Figure, label: str, unit_title: str) -> bytes:
    """Same histogram as the plotly figure, drawn with matplotlib. plotly's
    own image export (kaleido >= 1.0) drives a real Chrome and raises
    ChromeNotFoundError on a machine without one -- which would make 生成实验报告
    crash on exactly the kind of fresh machine a reviewer might run it on."""
    values = [float(v) for trace in fig.data for v in trace.x]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    cjk = next((name for name in _CJK_FONTS if name in installed), None)
    title, xlabel = fig.layout.title.text or "", fig.layout.xaxis.title.text or unit_title
    if cjk is None:
        title = "Distribution"
        xlabel = xlabel.replace(label, _ASCII_LABELS.get(label, label))
    with plt.rc_context({"font.sans-serif": [cjk] if cjk else ["DejaVu Sans"], "axes.unicode_minus": False}):
        fig_mpl, ax = plt.subplots(figsize=(8, 5), dpi=100)
        ax.hist(values, bins=30, color="#636efa", edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.yaxis.get_major_locator().set_params(integer=True)  # counts, no 2.5
        buf = io.BytesIO()
        fig_mpl.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig_mpl)
    return buf.getvalue()


def figure_png(fig: go.Figure, label: str) -> bytes:
    """PNG of a distribution figure: plotly's export when it works, the
    matplotlib redraw when it doesn't (no Chrome for kaleido, etc.)."""
    try:
        return fig.to_image(format="png", width=800, height=500)
    except Exception as exc:  # kaleido raises a variety of errors across versions
        logger.warning("plotly image export failed (%s: %s); drawing the histogram with matplotlib", type(exc).__name__, exc)
        return _histogram_png_without_chrome(fig, label, "")


def stat_rows(scale: Scale, kind: str) -> list[tuple[str, str, str]]:
    """(label, metric key, stat key) per row of the summary table; the
    metric key is "" for the count row."""
    unit = scale.length_unit
    rows = [("数量", "", "count")]
    for m in METRICS[kind]:
        rows += [
            (f"{m.mean_label} ({unit})", m.key, "mean"),
            (f"{m.label}标准差 ({unit})", m.key, "std"),
            (f"{m.label}最小值 ({unit})", m.key, "min"),
            (f"{m.label}最大值 ({unit})", m.key, "max"),
        ]
    return rows


def generate_report(
    output_path: str,
    source_name: str,
    mode: str,
    overlay_image,
    df: pd.DataFrame,
    stats: dict[str, dict],
    figures: list[tuple[str, go.Figure]],
    scale: Scale,
    kind: str = "particle",
) -> str:
    """`stats` is stats.measure_stats()'s output and `figures` is
    stats.distribution_figures()'s, both for the same `kind`."""
    doc = Document()

    doc.add_heading("显微图像智能分割与定量测量报告", level=1)
    doc.add_paragraph(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph(f"来源图像：{source_name}")
    doc.add_paragraph(f"识别模式：{mode}")
    doc.add_paragraph(f"尺度：{scale.source}")
    if not scale.calibrated:
        doc.add_paragraph("本报告所有长度以像素（px）、面积以平方像素（px²）为单位，未换算为物理尺寸。")

    doc.add_heading("分割结果", level=2)
    img_buf = io.BytesIO()
    Image.fromarray(overlay_image).save(img_buf, format="PNG")
    img_buf.seek(0)
    doc.add_picture(img_buf, width=Inches(6))

    doc.add_heading("统计摘要", level=2)
    table = doc.add_table(rows=1, cols=2)
    table.style = "Light Grid Accent 1"
    table.rows[0].cells[0].text = "指标"
    table.rows[0].cells[1].text = "数值"
    count = next(iter(stats.values()))["count"]
    for label, metric_key, stat_key in stat_rows(scale, kind):
        cells = table.add_row().cells
        cells[0].text = label
        value = count if stat_key == "count" else stats[metric_key][stat_key]
        cells[1].text = format_value(value) if isinstance(value, float) else str(value)

    for label, fig in figures:
        doc.add_heading(f"{label}分布直方图", level=2)
        hist_buf = io.BytesIO(figure_png(fig, label))
        doc.add_picture(hist_buf, width=Inches(6))

    doc.add_heading("详细数据", level=2)
    data_table = doc.add_table(rows=1, cols=len(df.columns))
    data_table.style = "Light Grid Accent 1"
    for i, col in enumerate(df.columns):
        data_table.rows[0].cells[i].text = str(col)
    for _, row in df.iterrows():
        cells = data_table.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = format_value(val) if isinstance(val, float) else str(val)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    return output_path
