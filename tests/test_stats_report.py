import io
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np
import plotly.graph_objects as go
from docx import Document
from PIL import Image

from core import report as report_module
from core.report import figure_png, generate_report
from core.scale import UNCALIBRATED, scale_from_um_per_px
from core.stats import (
    METRICS,
    distribution_figures,
    format_value,
    measure_stats,
    summarize,
    summary_stats,
)


def particle_df(scale):
    from core.measurement import compute_particle_properties

    mask = np.zeros((120, 120), bool)
    mask[10:30, 10:30] = True
    mask[60:100, 60:100] = True
    return summarize(compute_particle_properties(mask, pixel_size_um=scale.um_per_px))


class FormatTest(unittest.TestCase):
    def test_four_significant_digits_at_any_magnitude(self):
        self.assertEqual(format_value(40.1234), "40.12")
        self.assertEqual(format_value(0.0123456), "0.01235")  # '.2f' would print 0.01
        self.assertEqual(format_value(123456.0), "1.235e+05")


class StatsTest(unittest.TestCase):
    def test_summary_stats(self):
        import pandas as pd

        s = summary_stats(pd.DataFrame({"x": [1.0, 2.0, 3.0]}), "x")
        self.assertEqual((s["count"], s["mean"], s["min"], s["max"]), (3, 2.0, 1.0, 3.0))
        self.assertAlmostEqual(s["std"], 1.0)

    def test_single_row_has_zero_std_not_nan(self):
        import pandas as pd

        self.assertEqual(summary_stats(pd.DataFrame({"x": [5.0]}), "x")["std"], 0.0)

    def test_measure_stats_uses_the_scales_unit_columns(self):
        for scale, column in ((UNCALIBRATED, "equivalent_diameter_px"), (scale_from_um_per_px(0.5), "equivalent_diameter_um")):
            df = particle_df(scale)
            stats = measure_stats(df, scale, "particle")
            self.assertEqual(list(stats), ["equivalent_diameter"])
            self.assertAlmostEqual(stats["equivalent_diameter"]["mean"], df[column].mean())

    def test_figures_follow_metrics_and_label_the_unit(self):
        for kind, n in (("particle", 1), ("crack", 2)):
            self.assertEqual(len(METRICS[kind]), n)
        scale = scale_from_um_per_px(0.5)
        (label, fig), = distribution_figures(particle_df(scale), scale, "particle")
        self.assertEqual(label, "等效直径")
        self.assertIn("µm", fig.layout.xaxis.title.text)
        (_, fig_px), = distribution_figures(particle_df(UNCALIBRATED), UNCALIBRATED, "particle")
        self.assertIn("px", fig_px.layout.xaxis.title.text)


class ReportTest(unittest.TestCase):
    def build(self, scale, directory):
        df = particle_df(scale)
        out = Path(directory) / "not" / "yet" / "there" / "r.docx"  # report creates the directory
        generate_report(
            str(out), "sample.png", "专用模型-颗粒", np.zeros((20, 20, 3), np.uint8), df,
            measure_stats(df, scale, "particle"), distribution_figures(df, scale, "particle"), scale, "particle",
        )
        self.assertTrue(out.exists())
        doc = Document(str(out))
        return "\n".join(p.text for p in doc.paragraphs), doc

    def test_uncalibrated_report_declares_pixel_units(self):
        with tempfile.TemporaryDirectory() as d:
            text, doc = self.build(UNCALIBRATED, d)
        self.assertIn("未标定", text)
        self.assertIn("像素（px）", text)
        headers = [c.text for c in doc.tables[-1].rows[0].cells]
        self.assertIn("equivalent_diameter_px", headers)
        self.assertFalse([h for h in headers if h.endswith(("_um", "_um2"))])

    def test_calibrated_report_states_the_scale_and_no_pixel_disclaimer(self):
        scale = scale_from_um_per_px(0.5)
        with tempfile.TemporaryDirectory() as d:
            text, doc = self.build(scale, d)
        self.assertIn(scale.source, text)
        self.assertNotIn("未换算为物理尺寸", text)
        self.assertIn("equivalent_diameter_um", [c.text for c in doc.tables[-1].rows[0].cells])


class NoChromeFallbackTest(unittest.TestCase):
    """kaleido >= 1.0 needs a real Chrome to export plotly figures; without
    one 生成实验报告 used to raise ChromeNotFoundError."""

    @staticmethod
    def no_chrome(*args, **kwargs):
        raise RuntimeError("Kaleido requires Google Chrome to be installed.")

    def figure(self):
        scale = scale_from_um_per_px(0.5)
        return distribution_figures(particle_df(scale), scale, "particle")[0]

    def test_figure_png_falls_back_to_a_valid_png(self):
        label, fig = self.figure()
        with mock.patch.object(go.Figure, "to_image", self.no_chrome):
            png = figure_png(fig, label)
        image = Image.open(io.BytesIO(png))
        self.assertEqual(image.format, "PNG")
        self.assertGreater(image.width, 300)

    def test_report_still_builds_with_every_chart_when_export_fails(self):
        scale = scale_from_um_per_px(0.5)
        df = particle_df(scale)
        figures = distribution_figures(df, scale, "particle")
        with tempfile.TemporaryDirectory() as d, mock.patch.object(go.Figure, "to_image", self.no_chrome):
            out = generate_report(
                str(Path(d) / "r.docx"), "x.png", "t", np.zeros((10, 10, 3), np.uint8), df,
                measure_stats(df, scale, "particle"), figures, scale, "particle",
            )
            doc = Document(out)
        self.assertEqual(len(doc.inline_shapes), 1 + len(figures))  # overlay + each histogram

    def test_without_a_cjk_font_labels_degrade_to_ascii_not_empty_boxes(self):
        label, fig = self.figure()
        with mock.patch.object(report_module, "_CJK_FONTS", []), mock.patch.object(go.Figure, "to_image", self.no_chrome):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                figure_png(fig, label)
        self.assertFalse([w for w in caught if "missing from font" in str(w.message) or "Glyph" in str(w.message)])


if __name__ == "__main__":
    unittest.main()
