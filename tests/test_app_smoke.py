"""Drives app.py's real run_measurement_ui() through Streamlit's AppTest with
synthetic masks (file uploaders can't be driven by AppTest, so the detection
step is skipped; everything from measurement on is the real code)."""

import unittest

from streamlit.testing.v1 import AppTest

SCRIPT = '''
import numpy as np
from pathlib import Path
exec(Path("app.py").read_text(encoding="utf-8"))
img = np.full((120, 160, 3), 200, np.uint8)
masks = np.zeros((3, 120, 160), bool)
masks[0, 10:40, 10:40] = True
masks[0, 60:90, 60:90] = True   # second component inside the SAME mask
masks[1, 10:40, 100:130] = True
masks[2, 80:110, 10:45] = True
run_measurement_ui(img, masks, "synthetic.png", "t", kind="{kind}")
'''


def run(kind="particle", um_per_px=None):
    at = AppTest.from_string(SCRIPT.format(kind=kind), default_timeout=60).run()
    if um_per_px is not None:
        at.sidebar.radio(key="default_scale_mode").set_value("直接输入 µm/px").run()  # the number box only exists after this rerun
        at.sidebar.number_input(key="default_um_per_px").set_value(um_per_px)
        at.run()
    return at


class AppSmokeTest(unittest.TestCase):
    def test_labels_are_numbered_across_the_whole_image(self):
        for kind in ("particle", "crack"):
            with self.subTest(kind=kind):
                at = run(kind)
                self.assertFalse(at.exception)
                self.assertEqual(list(at.dataframe[0].value["label"]), [1, 2, 3, 4])

    def test_uncalibrated_columns_are_in_pixels(self):
        at = run("particle")
        cols = list(at.dataframe[0].value.columns)
        self.assertIn("area_px2", cols)
        self.assertFalse([c for c in cols if c.endswith(("_um", "_um2"))])
        self.assertTrue(any("未标定" in c.value for c in at.caption))

    def test_sidebar_scale_switches_columns_and_values(self):
        px = run("particle").dataframe[0].value
        um = run("particle", um_per_px=0.5).dataframe[0].value
        self.assertIn("area_um2", um.columns)
        self.assertNotIn("area_px2", um.columns)
        self.assertAlmostEqual(um["area_um2"].iloc[0], px["area_px2"].iloc[0] * 0.25)

    def test_crack_mode_reports_length_and_width(self):
        cols = list(run("crack").dataframe[0].value.columns)
        self.assertIn("length_px", cols)
        self.assertIn("mean_width_px", cols)
        self.assertNotIn("equivalent_diameter_px", cols)

    def test_option_labels_carry_no_parenthetical_notes(self):
        at = run("particle")
        self.assertEqual(list(at.sidebar.radio[0].options), ["少样本", "专用模型"])
        self.assertEqual(
            list(at.sidebar.radio(key="default_scale_mode").options), ["未标定", "直接输入 µm/px", "按标尺长度换算"]
        )

    def test_zero_scale_falls_back_to_pixels_with_a_warning(self):
        at = run("particle", um_per_px=0.0)
        self.assertIn("area_px2", at.dataframe[0].value.columns)
        self.assertTrue(at.sidebar.warning)


EDIT_SCRIPT = '''
import numpy as np
from pathlib import Path
exec(Path("app.py").read_text(encoding="utf-8"))
img = np.full((100, 160, 3), 200, np.uint8)
masks = np.zeros((3, 100, 160), bool)
masks[0, 10:40, 10:40] = True
masks[1, 10:40, 40:70] = True      # touches instance 0
masks[2, 60:90, 100:130] = True
out, n_pinned = mask_correction_ui(img, masks, "edit_t", sam, preserve_groups={preserve}, keep_instances={keep})
st.session_state["_n_rows"] = int(out.shape[0])
st.session_state["_n_pinned"] = int(n_pinned)
'''


def edit_rows(preserve=False, keep=False):
    at = AppTest.from_string(EDIT_SCRIPT.format(preserve=preserve, keep=keep), default_timeout=60).run()
    at.checkbox(key="edit_t_edit_on").set_value(True).run()
    assert not at.exception, at.exception
    return at.session_state["_n_rows"], at.session_state["_n_pinned"]


class EditInstancesTest(unittest.TestCase):
    def test_default_editor_fuses_touching_instances(self):
        self.assertEqual(edit_rows(), (2, 0))

    def test_keep_instances_preserves_touching_neighbours(self):
        self.assertEqual(edit_rows(keep=True), (3, 0))

    def test_preserve_groups_measures_every_row_whole(self):
        self.assertEqual(edit_rows(preserve=True), (3, 3))


if __name__ == "__main__":
    unittest.main()
