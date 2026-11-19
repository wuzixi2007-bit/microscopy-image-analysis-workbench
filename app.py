"""Streamlit entry point (competition branch -- 零样本/Grounding-DINO mode
removed, see git history on main if it's ever needed back). Both remaining
modes run real inference against the shared SAM backend; M5 (manual mask
correction) is wired in as an optional step before measurement."""

import io

import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates

import config
from core.detectors.base import DetectionPrompt
from core.detectors.crack_seg import CrackSegmenter
from core.detectors.few_shot import FewShotDetector
from core.detectors.instance_seg import InstanceSegmenter
from core.detectors.particle_classifier import ParticleClassifier
from core.detectors.tiling import needs_tiling
from core.image_io import ImageLoadError, load_rgb
from core.mask_editor import apply_pinned_groups, canvas_layer_to_mask, masks_to_combined, stroke_bounding_box
from core.measurement import compute_crack_properties, compute_particle_properties
from core.preprocessing import preprocess
from core.reliability import check_few_shot_result, check_reference_selection
from core.report import generate_report
from core.scale import (
    LENGTH_UNITS,
    UNCALIBRATED,
    Scale,
    scale_from_field_width,
    scale_from_known_length,
    scale_from_um_per_px,
)
from core.segmentation.sam_wrapper import SamSegmenter
from core.stats import distribution_figures, format_value, measure_stats, summarize

st.set_page_config(page_title="显微图像智能分割与定量测量工作台", layout="wide")
st.title("显微图像智能分割与定量测量工作台")


@st.cache_resource(show_spinner="加载SAM模型…")
def load_sam():
    return SamSegmenter(config.SAM_CHECKPOINT, config.SAM_MODEL_TYPE, config.DEVICE)


@st.cache_resource(show_spinner=False)
def load_particle_classifier():
    """None if the classifier hasn't been trained yet (scripts/train_particle_classifier.py) --
    callers must treat that as "exhaustive mode unavailable", not an error."""
    path = config.MODELS_DIR / "particle_classifier.npz"
    return ParticleClassifier(str(path)) if path.exists() else None


@st.cache_resource(show_spinner="加载专用实例分割模型…")
def load_instance_segmenter():
    """None if the model hasn't been trained yet (scripts/train_instance_segmenter.py) --
    same "feature unavailable, not an error" pattern as load_particle_classifier()."""
    path = config.MODELS_DIR / "instance_segmenter.pth"
    return InstanceSegmenter(path, config.DEVICE) if path.exists() else None


@st.cache_resource(show_spinner="加载裂纹分割模型…")
def load_crack_segmenter():
    """None if the crack model hasn't been trained yet (scripts/train_crack_segmenter.py).
    Loaded lazily from the 专用模型 branch (only when 裂纹 is selected), not at
    startup like the particle model -- a particle-only session never pays for it."""
    path = config.MODELS_DIR / "crack_segmenter.pth"
    return CrackSegmenter(path, config.DEVICE) if path.exists() else None


sam = load_sam()
particle_classifier = load_particle_classifier()
instance_segmenter = load_instance_segmenter()


@st.cache_data(show_spinner="预处理中…")
def cached_preprocess(image_bytes: bytes, denoise_on: bool, equalize_on: bool, contrast_on: bool) -> np.ndarray:
    """`preprocess()` includes cv2.fastNlMeansDenoisingColored, which is slow
    (many seconds on a real image) -- Streamlit reruns the ENTIRE script on
    every widget interaction, including every mask_correction_ui() canvas
    stroke, so without this cache every stroke would re-run denoising from
    scratch (this combined with uncached detection was the exact cause of
    the page appearing to freeze after painting -- see README.md). Keyed on
    the raw file bytes rather than the UploadedFile object, which isn't
    stably hashable across reruns."""
    image = load_rgb(image_bytes)
    return preprocess(image, denoise_on=denoise_on, equalize_on=equalize_on, contrast_on=contrast_on)


mode = st.sidebar.radio("识别模式", ["少样本", "专用模型"])
st.sidebar.divider()
st.sidebar.subheader("预处理")
denoise_on = st.sidebar.checkbox("去噪", value=True)
equalize_on = st.sidebar.checkbox("均衡化", value=True)
contrast_on = st.sidebar.checkbox("对比度增强", value=False)


def _bar_drag_tool(key: str, image: np.ndarray) -> None:
    """Drag along a scale bar on the image; the measured pixel length is
    written into the bar-length input below it (same widget key, set before
    that widget is created in this run, which Streamlit allows). The
    component keeps returning the last drag on every rerun, so a drag is only
    applied once -- otherwise it would overwrite a value the user then types
    by hand."""
    st.caption(
        "在下图沿标尺从一端拖到另一端，像素长度会自动填入下方。请对准两端竖线的中心。"
        "标尺越长误差越小：一条约27 px的标尺，±1 px 就是约±3.7%；能读到视野宽度时，优先用「按视野宽度换算」。"
    )
    # width="stretch": by default the component draws the image at its NATURAL
    # size and clips it to the column -- found in a real-browser test, where a
    # 1024 px SEM frame in a ~414 px column was cut off before reaching the
    # scale bar at x~890-917, making the tool unusable. Stretch scales the
    # whole image to the column, so every part is reachable.
    drag = streamlit_image_coordinates(image, click_and_drag=True, width="stretch", key=f"{key}_bar_drag")
    info_key = f"{key}_bar_drag_info"
    if drag and drag.get("x2") is not None:
        drag_id = (drag.get("unix_time"), drag["x1"], drag["y1"], drag["x2"], drag["y2"])
        applied_key = f"{key}_bar_drag_applied"
        if st.session_state.get(applied_key) != drag_id:
            st.session_state[applied_key] = drag_id
            # component reports display-size coordinates -- map back to image
            # pixels (same conversion as the few-shot reference click above)
            orig_h, orig_w = image.shape[:2]
            scale_x = orig_w / drag["width"] if drag["width"] > 0 else 1.0
            scale_y = orig_h / drag["height"] if drag["height"] > 0 else 1.0
            length = float(np.hypot((drag["x2"] - drag["x1"]) * scale_x, (drag["y2"] - drag["y1"]) * scale_y))
            if length >= 2:
                st.session_state[f"{key}_bar_px"] = round(length, 1)
                # Be upfront about the tool's real precision: one screen pixel
                # of hand jitter is `scale_x` image pixels, so a narrow column
                # (small display scale) makes a short bar a rough measurement.
                st.session_state[info_key] = (
                    f"本次拖拽测得 {length:.1f} px。图像当前按 {1 / scale_x:.2f} 倍显示，"
                    f"鼠标精度约 ±{scale_x:.1f} px；标尺较短或显示较小时误差明显，可手动修正下方数值。"
                )
    if info_key in st.session_state:
        st.caption(st.session_state[info_key])


def scale_inputs_ui(container, key: str, image: np.ndarray | None = None) -> Scale:
    """Calibration widgets -> a Scale. Used twice: the sidebar default (no
    image, so only the options that don't need one) and each image's own
    override (image given: adds conversion from the field width, which needs
    the image's pixel width, and the drag-to-measure tool).

    Invalid input never silently becomes a scale: a zero/negative/blank value
    shows a warning and falls back to UNCALIBRATED (pixel units). The unit is
    in every downstream column name (see core/scale.py), so a pixel result
    can no longer be mistaken for a physical size."""
    options = ["未标定", "直接输入 µm/px", "按标尺长度换算"]
    if image is not None:
        options.append("按视野宽度换算")
    choice = container.radio("尺度", options, key=f"{key}_scale_mode")
    try:
        if choice == options[1]:
            value = container.number_input(
                "每像素对应的实际长度 (µm/px)", min_value=0.0, step=0.001, format="%.6f", key=f"{key}_um_per_px"
            )
            return scale_from_um_per_px(value)
        if choice == options[2]:
            if image is not None:
                _bar_drag_tool(key, image)
            st.session_state.setdefault(f"{key}_bar_px", 0.0)
            px = container.number_input("标尺的像素长度 (px)", min_value=0.0, step=1.0, format="%.1f", key=f"{key}_bar_px")
            real = container.number_input("标尺的实际长度", min_value=0.0, step=1.0, format="%.3f", key=f"{key}_bar_real")
            unit = container.selectbox("标尺长度的单位", LENGTH_UNITS, index=0, key=f"{key}_bar_unit")
            return scale_from_known_length(px, real, unit)
        if image is not None and choice == options[3]:
            container.caption("仅适用于未裁剪、未缩放的原始整幅图（SEM 信息栏通常会标注，例如 Width = 7.520 µm）。")
            width = container.number_input("视野宽度", min_value=0.0, step=1.0, format="%.3f", key=f"{key}_field_w")
            unit = container.selectbox("视野宽度的单位", LENGTH_UNITS, index=1, key=f"{key}_field_unit")
            return scale_from_field_width(width, unit, image.shape[1])
    except ValueError as exc:
        container.warning(f"{exc}。暂按未标定处理。")
    return UNCALIBRATED


st.sidebar.divider()
st.sidebar.subheader("尺度标定")
st.sidebar.caption(
    "默认尺度应用到本页所有图像；放大倍数不同的图像，请在各自测量区的「此图尺度」里单独设置。"
    "未标定时所有尺寸以像素为单位，不会冒充物理尺寸。"
)
default_scale = scale_inputs_ui(st.sidebar, "default")

PALETTE = [(255, 80, 80), (80, 200, 255), (120, 255, 120), (255, 210, 80), (200, 120, 255)]


def overlay_masks(image: np.ndarray, masks: np.ndarray) -> np.ndarray:
    out = image.copy()
    for i, mask in enumerate(masks):
        color = np.array(PALETTE[i % len(PALETTE)])
        out[mask] = (0.5 * out[mask] + 0.5 * color).astype(np.uint8)
    return out


def cached_detect(key_prefix: str, base_signature: tuple, compute_fn, stale_token=None):
    """Streamlit reruns the ENTIRE script on every widget interaction --
    including every single brush stroke in mask_correction_ui()'s canvas.
    Without this cache, that means re-running detection (seconds to tens of
    seconds) on every stroke, which looks like the page freezing (this is
    exactly what happened before this was added -- see README.md). Cache
    the result per file+settings in session_state; only recompute when
    `base_signature` (preprocessing toggles, detector identity, text
    prompt, ...) actually changed -- these are small, single-file-scope
    changes the user just made and expects reflected immediately.

    `stale_token` is a SEPARATE, softer invalidation axis (right now:
    few_shot_ref_generation) -- when it changes but `base_signature`
    hasn't, this does NOT eagerly recompute. It exists because adding a
    reference (see add_correction_as_reference) bumps it for every OTHER
    target image in the same batch at once; eagerly recomputing all of
    them would block the whole page behind a full synchronous re-detection
    of images the user isn't even looking at -- for exhaustive mode with
    several targets that's a multi-minute wait before the page responds
    again, reported as "second image never finishes loading" (see
    README.md). Returns (result, is_stale, refresh) instead: the caller
    shows the old (still-valid) result plus an explicit "重新检测" button,
    and only pays the recompute cost for images someone actually asks to
    refresh. `refresh` is a zero-arg callable that recomputes and updates
    the cache; call it only from inside that button's handler."""
    cache_key = f"{key_prefix}_detect_cache"
    full_signature = (*base_signature, stale_token)

    def refresh():
        result = compute_fn()
        st.session_state[cache_key] = (full_signature, result)
        return result

    cached = st.session_state.get(cache_key)
    if cached is None:
        return refresh(), False, refresh

    old_signature, result = cached
    if old_signature[:-1] != base_signature:
        return refresh(), False, refresh
    is_stale = stale_token is not None and old_signature[-1] != stale_token
    return result, is_stale, refresh


def mask_correction_ui(
    image: np.ndarray,
    masks: np.ndarray,
    key_prefix: str,
    sam_segmenter: SamSegmenter,
    on_add_reference=None,
    preserve_groups: bool = False,
    keep_instances: bool = False,
) -> tuple[np.ndarray, int]:
    """M5: optional manual paint/erase correction before measurement, off by
    default. "增加" doesn't use the user's freehand stroke pixels directly --
    tracing a precise particle outline by hand is slow and imprecise, and
    the stroke doesn't snap to the real edge. Instead the stroke's bounding
    box is used as a SAM box prompt (SamSegmenter.decode_box()) and SAM's
    own segmentation of that region is what gets added -- the user only
    needs to roughly cover the target, not outline it. This is also the
    fix for "one detection spans two touching particles" (see README.md):
    delete that instance, then roughly box each real particle separately
    (leave a gap between the two boxes) and let SAM find each one's real
    boundary. "擦除" stays literal pixel subtraction -- removing is exactly
    "these pixels, no more, no less", there's no boundary to snap to.

    Connected components re-split the combined result into per-instance
    masks once editing is done. On top of that, "实例操作" lets the user
    delete or merge specific instances by selecting them from the current
    list.

    `on_add_reference`, if given, is a callback(mask) that turns one selected
    instance into a new few-shot reference (see the "少样本" call sites) --
    only meaningful in few-shot mode, so zero-shot's call site leaves it
    None and the button simply doesn't render there.

    Returns (masks, n_pinned): n_pinned is how many of the leading rows in
    `masks` are user-merged groups (possibly non-touching pixels) rather
    than plain connected components -- callers must pass merge_disjoint=True
    for exactly those rows to compute_particle_properties, or its own
    connected-component call will silently re-split them.

    `preserve_groups=True` treats every incoming mask as an already-merged
    group, for detectors whose post-processing deliberately unions NEARBY
    but not touching pieces (the crack model's merge step): connected-
    component re-splitting would otherwise break those groups right back
    apart. With it set, the editing-off return also reports every row as a
    merged group, so run_measurement_ui measures each as one region.

    `keep_instances=True` is the particle counterpart: each incoming mask
    stays its own instance through editing even where neighbours touch
    (dense particle images). Without it, enabling the editor unions all
    masks and re-splits by connected components, which fuses touching
    particles -- 51 detected instances became 9 on a real BAM_TiO2 image.
    Rows are still measured as ordinary connected regions; only the user's
    own merges are measured whole."""
    enabled = st.checkbox("手动修正检测结果", value=False, key=f"{key_prefix}_edit_on")
    if not enabled:
        return masks, (masks.shape[0] if preserve_groups else 0)

    # `masks` is the exact object cached_detect() returned: same object
    # (same id()) across reruns triggered by canvas edits, a NEW object
    # whenever detection actually re-ran (preprocessing/reference changed).
    # Reset the edit state in that second case, or a stale edited mask from
    # a previous detection result would silently stick around.
    source_key = f"{key_prefix}_edit_source"
    state_key = f"{key_prefix}_edited_mask"
    pinned_key = f"{key_prefix}_pinned_groups"
    if state_key not in st.session_state or st.session_state.get(source_key) != id(masks):
        st.session_state[state_key] = masks_to_combined(masks, image.shape[:2])
        st.session_state[source_key] = id(masks)
        st.session_state[pinned_key] = []  # the user's own merges only; detector masks are passed in below
    st.session_state.setdefault(pinned_key, [])

    edit_mode = st.radio("画笔模式", ["增加", "擦除"], horizontal=True, key=f"{key_prefix}_edit_mode")

    orig_h, orig_w = image.shape[:2]
    disp_w = min(700, orig_w)
    disp_h = max(1, int(orig_h * disp_w / orig_w))

    # Latest user merge first, so a merge of already-merged groups claims
    # their pixels before they do; the crack model's own groups come after
    # every user merge for the same reason.
    originals = list(masks)
    current_masks, n_pinned = apply_pinned_groups(
        st.session_state[state_key],
        list(reversed(st.session_state[pinned_key])) + (originals if preserve_groups else []),
        originals if keep_instances and not preserve_groups else (),
    )
    preview = overlay_masks(image, current_masks) if current_masks.shape[0] else image
    background = Image.fromarray(preview).resize((disp_w, disp_h))

    gen_key = f"{key_prefix}_canvas_gen"
    st.session_state.setdefault(gen_key, 0)

    stroke_color = "rgba(0, 220, 0, 0.6)" if edit_mode == "增加" else "rgba(255, 0, 0, 0.6)"
    if edit_mode == "增加":
        st.caption(
            "大致框住/涂抹要新增的目标即可，不需要精细描边——应用后 SAM 会在这个范围内自动定位真实边界。"
            "如果一个检测其实是两颗粘连的颗粒，先在下方「实例操作」删除它，再分别框住两颗颗粒（两笔之间留出空隙）。"
        )
    else:
        st.caption("精确涂抹要擦除的区域，应用后按涂抹的像素原样去掉（可以反复涂抹多笔）。")
    canvas_result = st_canvas(
        fill_color=stroke_color,
        stroke_color=stroke_color,
        stroke_width=12,
        background_image=background,
        height=disp_h,
        width=disp_w,
        drawing_mode="freedraw",
        return_image_data=True,
        key=f"canvas_{key_prefix}_{st.session_state[gen_key]}",
    )

    col_a, col_b = st.columns(2)
    if col_a.button("应用这一笔", key=f"{key_prefix}_apply"):
        stroke_mask = canvas_layer_to_mask(canvas_result.image_data, image.shape[:2])
        if edit_mode == "增加":
            box = stroke_bounding_box(stroke_mask)
            if box is not None:
                with st.spinner("定位边界中…"):
                    sam_segmenter.set_image(image)
                    detected_mask = sam_segmenter.decode_box(box)
                # SAM occasionally returns nothing usable for a degenerate
                # box (e.g. a single dot) -- fall back to the raw stroke
                # rather than silently discarding what the user just drew.
                add_mask = detected_mask if detected_mask.any() else stroke_mask
                st.session_state[state_key] = st.session_state[state_key] | add_mask
        else:
            st.session_state[state_key] = st.session_state[state_key] & ~stroke_mask
        st.session_state[gen_key] += 1
        st.rerun()
    if col_b.button("重置为原始检测结果", key=f"{key_prefix}_edit_reset"):
        st.session_state[state_key] = masks_to_combined(masks, image.shape[:2])
        st.session_state[pinned_key] = []
        st.session_state[gen_key] += 1
        st.rerun()

    if current_masks.shape[0]:
        st.divider()
        st.caption("实例操作：按住 Ctrl/Cmd 多选，可以删除误检实例，或把断裂的同一目标合并成一个（比如裂纹）。")
        options = list(range(current_masks.shape[0]))

        def _label(i: int) -> str:
            tag = "（已合并）" if i < n_pinned else ""
            return f"实例 {i + 1}{tag} · {int(current_masks[i].sum())}px"

        selected = st.multiselect(
            "选择实例", options, format_func=_label, key=f"{key_prefix}_instance_select"
        )
        n_cols = 3 if on_add_reference is not None else 2
        cols = st.columns(n_cols)
        if cols[0].button("删除选中实例", key=f"{key_prefix}_delete_instances", disabled=not selected):
            union = np.zeros(image.shape[:2], dtype=bool)
            for i in selected:
                union |= current_masks[i]
            st.session_state[state_key] = st.session_state[state_key] & ~union
            st.rerun()
        if cols[1].button(
            "合并选中实例", key=f"{key_prefix}_merge_instances", disabled=len(selected) < 2
        ):
            union = np.zeros(image.shape[:2], dtype=bool)
            for i in selected:
                union |= current_masks[i]
            st.session_state[pinned_key].append(union)
            st.rerun()
        if on_add_reference is not None:
            if cols[2].button(
                "加入参考图集",
                key=f"{key_prefix}_add_reference",
                disabled=len(selected) != 1,
                help="把选中的这一个实例作为新的参考样例，本次会话里之后再处理的目标图片会自动用上它。",
            ):
                on_add_reference(current_masks[selected[0]])
                st.rerun()

    return current_masks, n_pinned


def render_stale_notice(is_stale: bool, key: str) -> bool:
    """Shown next to a target image whose cached detection predates a
    reference added elsewhere (see cached_detect's stale_token). Returns
    whether "重新检测" was clicked this render -- the caller is the one that
    knows how to unpack its own compute_fn's return shape, so it calls
    refresh() itself rather than this function doing it generically."""
    if not is_stale:
        return False
    st.info("参考图集自上次检测后已更新，下面暂时仍是旧结果。")
    return st.button("用最新参考图集重新检测", key=f"{key}_refresh")


def add_correction_as_reference(detector: FewShotDetector, image: np.ndarray, mask: np.ndarray, key_prefix: str) -> None:
    """Backs mask_correction_ui's "加入参考图集" button: turns one manually
    confirmed instance into a new few-shot reference, so images processed
    later in the same session (further down the same target_files loop, or
    on the next rerun for ones already rendered above this one) pick it up
    automatically. Bumping few_shot_ref_generation is what makes that
    happen -- every other target image's cached_detect signature includes
    it, so this invalidates their stale cache and forces a redetect with
    the enriched reference set.

    Immediately re-stamping THIS file's own cache entry with the new
    generation is required, not optional: without it, the bump we just
    caused would also invalidate this file's own cache on the very next
    rerun, re-running its detection from scratch and wiping out whatever
    manual correction the user just made on it (mask_correction_ui resets
    its edit state whenever the underlying detection result object
    changes -- see its own id()-based check)."""
    detector.set_reference(image, mask)
    st.session_state.few_shot_ref_generation += 1

    cache_key = f"{key_prefix}_detect_cache"
    cached = st.session_state.get(cache_key)
    if cached is not None:
        old_signature, result = cached
        new_signature = (*old_signature[:-1], st.session_state.few_shot_ref_generation)
        st.session_state[cache_key] = (new_signature, result)

    st.toast(f"已把该实例加入参考图集，当前共 {detector.n_references} 张参考图。")


def run_measurement_ui(
    image: np.ndarray, masks: np.ndarray, source_name: str, mode_label: str, n_merged: int = 0, kind: str = "particle"
):
    """`kind` picks the measurement: "particle" (equivalent diameter etc.) or
    "crack" (skeleton length + mean width, see core/measurement.py)."""
    if masks.shape[0] == 0:
        st.warning("没有检测到目标")
        return
    overlay = overlay_masks(image, masks)
    st.image(overlay, caption="分割结果", width="stretch")

    # Scale is per image: batches can mix magnifications, and one default
    # applied silently to all of them would be wrong for some -- exactly the
    # kind of unit error calibration exists to remove. Default = sidebar.
    scale_key = f"scale_{source_name}_{mode_label}"
    with st.expander("此图尺度", expanded=False):
        if st.checkbox("此图单独设置尺度", key=f"{scale_key}_override"):
            scale = scale_inputs_ui(st, scale_key, image)
        else:
            scale = default_scale
    st.caption(f"当前尺度：{scale.source}")

    compute_properties = compute_crack_properties if kind == "crack" else compute_particle_properties
    all_props = []
    for i, mask in enumerate(masks):
        all_props.extend(compute_properties(mask, pixel_size_um=scale.um_per_px, merge_disjoint=i < n_merged))
    if not all_props:
        st.warning("检测到目标区域，但未能从中提取有效轮廓（可能目标过小）")
        return
    # per-mask measurement numbers every mask's regions from 1, so the raw
    # label column repeats (1, 1, 1, ...) -- renumber across the whole image
    for n, props in enumerate(all_props, start=1):
        props["label"] = n

    df = summarize(all_props)
    st.dataframe(df, width="stretch")

    unit = scale.length_unit
    stats = measure_stats(df, scale, kind)
    cols = st.columns(4)
    if kind == "crack":
        length, width = stats["length"], stats["mean_width"]
        cols[0].metric("数量", length["count"])
        cols[1].metric(f"平均长度 ({unit})", format_value(length["mean"]))
        cols[2].metric(f"长度范围 ({unit})", f"{format_value(length['min'])} ~ {format_value(length['max'])}")
        cols[3].metric(f"平均宽度 ({unit})", format_value(width["mean"]))
        st.caption(
            "长度 = 裂纹骨架（中轴线）的长度，沿裂纹走向、含分支，由多段合并成的一条裂纹只计各段本身、不含段间空隙；"
            "平均宽度 = 面积 ÷ 长度。骨架提取对mask边缘的毛刺敏感，形状很不规则时长度会略偏大。"
        )
    else:
        diameter = stats["equivalent_diameter"]
        cols[0].metric("数量", diameter["count"])
        cols[1].metric(f"平均等效直径 ({unit})", format_value(diameter["mean"]))
        cols[2].metric(f"标准差 ({unit})", format_value(diameter["std"]))
        cols[3].metric(f"范围 ({unit})", f"{format_value(diameter['min'])} ~ {format_value(diameter['max'])}")
    figures = distribution_figures(df, scale, kind)
    for (_, fig), col in zip(figures, st.columns(len(figures))):
        col.plotly_chart(fig, width="stretch")

    export_buttons(image, overlay, df, stats, figures, source_name, mode_label, scale, kind)


def export_buttons(image, overlay, df, stats, figures, source_name: str, mode_label: str, scale: Scale, kind: str):
    stem = source_name.rsplit(".", 1)[0]
    col_a, col_b, col_c = st.columns(3)

    overlay_buf = io.BytesIO()
    Image.fromarray(overlay).save(overlay_buf, format="PNG")
    col_a.download_button("下载效果图", overlay_buf.getvalue(), file_name=f"{stem}_annotated.png", mime="image/png", key=f"dl_img_{stem}_{mode_label}")

    # The unit is already in the measurement column names; the scale value and
    # where it came from travel with the CSV too, so the file stays
    # interpretable on its own (empty scale_um_per_px = pixel units).
    export_df = df.assign(scale_um_per_px=scale.um_per_px if scale.calibrated else float("nan"), scale_source=scale.source)
    col_b.download_button("下载CSV数据表", export_df.to_csv(index=False).encode("utf-8-sig"), file_name=f"{stem}_data.csv", mime="text/csv", key=f"dl_csv_{stem}_{mode_label}")

    if col_c.button("生成实验报告", key=f"gen_report_{stem}_{mode_label}"):
        with st.spinner("生成报告中…"):
            report_path = config.OUTPUTS_DIR / f"{stem}_report.docx"
            generate_report(str(report_path), source_name, mode_label, overlay, df, stats, figures, scale, kind)
        with open(report_path, "rb") as f:
            st.download_button(
                "下载实验报告 (docx)",
                f.read(),
                file_name=f"{stem}_report.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key=f"dl_report_{stem}_{mode_label}",
            )


if mode.startswith("少样本"):
    st.subheader("第一步：上传参考样例并点击目标结构")
    st.caption("建议上传3张参考图以获得更好精度（验证中3张参考图的IoU比1张提升了一倍以上）")
    ref_files = st.file_uploader(
        "参考样例", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True, key="ref_uploader"
    )

    # Lazy-init once, rather than resetting on every change to the uploaded
    # file list: `accept_multiple_files=True` lets the user add files
    # incrementally, and treating "the list changed" as "start over" wiped
    # out already-confirmed references (and their baked-in embeddings) the
    # moment a 3rd/4th file was added.
    if "few_shot_detector" not in st.session_state:
        st.session_state.few_shot_detector = FewShotDetector(sam)
        st.session_state.ref_confirmed = {}
        st.session_state.few_shot_ref_generation = 0

    if ref_files:
        detector = st.session_state.few_shot_detector

        # FewShotDetector has no way to "un-bake" a reference from its
        # running-average embedding, so if a file that was already confirmed
        # disappears from the uploader (user removed it), the only correct
        # option is a full reset -- but only in that specific case, not on
        # every list change.
        current_names = {f.name for f in ref_files}
        removed = set(st.session_state.ref_confirmed) - current_names
        if removed:
            st.warning(f"已确认的参考图被移除（{', '.join(removed)}），无法从已有特征中撤回，已重置全部参考图，请重新确认。")
            st.session_state.few_shot_detector = FewShotDetector(sam)
            st.session_state.ref_confirmed = {}
            st.session_state.few_shot_ref_generation = 0
            detector = st.session_state.few_shot_detector

        if st.button("清空全部参考图，重新开始"):
            st.session_state.few_shot_detector = FewShotDetector(sam)
            st.session_state.ref_confirmed = {}
            st.session_state.few_shot_ref_generation = 0
            st.rerun()
        for i, file in enumerate(ref_files):
            # Must match the target image's preprocessing exactly (same
            # denoise/equalize settings) -- otherwise the reference embedding
            # and target embedding come from different feature domains
            # (CLAHE in particular reshapes local contrast, which a ViT
            # encoder is sensitive to) and matching quality degrades.
            try:
                ref_image = cached_preprocess(file.getvalue(), denoise_on, equalize_on, contrast_on)
            except ImageLoadError as exc:
                st.error(f"参考图 {file.name}：{exc}")
                continue
            confirmed = st.session_state.ref_confirmed.get(file.name, False)
            st.write(f"参考图 {i + 1}: {file.name}" + ("（已确认）" if confirmed else " — 点击目标结构"))

            coords = streamlit_image_coordinates(ref_image, key=f"ref_click_{file.name}")
            if coords and not confirmed:
                # 组件返回的是"显示坐标系"坐标 + 显示尺寸（width/height）。
                # 大图在页面里会被缩放显示，必须按比例换算回原图像素，
                # 否则点击位置错位、参考 mask 选错区域。
                disp_w, disp_h = int(coords["width"]), int(coords["height"])
                orig_h, orig_w = ref_image.shape[:2]
                scale_x = orig_w / disp_w if disp_w > 0 else 1.0
                scale_y = orig_h / disp_h if disp_h > 0 else 1.0
                px_x = min(max(int(round(coords["x"] * scale_x)), 0), orig_w - 1)
                px_y = min(max(int(round(coords["y"] * scale_y)), 0), orig_h - 1)
                point_prompt = DetectionPrompt(
                    points=np.array([[px_x, px_y]], dtype=float),
                    point_labels=np.array([1], dtype=int),
                )
                ref_masks = sam.segment(ref_image, point_prompt)
                if ref_masks.shape[0] > 0:
                    col_a, col_b = st.columns([3, 1])
                    col_a.image(overlay_masks(ref_image, ref_masks), caption="点击处的分割预览", width=300)
                    prior_mean_area = detector.reference_mask_area if detector.n_references > 0 else None
                    selection_check = check_reference_selection(ref_masks[0], ref_image.shape, prior_mean_area)
                    for w in selection_check.warnings:
                        st.warning(w)
                    if col_b.button("确认此参考图", key=f"confirm_{file.name}"):
                        detector.set_reference(ref_image, ref_masks[0])
                        st.session_state.few_shot_ref_generation += 1
                        st.session_state.ref_confirmed[file.name] = True
                        st.rerun()

        st.divider()
        st.subheader("第二步：上传目标图像批量检测")

        exhaustive_mode = False
        if particle_classifier is not None:
            exhaustive_mode = st.checkbox(
                "穷举检测——找出图中所有相似目标，而非单个最佳匹配",
                value=False,
                help="适合密集颗粒场景（一张图几十上百颗同类颗粒）。13图留出评估（严格一对一匹配，"
                "IoU≥0.5）：召回率约25%、精确率约74%（仍有约四分之三真实目标漏检，详见 "
                "README.md「训练轻量分类头」）。命中的检测本身质量较高（平均IoU 0.83），"
                "问题主要在漏检而非分割不准。默认的单目标模式精度更高但只返回一个检测结果，"
                "不适合数一张图里的所有颗粒。",
            )
        else:
            st.caption(
                "穷举检测尚未启用——未找到已训练的分类器权重（`models/particle_classifier.npz`），"
                "需先运行 `scripts/build_particle_classifier_dataset.py` + `scripts/train_particle_classifier.py`。"
            )

        candidate_source = "分类器峰值"
        similarity_threshold = 0.78
        if exhaustive_mode and instance_segmenter is not None:
            candidate_source = st.radio(
                "候选来源",
                ["分类器峰值", "专用模型 + 参考图打分"],
                horizontal=True,
                help="比赛分支专用：候选直接来自训练过的实例分割模型（专用模型模式同一个），"
                "再用参考图相似度过滤掉不是目标品类的候选。跟主分支的「打分展示，用户自己筛」"
                "设计不同——这里是硬阈值自动过滤，图快，但没有主分支那么保守（验证发现参考图"
                "相似度不能完全区分品类，硬阈值会有误杀/漏筛，见 README.md）。",
            )
            if candidate_source.startswith("专用模型"):
                similarity_threshold = st.slider(
                    "参考图相似度过滤阈值", 0.0, 1.0, 0.78, 0.01,
                    help="低于这个分数的候选会被当成「不是参考图那个品类」过滤掉。0.78是"
                    "在EMPS不同源图像上验证出的经验起点，没有针对这条完整流水线专门标定过。",
                )

        target_files = st.file_uploader(
            "目标显微图像", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True, key="target_uploader"
        )
        if target_files:
            if detector.n_references == 0:
                st.warning("请先在上方点击并确认至少1张参考图")
            else:
                st.caption(f"已确认 {detector.n_references} 张参考图")
                for file in target_files:
                    st.subheader(file.name)
                    try:
                        processed = cached_preprocess(file.getvalue(), denoise_on, equalize_on, contrast_on)
                    except ImageLoadError as exc:
                        st.error(f"{file.name}：{exc}")
                        continue

                    if exhaustive_mode:
                        if candidate_source.startswith("专用模型"):
                            exhaustive_key = f"exhaustive_instanceseg_{file.name}"

                            def _compute_exhaustive():
                                with st.spinner("专用模型检测中…"):
                                    raw_image = load_rgb(file.getvalue())
                                    candidates = instance_segmenter.predict(raw_image)
                                    if candidates.shape[0] == 0:
                                        return candidates
                                    scores = detector.score_masks(processed, candidates)
                                return candidates[scores >= similarity_threshold]

                            signature = (denoise_on, equalize_on, contrast_on, id(detector), "instance_seg", similarity_threshold)
                        else:
                            exhaustive_key = f"exhaustive_{file.name}"

                            def _compute_exhaustive():
                                with st.spinner("穷举检测中（比单目标模式慢，需扫描全图候选区域）…"):
                                    kept = detector.detect_all(processed, particle_classifier)
                                return np.stack([m for _, m in kept]) if kept else np.empty((0, *processed.shape[:2]), dtype=bool)

                            signature = (denoise_on, equalize_on, contrast_on, id(detector))

                        masks, is_stale, refresh = cached_detect(
                            exhaustive_key,
                            signature,
                            _compute_exhaustive,
                            stale_token=st.session_state.few_shot_ref_generation,
                        )
                        if render_stale_notice(is_stale, exhaustive_key):
                            masks = refresh()
                        st.caption(f"找到 {masks.shape[0]} 个候选目标（实验性功能，可能有漏检/误检，建议人工核对）")
                        masks, n_merged = mask_correction_ui(
                            processed,
                            masks,
                            key_prefix=exhaustive_key,
                            sam_segmenter=sam,
                            on_add_reference=lambda mask: add_correction_as_reference(detector, processed, mask, exhaustive_key),
                            keep_instances=True,
                        )
                        run_measurement_ui(processed, masks, file.name, "少样本-穷举", n_merged)
                        continue

                    fewshot_key = f"fewshot_{file.name}"

                    def _compute_single():
                        with st.spinner("检测中…"):
                            prompt = detector.detect(processed)
                            masks = sam.segment(processed, prompt)
                        return prompt, masks

                    (prompt, masks), is_stale, refresh = cached_detect(
                        fewshot_key,
                        (denoise_on, equalize_on, contrast_on, id(detector)),
                        _compute_single,
                        stale_token=st.session_state.few_shot_ref_generation,
                    )
                    if render_stale_notice(is_stale, fewshot_key):
                        prompt, masks = refresh()
                    if len(prompt.scores) > 0:
                        st.caption(f"相似度得分: {round(float(prompt.scores[0]), 3)}")

                    if masks.shape[0] > 0:
                        reliability = check_few_shot_result(masks[0], detector.reference_mask_area)
                        if not reliability.is_reliable:
                            st.error(
                                "⚠️ 结果可能不可靠——相似度分数不能保证这次匹配是对的："
                                + "；".join(reliability.reasons)
                                + "。建议人工核对分割结果，或换一张更有代表性的参考图重试。"
                            )
                    masks, n_merged = mask_correction_ui(
                        processed,
                        masks,
                        key_prefix=fewshot_key,
                        sam_segmenter=sam,
                        on_add_reference=lambda mask: add_correction_as_reference(detector, processed, mask, fewshot_key),
                    )
                    run_measurement_ui(processed, masks, file.name, "少样本", n_merged)
    else:
        st.caption("上传参考样例开始。")

else:
    target_type = st.radio("目标类型", ["颗粒", "裂纹"], horizontal=True, key="instanceseg_target_type")
    if target_type == "颗粒":
        type_tag, mode_label, segmenter = "particle", "专用模型-颗粒", instance_segmenter
        weights_name, train_script = "instance_segmenter.pth", "scripts/train_instance_segmenter.py"
        preserve_groups = False
        keep_instances = True
        model_caption = (
            "在 EMPS 显微颗粒数据集上训练的实例分割模型（Mask R-CNN），不需要参考图或文字提示，直接找出图中所有实例。"
            "留出评估（严格一对一匹配，IoU≥0.5）：同域（EMPS）召回率88.8%/精确率91.7%（检测上限300）；域外（BAM_TiO2颗粒图像）"
            "召回率40.7%/精确率78.4%——训练数据只有EMPS，材料/成像方式差异较大的图像效果可能明显打折，建议先用少量"
            "图像试跑核对，不放心可配合下方手动修正（详见 README.md「训练专门的实例分割模型」）。"
            "此模式使用未经预处理的原始图像（训练时同样未做去噪/均衡化/对比度增强），边栏的预处理开关在此模式下不生效。"
        )
    else:
        type_tag, mode_label, segmenter = "crack", "专用模型-裂纹", load_crack_segmenter()
        weights_name, train_script = "crack_segmenter.pth", "scripts/train_crack_segmenter.py"
        # The crack model's merge step unions nearby-but-not-touching pieces
        # of one crack; connected-component splitting in M5 and in
        # compute_particle_properties would break those groups back apart.
        preserve_groups = True
        keep_instances = False
        model_caption = (
            "在 DenseCrack（显微尺度密集微裂纹，光学/荧光显微）上训练的实例分割模型（Mask R-CNN，针对细长形状加宽了锚框比例），"
            "不需要参考图或文字提示。DenseCrack 测试集（100张，严格一对一匹配，IoU≥0.5）：召回率约72%~75%/精确率约71%~73%（合并半径在另一半图上选定的留出结果为72.2%/71.4%，同集选择为74.8%/73.0%）。"
            "这个精确率包含一步后处理：同一条裂纹常被拆成沿长度方向的几段局部检测，已按10px邻近半径合并成一个实例"
            "（合并前精确率仅42.3%；评估时真值也按同一半径合并，口径对齐，真值本身来自二值mask连通域而非逐实例标注）。"
            "训练数据只有 DenseCrack：其它材料、成像方式，尤其宏观照片（路面、墙面等），效果会明显下降"
            "（宏观裂纹集 CrackSeg9k 上召回率仅3.1%），建议先用少量图像试跑核对。"
            "此模式使用未经预处理的原始图像，边栏的预处理开关在此模式下不生效。"
        )

    if segmenter is None:
        st.caption(
            f"专用模型（{target_type}）尚未启用——未找到已训练的权重（`models/{weights_name}`），"
            f"需先运行 `{train_script}`。"
        )
    else:
        st.caption(model_caption)
        target_files = st.file_uploader(
            "上传显微图像", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True, key="instanceseg_uploader"
        )
        for file in target_files or []:
            st.subheader(file.name)
            try:
                image = load_rgb(file.getvalue())
            except ImageLoadError as exc:
                st.error(f"{file.name}：{exc}")
                continue
            key_prefix = f"instanceseg_{type_tag}_{file.name}"
            if needs_tiling(*image.shape[:2]):
                st.info(
                    f"图像较大（{image.shape[1]}×{image.shape[0]}）：模型内部会把图像缩小到短边800像素，小目标会成片漏检"
                    "（实测4096×3072的图检出0个），所以这张图按800像素分块检测、再拼接结果。分块边界附近超过约160像素的目标可能被截断。"
                )

            def _compute():
                with st.spinner("检测中…"):
                    return segmenter.predict(image)

            try:
                masks, _, _ = cached_detect(key_prefix, ("instance_seg", type_tag), _compute)
            except ValueError as exc:  # tiled result too large to hold as per-instance masks
                st.error(f"{file.name}：{exc}")
                continue
            st.caption(f"找到 {masks.shape[0]} 个实例")
            masks, n_merged = mask_correction_ui(
                image, masks, key_prefix=key_prefix, sam_segmenter=sam,
                preserve_groups=preserve_groups, keep_instances=keep_instances,
            )
            run_measurement_ui(image, masks, file.name, mode_label, n_merged, kind="crack" if type_tag == "crack" else "particle")
