# 显微图像智能分割与定量测量工作台：运行说明

本地运行的 Streamlit 应用：上传显微图像 → AI 识别（颗粒/裂纹专用实例分割，或少样本参考匹配）→ 人工修正 → 尺度标定与形态测量 → 导出标注图、CSV 和 Word 实验报告。图像与推理都在本机完成，不上传数据。

## 1. 技术栈

Python 3.11 · Streamlit · PyTorch / torchvision（Mask R-CNN ResNet50-FPN v2）· segment-anything（SAM ViT-B）· OpenCV · scikit-image · SciPy · Pandas · Plotly · python-docx。完整版本见 `requirements.txt`。

## 2. 运行环境

| 项目 | 已验证 | 未验证 |
|---|---|---|
| 系统 | macOS（Apple Silicon，M5 Pro，51.5 GB 内存） | Windows、Linux |
| Python | 3.11.16 | 其他版本 |
| 推理设备 | MPS（SAM 编码 0.33 秒/张）、CPU（1.30 秒/张） | CUDA |
| 主要依赖 | torch 2.14.0、torchvision 0.29.0、streamlit 1.63.0、numpy 1.26.4 | |

模型权重共约 0.75 GB（虚拟环境另计，PyTorch 等依赖需要数 GB）。CPU 也能运行，只是更慢。

## 3. 安装与启动

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

**把模型权重放进 `models/`**（`.pth` 文件不在源码包里，随提交材料单独提供，必须是真实文件而不是链接）：

| 文件 | 大小 | 用途 |
|---|---|---|
| `sam_vit_b_01ec64.pth` | 约 375 MB | SAM 图像编码（少样本模式、人工修正的"增加"） |
| `instance_segmenter.pth` | 约 184 MB | 颗粒专用模型 |
| `crack_segmenter.pth` | 约 184 MB | 裂纹专用模型 |

`particle_classifier.npz`（穷举检测用的轻量分类头）已在源码包的 `models/` 里。缺少 `instance_segmenter.pth` 或 `crack_segmenter.pth` 时，对应的专用模型会在界面上提示"尚未启用"；**缺少 `sam_vit_b_01ec64.pth` 则应用无法启动**。

启动（在项目根目录）：

```bash
python -m streamlit run app.py --server.address 127.0.0.1
```

浏览器打开 `http://localhost:8501`。首次启动要等几秒到十几秒加载模型。

> 报告里的统计图用 plotly 导出图片，`kaleido` 1.x 需要本机装有 Chrome/Chromium；没有时软件自动改用 matplotlib 重画（样式更朴素），报告照常生成。

## 4. 快速上手（用自带的演示图）

演示图在 `data/demo_images/`（每张图的标尺读数和预期结果见该目录的 `README.md`）。

1. 侧边栏选"专用模型"，目标类型选"颗粒"，上传 `data/demo_images/particles/emps_04dbda93f9.png`，应显示"找到 32 个实例"。
2. 展开"此图尺度"，勾选"此图单独设置尺度"，选"按标尺长度换算"，输入 67 px = 100 nm，表格列名随之变为 `*_um`；平均等效直径约 0.0653 µm。
3. 点"下载CSV数据表"或"生成实验报告"导出（CSV 带 `scale_um_per_px`、`scale_source` 两列）。
4. 切到目标类型"裂纹"，上传 `data/demo_images/cracks/densecrack_image_1320.png`，应检出 5 条，指标为长度与平均宽度（未标定时单位是 px）。

5. 少样本：侧边栏选"少样本"，上传 `data/demo_images/fewshot/examples/` 的两张示例图，各点击一个颗粒并点"确认此参考图"；勾选"穷举检测"，再上传 `data/demo_images/fewshot/targets/` 的五张识别图，每张会显示找到的候选目标数。这五张图是从测试图中挑出的效果较好的几张，少样本整体效果有限（详见该目录的 `README.md`）。

**注意**：页面会把已上传的所有图都按当前模式处理，每种模式只上传对应类型的图。

## 5. 目录说明

| 路径 | 内容 |
|---|---|
| `app.py`、`config.py` | 交互界面与路径配置 |
| `core/` | 检测器（`detectors/`）、测量（`measurement.py`）、尺度标定（`scale.py`）、统计与报告、人工修正、图像读取、分块检测 |
| `models/` | 权重存放处；已含穷举检测用的 `particle_classifier.npz` |
| `data/demo_images/` | 演示图（含来源、许可与预期结果说明） |
| `requirements.txt` | 运行依赖 |

## 6. 已知局限（详见作品方案）

- Windows 与 CUDA 未测试；干净环境按本说明完整重装未验证。
- 颗粒专用模型在域外数据（BAM_TiO2）上召回率只有 40.7%；裂纹模型只在 DenseCrack 上训练，对其他材料和宏观裂纹明显退化。
- 短边超过约 1066 像素或长边超过约 1777 像素的大图会按 800 像素分块检测，分块后的精度没有重新评估；目标非常多时逐实例掩膜占用内存大，超过 3 GB 会提示裁剪。
- 尺度标定依赖用户输入；用标尺换算时短标尺读数有 ±1 px 的误差，能读到视野宽度时优先使用视野宽度（仅适用于未裁剪、未缩放的整幅图）。
- 示例与评估数据遵循各自许可（EMPS：CC BY 4.0；DenseCrack：CC BY-NC-SA 4.0），使用或发布时请署名。
