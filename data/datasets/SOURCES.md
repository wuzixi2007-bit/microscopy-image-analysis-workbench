# 样本集来源与许可说明

本目录存放从公开渠道下载的显微图像样本集，按数据集名分目录组织，用于 M0（零样本）
与 M1（少样本）检测/分割验证。下载日期：2026-09-15（第二批 2026-09-16 补齐完成）。

## 目录总览

| 目录 | 内容 | 规模 | 标注 | 许可 | 来源 |
|---|---|---|---|---|---|
| `Aachen-Heerlen/` | 贝氏体钢 SEM 图 | 1705 张 PNG | ✅ 专家多边形标注（8909 个）+ POI + 元数据 | CC BY 4.0（figshare） | figshare collection c.5185004 |
| `DoITPoMS/` | 剑桥大学教学显微图像库（金属/陶瓷/聚合物等） | 783 张 JPEG | ❌ 无标注 | 教学用途，需署名 | doitpoms.ac.uk/miclib |
| `cigRockSEM/` | 岩石（泥岩/砂岩/页岩）SEM 微观结构 | 14954 图 + 14954 掩膜（9.8GB） | ✅ 像素级掩膜（imgs/masks） | CC BY 4.0 | Zenodo DOI 10.5281/zenodo.14988631 |
| `CrackSeg9k/` | 裂纹分割基准合集（Crack500/DeepCrack/SDNET/Gaps/Volker/Rissbilder/Noncrack/Masonry/**Ceramic** 等） | 9255 张（Vol1+Vol2） | ✅ 像素级裂纹标注 | 各子集许可不同（学术使用） | Harvard Dataverse DOI 10.7910/DVN/EGIEBY |
| `NIST_AM_Pores/` | 增材制造 lack-of-fusion 气孔（合成 + 训练） | 3503 个气孔，32³ 二值体积（tiff 栈） | ✅ 二值 GT + metadata 含等效球直径(µm) | 美国政府作品（公共领域） | NIST data.gov mds2-3963 |
| `MoNuSeg/` | 生物组织 H&E 病理切片（多器官核分割） | 训练 37 图 + 测试 14 图（各带 xml 标注） | ✅ 核边界多边形标注 | CC BY-NC-SA 4.0（**非商用**） | MoNuSeg grand-challenge / Google Drive |
| `DenseCrack/` | SHCC 应变硬化水泥基复合材料的致密微裂纹（显微尺度，非 SEM） | 510 图 + 510 掩膜（train 410 / test 100） | ✅ 像素级二值（0/255） | CC BY-NC-SA 4.0（**非商用**） | Kaggle: renshuddd/dense-micro-cracks |
| `BAM_TiO2/` | TiO2 纳米颗粒电子显微图（SEM + TSEM，颗粒密集） | 80 图 + 80 二值掩膜（**SEM-only 划分 train 32 / test 8**） | ✅ 像素级二值（4-连通人工标注） | CC BY-NC-ND 4.0（**非商用 + 禁演绎**） | GitHub: BAMresearch/automatic-sem-image-segmentation（Zenodo 4563942） |
| `EMPS/` | 文献挖掘的电子显微颗粒图（SEM/TEM，实例级） | 465 图 + 465 实例掩膜（train 366 / test 99） | ✅ 实例 ID 掩膜（每颗粒独立编号，共 11594 颗粒） | CC BY 4.0 | GitHub: by256/emps（DatasetNinja 镜像） |
| `MetalDAM/` | 增材制造钢 SEM 显微组织（5 类相语义分割） | 42 图 + 42 像素级标签（官方 train/test 划分） | ✅ 5 类语义分割（Matrix/Austenite/M-A/Precipitate/Defect） | MIT | GitHub: ari-dasci/OD-MetalDAM（Release 1.0） |

## 各数据集说明

### Aachen-Heerlen（P 补充：贝氏体钢）
- 文件：`images/IMG_*.png` 1705 张；`annotations/` 下为多边形标注（CSV/pickle）、POI、样品元数据、几何 pickle。
- 论文：Iren et al., Scientific Data 8:140 (2021), DOI: 10.1038/s41597-021-00926-7

### DoITPoMS（多样性无标注集）
- 文件：`images/{000001..000975}.jpg`（编号不连续，783 张；少量源文件为 BMP 已转码）。
- 来源：https://www.doitpoms.ac.uk/miclib/

### cigRockSEM（P1 气孔/孔隙率，真实 SEM + GT）
- 文件：`data/{mudstone,sandstone,shale}/{imgs,masks}/`，分别 2477 / 1152 / 11325 张，
  共 14954 图 + 14954 像素级掩膜（9.8GB）。
- 论文：cigRockSEM benchmark dataset（岩石 SEM 微观结构，孔隙/矿物相像素级掩膜）。
- 用途：真实 SEM 孔隙分割 GT，与 Aachen 同样可做定量对比。

### CrackSeg9k（P2 裂纹，多源合集）
- 文件：`images/`（9159 张）+ `masks/`（9159 张，一一对应）+ `train.txt`/`test.txt`（7332/1827 划分）。
  源子集：Crack500、Deepcrack、SDNET、Gaps、Volker、Rissbilder、Noncrack、Masonry、Ceramic 等。
- 论文：CrackSeg9k (arXiv 2208.13054)；数据：Harvard Dataverse 10.7910/DVN/EGIEBY。
- 说明：多为宏观/近距裂纹（路面/结构/陶瓷表面），非 SEM 显微尺度，用于裂纹分割鲁棒性测试。

### NIST_AM_Pores（P1 气孔 GT + 已知尺寸）
- 文件：`tiff_masks/synthetic_pore_*.tiff`（3503 个，32 切片栈，0/255 二值）、
  `_download/synthetic_pore_metadata.csv`（含 equivalent_spherical_diameter 与 diameter_range，µm）、
  `_download/readme.md`。
- 来源：NIST mds2-3963，DOI 10.18434/mds2-3963（合成 LOF 气孔，基于 CoCr XCT 训练）。
- 说明：单个气孔的 3D 二值形状，切片为 32×32；扫描分辨约 2.43–2.77 µm/voxel。

### MoNuSeg（P3 陌生材料：生物组织）
- 文件：`training/images/*.tif`（37 图）+ `training/annotations/*.xml`（核边界多边形）；
  `testing/images/*.tif`（14 图）+ `testing/annotations/*.xml`。
- 来源：MoNuSeg 2018（grand-challenge.org），H&E 染色 40x 病理切片（TCGA 多器官）。
- 用途：与钢/玻璃完全不同的图像域，做"真·未知类别少样本泛化"演示。
- 许可：CC BY-NC-SA 4.0（非商用，比赛/论文需注意）。

### DenseCrack（P2 裂纹补充：显微尺度微裂纹）
- 文件：`images/`（510 张）+ `masks/`（510 张，0/255 二值，一一对应）+ `train.txt`/`test.txt`（410/100 划分，与 CrackSeg9k 同布局）。
- 来源：Kaggle renshuddd/dense-micro-cracks（原 zip 顶层目录名 "DenseCrack A Segmentation Dataset of Dense Micro-C"）。
- 内容：应变硬化水泥基复合材料（SHCC）表面的**致密微裂纹**，512×512，像素级分割标注；
  裂纹像素占比约 2.6%–15.8%。
- 说明：显微尺度裂纹，但成像为光学/荧光显微而非 SEM，与 CrackSeg9k（宏观照片）互补，
  用于"细密微裂纹"分割鲁棒性测试；答辩展示时如实说明成像方式。
- 许可：CC BY-NC-SA 4.0（非商用，须署名；与 MoNuSeg 相同限制）。

### BAM_TiO2（NIST_CalibrationParticles 替代：密集颗粒 + 像素级 mask）
- 文件：`images/`（80 张）+ `masks/`（80 张二值 4-连通掩膜，一一对应）+ `train.txt`/`test.txt`（32/8）。
- 来源：BAM 联邦材料研究所（GitHub: BAMresearch/automatic-sem-image-segmentation；Zenodo DOI 10.5281/zenodo.4563942）。
- 论文：Ruehle et al., "Workflow towards Automated Segmentation of Agglomerated, Non-Spherical
  Particles from Electron Microscopy Images using Artificial Neural Networks", Scientific Reports (2021),
  DOI: 10.1038/s41598-021-84287-6。
- 内容：80 张中 **40 张 SEM（偶数编号）+ 40 张 TSEM（奇数编号）**；SEM 图与掩膜 768×1024 完美对应，
  **TSEM 掩膜被裁切（712×1024 vs 原图 768×1024），尺寸不匹配**。
- **使用注意（重要）**：`train.txt`/`test.txt` 为 **SEM-only 划分（32/8）**，已筛除 TSEM（奇数编号）；
  如需全量 80 张请自行按文件编号重建。比例尺烧录在图像内，无独立 pixel-size 元数据。
- **已修复：图像/掩膜尺寸不匹配**（`scripts/prepare_bam_tio2.py`）——TSEM（奇数编号）原图 768×1024，
  掩膜被裁到 712×1024（底部56px仪器信息条被annotator直接裁掉，SEM那批是同样的信息条但mask保留
  整图756×1024、信息条区域置零——两批标注处理方式不一致，不是数据损坏）。跑过脚本后
  `prepared/images/` 下的图像已裁剪到与 mask 对齐，两者可以直接配对使用。
- **少样本检测实测：对参考颗粒的选择敏感，不能随便选**——用真实数据测过：
  - 选稀疏图（22颗粒/图）+ 小颗粒（658px）当参考 → 复现了 NIST_CalibrationParticles 那种"匹配到背景"
    失败（预测面积占了82%的图像，跟真实颗粒完全不重合）
  - 选密集图（162颗粒/图）+ 最大连通区域当参考 → 定位到了正确颗粒（point_in_gt=True），但 IoU 只有
    0.076——TiO2 颗粒容易粘连成团簇，"面积最大的连通区域"大概率是好几个颗粒粘在一起而非单颗粒，
    团簇去匹配团簇形状对不齐
  - **建议**：选参考颗粒时用密度较高的图（162 那张比 22 那张的图安全得多），并且手动挑一个**孤立、不
    与相邻颗粒粘连**的中等大小颗粒，不要用"面积最大"或"随便一个"这种启发式选取
- 许可：**CC BY-NC-ND 4.0（非商用 + 禁止演绎 + 须署名）**——对外发布需谨慎。

### EMPS（NIST_CalibrationParticles 替代补充：实例级颗粒）
- 文件：`images/`（465 张）+ `masks/`（465 张实例掩膜）+ `train.txt`/`test.txt`（366/99，作者原始划分）。
- 来源：剑桥大学 Yildirim & Cole（GitHub: by256/emps；DatasetNinja 镜像，Supervisely 格式）。
- 论文：Yildirim & Cole, "Bayesian Particle Instance Segmentation for Electron Microscopy Image
  Quantification", J. Chem. Inf. Model. 61(3), 1136-1149 (2021), DOI: 10.1021/acs.jcim.0c01455。
- 内容：465 张 SEM/TEM 颗粒图（采自已发表文献，模态与分辨率异构），共 11594 个标注颗粒（均值约 25/图）。
  掩膜像素值为**实例 ID（0=背景，每个颗粒一个独立编号）**——实例级标注，适合 point-matching 逐目标对比；
  已知瑕疵：相互接触的颗粒会粘成单个连通域。
- 注意：无像素尺寸元数据，比例尺在图内且不统一，不能做物理标定，只做形态/分割验证。
- 许可：**CC BY 4.0**（宽松，可商用，须署名）。

### 已评估弃用（勿重新下载）
- **EM3M（HuggingFace: UniParser/EM3M）**：ECC V2026 论文配套数据集，MIT 许可，instance 级 mask 格式很好，
  但**混入大量 AI 合成/伪显微图**（论文自带扩散模型"合成增强引擎"；抽查 16 张约 4 张明显合成/抽象，
  caption 中 31/152 标 "abstract"），无法逐张可靠区分真实与合成，会污染定量测量结论，**已删除**。
  如需形态多样性参考可重下（HF 缓存仍在），但不得作为正式验证基准。
- **MetalDAM（GitHub: ari-dasci/OD-MetalDAM；HF 镜像 Voxel51/OD_MetalDAM 已下线）**：真实数据，MIT 许可，
  **已于 2026-09-17 重新下载恢复**（Release 1.0，labeled 42 张 + metadata SQLite）。
  注意其标注是**相分布式语义分割**（同一类别在图内散布几十个不连续区域，如 Martensite/Austenite 一类在
  单张图里就有 36 个连通区域），**不适合少样本 point-matching"找一个跟参考最像的目标"**（实测 2/2 用例
  完全未命中任何标注实例）；适合**语义分割模型训练（U-Net 类）/整体相面积占比统计**。metadata.sql 含
  micron_bar（µm 比例尺）、magnification、各相像素数与官方 train/test 划分，可做相占比定量分析。
  另有 unlabeled 164 张（191MB，Release 同页）未下载，需要时再取。
- **NIST_CalibrationParticles（NIST mds2-3043）**：真实月球风化层颗粒 SEM 图，公共领域，但（1）
  `.dat` 测量文件用独立 X 射线 CT 扫描的 STL 文件名索引，与 SEM 图像文件名无法逐颗粒对应，原定的
  "已知真实尺寸校准"用途做不到；（2）多数图像背景大片空白、颗粒稀疏孤立，少样本检测反复匹配到
  背景噪声（相似度还异常虚高，0.999+），已被 `BAM_TiO2`/`EMPS`（颗粒密度更高、有像素级 mask）
  取代，**已删除**。

## 许可提醒
- 若分析结果用于对外发布（比赛提交、论文等），逐项核对许可并保留署名依据：
  - Aachen-Heerlen / cigRockSEM：CC BY 4.0（须署名）；
  - MoNuSeg / DenseCrack：CC BY-NC-SA 4.0（**禁止商用**）；
  - BAM_TiO2：CC BY-NC-ND 4.0（**禁止商用 + 禁止演绎**，须署名）；
  - EMPS：CC BY 4.0（可商用，须署名）；
  - MetalDAM：MIT（宽松）；
  - CrackSeg9k：各子集许可不一，学术使用惯例引用论文；
  - NIST_AM_Pores：美国政府作品，公共领域；
  - DoITPoMS：教学用途，遵守其使用条款。
- 不同数据集的显微镜倍率、灰度/伪彩、目标类别差异大，混用前注意域差异；
  CrackSeg9k 为宏观裂纹，勿与 SEM 显微裂纹混为一谈。
