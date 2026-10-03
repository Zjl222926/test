# 婆罗洲碎屑锆石 U–Pb 论文复现包（智星云/普通 Linux 服务器）

版本：2026-09-29  
主实验：E1–E3 三分类、117 个碎屑样品、`n >= 20`、KDE（bandwidth 20 Ma，0–4600 Ma，10 Ma 网格）、4-fold DOI/study grouped CV。

## 1. 本包复现什么

本包锁定论文当前正式分析口径：

- E1：SW Borneo / Schwaner affinity（30 个样品，5 个 study）；
- E2：Malay–Thai Peninsula / Western Sundaland affinity（53 个样品，9 个 study）；
- E3：Palawan–PCT affinity（34 个样品，7 个 study）；
- 合计 117 个碎屑样品、21 个分组 study、11,637 颗模型 U–Pb 年龄；
- 同一 DOI/study 不跨训练折和测试折；
- TARGET 不参与训练、调参或模型选择。

本包不使用旧的四分类、Broad-E3 或 747 样品实验。特别不要混入旧目录 `borneo_method_comparison_v2_corrected`，它与当前论文口径不兼容。

## 2. 文件结构

```text
data/training/                 冻结的 117 样品清单、固定 folds、KDE/年龄数组
data/target/                   140 个 TARGET_STRATIGRAPHIC 颗粒级输入
data/audits/                   E1/E1–E3/E4 地质审计依据
data/source_support/           E1 河砂作者接受标记原始补充表
source_archives/               冻结数据库 ZIP（用于全源重建和开放世界分析）
reference_outputs/             本地正式结果，用于自动核对
src/run_main_methods.py        传统方法、RF、XGBoost、LightGBM 正式 grouped CV
src/run_tpot.py                TPOT 有限预算 grouped nested CV
src/fit_final_and_predict.py   全部 117 样品拟合最终 RF 并预测 TARGET
src/compare_target.py          七种方法 TARGET 对比
src/run_sensitivities.py       n>=10/20/30 与 KDE/Piecewise-A 敏感性
src/open_world_compare.py      C1/C2/C3/E4/Cathaysia-Hainan 外部亲缘检验
src/run_gpu_optional.py        XGBoost/LightGBM GPU 后端敏感性（不是主结果）
```

模型输入矩阵不重复保存标签；其行顺序必须与 `data/training/model_inputs/model_index.csv` 完全一致。不要在 Excel 中重新排序后覆盖这些文件。

## 3. 建议的云端资源

本研究不是深度学习项目，117×461 的主矩阵很小。建议：

- Ubuntu 22.04 或更新的 Linux 镜像；
- 8–16 个 CPU 核；
- 32 GB RAM；
- 50 GB 磁盘；
- GPU 非必需。若要做 GPU 后端对照，一张 NVIDIA RTX 3090/4090、A10 或同级卡已足够，无需 A100/H100。

RF、R²、KS、Wasserstein 和当前 TPOT 流程主要运行在 CPU 上。GPU 只用于可选的 XGBoost/LightGBM 加速，而且本数据很小，GPU 可能不比 CPU 快。

## 4. 上传到智星云

优先上传最终 ZIP，而不是整个桌面工程。平台控制台会显示 SSH 地址、端口和密码。Windows PowerShell 示例：

```powershell
scp -P <SSH端口> Borneo_Zhixingyun_GPU_REPRO_RELEASE_2026-09-29.zip root@<服务器地址>:/root/data/
```

也可通过平台网页/Jupyter 文件管理器上传。云端执行：

```bash
cd /root/data
unzip Borneo_Zhixingyun_GPU_REPRO_RELEASE_2026-09-29.zip
cd Borneo_Zhixingyun_GPU_REPRO_RELEASE_2026-09-29
```

平台本地数据盘可能随实例释放而丢失；运行结束后应把 `outputs/`、日志和环境记录下载或同步到保留盘。

## 5. 安装环境

精确版本来自论文正式运行环境。推荐新建独立 Conda 环境：

```bash
conda env create -f environment.yml
conda activate borneo-dz-repro
python --version
python -c "import numpy,pandas,scipy,sklearn,xgboost,lightgbm,tpot; print(numpy.__version__, pandas.__version__, scipy.__version__, sklearn.__version__, xgboost.__version__, lightgbm.__version__, tpot.__version__)"
```

如果镜像没有 Conda：

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-exact.txt
```

正式版本为 Python 3.13.6、NumPy 2.4.6、Pandas 3.0.3、SciPy 1.18.0、scikit-learn 1.6.1、XGBoost 3.4.0、LightGBM 4.7.0、TPOT 1.1.0、joblib 1.5.3。

## 6. 先做论文原样复现（CPU）

即使租用了 GPU 实例，第一遍也应运行冻结的 CPU 配置：

```bash
mkdir -p logs
chmod +x *.sh
bash run_cpu_exact.sh 2>&1 | tee logs/cpu_exact.log
```

该命令依次完成：输入审计、传统方法与三种固定机器学习方法的 4-fold grouped CV、TPOT 有限预算实验、最终 RF 的 140 个 TARGET 预测、与冻结结果核对。

正式方法比较应得到下列总体 Macro-F1 / Balanced Accuracy：

| 方法 | Macro-F1 | Balanced Accuracy |
|---|---:|---:|
| Pearson R² | 0.458 | 0.458 |
| KS | 0.591 | 0.590 |
| Wasserstein | 0.382 | 0.426 |
| Random Forest | 0.594 | 0.602 |
| XGBoost | 0.530 | 0.551 |
| LightGBM | 0.530 | 0.551 |
| TPOT（有限预算、探索性） | 0.442 | 0.450 |

核对通过时生成 `outputs/reproduction_verification.json`，其中 `status` 应为 `PASS`。

TPOT 使用固定的 2 分钟/折搜索预算。它的搜索受操作系统、调度速度和可用 CPU 影响，跨平台可能选择不同但近似的流水线。因此，TPOT 应按论文中的“有限预算探索性结果”报告；不要通过延长时间或反复运行后挑选最好结果来替换正式结果。

## 7. GPU 后端敏感性（可选）

确认 CPU 结果后再执行：

```bash
bash run_gpu_optional.sh 2>&1 | tee logs/gpu_optional.log
```

XGBoost 使用 `tree_method="hist", device="cuda"`。LightGBM 的 CUDA 后端需要安装时已启用 CUDA；普通 CPU wheel 可能不支持，脚本会把失败原因写入 `outputs/gpu_sensitivity/gpu_run_metadata.json`，不会覆盖 CPU 正式结果。

GPU 结果只回答“更换计算后端是否改变分类输出/运行时间”，不能取代论文正式数值。GPU 浮点归约顺序可能使概率出现微小差异。

## 8. 全源重建、敏感性与开放世界分析

核心复现不需要解压 26 万余条数据库颗粒表。若要从冻结数据库重新生成训练输入，并运行阈值敏感性和开放世界检验：

```bash
bash prepare_full_source.sh 2>&1 | tee logs/prepare_full_source.log
bash run_extended_analysis.sh 2>&1 | tee logs/extended_analysis.log
```

`prepare_full_source.sh` 只解压本包内的冻结数据库 ZIP，不修改源档案。重建输出写到 `outputs/rebuilt_training/`，不会覆盖 `data/training/` 的正式冻结输入。

## 9. 关键输出

```text
outputs/method_comparison_cpu/main_method_comparison.csv
outputs/method_comparison_cpu/main_fold_metrics.csv
outputs/method_comparison_cpu/main_confusion_long.csv
outputs/method_comparison_cpu/main_oof_predictions.csv
outputs/method_comparison_cpu/tpot_*.csv
outputs/target_rf/TARGET_predictions_key.csv
outputs/target_rf/TARGET_traditional_reference_scores.csv
outputs/target_multimethod/02_TARGET_independent130_multimethod.csv
outputs/open_world_affinity/07_potential_omitted_affinity_target_list.csv
outputs/reproduction_verification.json
```

TARGET 输出中的 10 个训练物理重复样品必须继续标记为非独立；另有 17 个同 DOI 但不同物理样品，需要单独报告。分类概率不是物源贡献率。

## 10. 每次运行必须保存的环境证据

```bash
nvidia-smi > logs/nvidia-smi.txt 2>&1 || true
python --version > logs/python-version.txt 2>&1
python -m pip freeze > logs/pip-freeze.txt
uname -a > logs/uname.txt
```

同时保留：上传 ZIP 的 SHA-256、命令日志、所有生成的 `run_metadata.json`、异常日志和最终 `outputs/`。论文或补充材料中应区分：本地冻结正式结果、云端 CPU 复现结果、可选 GPU 后端敏感性结果。

## 11. 常见错误

- 结果变成四分类或 E3 样品数远大于 34：混入了旧 Broad-E3/旧 E4 代码。
- 样品数不是 117：训练清单或 `n>=20` 口径被改动。
- 同一 DOI 出现在训练和测试：固定 fold 文件被重排或重新随机划分。
- XGBoost GPU 报无 CUDA：当前 XGBoost wheel/驱动不匹配，先完成 CPU 正式复现。
- LightGBM 报 CUDA device 未启用：需要 CUDA 版 LightGBM；这不影响论文 CPU 正式复现。
- TPOT 结果略有变化：保存完整日志，不挑选最好一次；以冻结结果和有限预算性质为准。
