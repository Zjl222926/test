# E1–E3 方法比较：预设实验口径（尚未训练）

## 共同样品和外层验证

所有方法读取同一 `model_inputs/model_index.csv` 的 117 行、相同 E1/E2/E3 标签和 `test_fold`，绝不按方法换样品。外层按 DOI/study 整组留出，共 4 折；`cv_train_test_rows.csv` 是唯一的 train/test 划分依据。任何同 DOI 样品、其复制版或 bootstrap 版均不得同时落在训练和测试侧。测试折原始比例保持不变。

## 准备比较的方法

1. **传统分布距离**：对 `age_arrays.jsonl` 中每个样品的原始 U–Pb 年龄，计算 Wasserstein-1 最近训练样品基线；可另设 KS 距离最近样品基线。只从训练样品找邻居。
2. **普通机器学习**：在同一 KDE 特征上比较 Logistic Regression、Random Forest、XGBoost、LightGBM（已安装时）。不能安装的包只记录，不更改数据或折。KDE 在此前四分类特征工程实验中表现较好，但 E1 已重新定义且 E4 已移出，故该旧结果不是三分类性能保证。
3. **自动机器学习**：TPOT 仅在能把 DOI 分组正确传入内层验证、且不穿透外层测试折时使用；否则记录不运行原因，不用普通随机 CV 冒充 grouped CV。
4. **特征敏感性**：同样样品、同样外层折下以 Piecewise-A、25 Ma 固定分箱重复可运行的普通 ML 方法；不得针对某一特征单独改变端元定义。

KDE、分箱和年龄数组均为逐样品的无监督表示，预先计算不使用其他样品标签。若某方法需标准化、特征选择、降维或概率校准，必须只在每个训练折拟合，再应用于留出的测试折。超参数搜索（如执行）也只能使用训练折内部的 DOI 分组验证，不能查看外层测试折后再调整搜索空间。

## 不平衡与指标

类样品数为 E1 30、E2 53、E3 34；study 数分别 5、9、7。主要输出 pooled out-of-fold Balanced Accuracy、Macro-F1、每类 Recall/Precision/F1 和混淆矩阵；报告每折原始 Accuracy、Balanced Accuracy、Macro-F1 及折间波动，并给出 study 等权敏感性。普通 Accuracy 仅作辅助。类权重或 study 权重必须在各训练折内计算；训练折内重采样不能改变测试折，复制样品不算新的独立地质样品。

方法优劣不得只看最高单折或总体 Accuracy。不得利用 TARGET、Cathaysia/Hainan、E4、C1–C3 或基岩来挑选三分类方法、特征、超参数或阈值。分类概率不直接解释为物源贡献率。

## 运行前最终核对

- 确认 `validation_checks.json` 中 `sample_uid_unique`、`spectrum_fingerprint_unique`、`grain_record_uid_unique`、`no_cross_fold_study_split` 和 `all_fold_tests_have_all_three_classes` 均为 `true`。
- 核对各特征文件均为 117 行；KDE 为 461 维、Piecewise-A 为 52 维、固定分箱为 184 维；与 `model_index.csv` 行序相同。
- 核对 11,637 个颗粒只属于最终 117 个 UID；E1 河砂的 188 个原文拒绝记录不在模型输入中。
- 比较报告应同时附 study 分布，明确 E1 5 study 的泛化精度仍受限制。
