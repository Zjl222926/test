# TPOT FORMAL V2.1 协议修正

V2试运行日志出现 `y_pred contains classes not in y_true`。审计显示，带随机打乱的内层 `StratifiedGroupKFold` 在部分外层训练子集中生成了缺少E1、E2或E3的验证折。虽然study仍然完全隔离，且外层预测本身有效，但不同内层折的Balanced Accuracy并非在相同三类支持上计算，因此V2不作为正式结果。

V2.1仅修正内层折构造，不改变以下内容：117个样品、E1/E2/E3标签、KDE特征、固定4个外层study折、TPOT搜索空间、population、generations、优化指标、随机种子或TARGET隔离规则。

V2.1使用非打乱的 `StratifiedGroupKFold(n_splits=3)`。在当前冻结数据中，每一个外层训练子集的三个内层验证折均同时包含E1、E2和E3。脚本对此进行强制断言，并在 `inner_group_splits.json` 中保存每折类别计数。

V2试运行结果保留在 `outputs/tpot_formal_v2/`，状态为废弃协议试运行；V2.1正式结果写入独立目录：

```text
outputs/tpot_formal_v2_1/
```

正式启动脚本为：

```text
run_tpot_formal_v2_1.sh
```
