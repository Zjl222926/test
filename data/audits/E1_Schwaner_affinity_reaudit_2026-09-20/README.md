# E1 = SW Borneo / Schwaner affinity 碎屑端元复审

审计日期：2026-09-20。此文件夹是**从冻结候选数据库衍生的训练清单**，没有改动 `Borneo_DZ_database_FINAL_v1_2026-09-14.zip` 或解压后的正式 CSV，也没有改动原始 E1 标签。没有设置 core/expanded 分级或最低颗粒数门槛。

## 结果

| 口径 | 物理碎屑样品 | 独立 DOI/study | 数据库 `VALID_AGE` 年龄行 |
|---|---:|---:|---:|
| 原 E1 `MAIN_TRAIN_CANDIDATE` | 39 | 8 | 4,050 |
| 统一 E1 Schwaner affinity 保留 | **30** | **5** | **3,174** |
| 转外部参照，不进入 E1 训练 | 9 | 4（与保留集可共享 DOI） | 876 |

保留的 30 个样品中，21 个是沉积岩碎屑样品（2,149 行），9 个是现代河砂（1,025 行）。Breitfeld et al. (2020) 的 9 个河砂样品在原始附表中共 1,025 次分析，其中原文 `accept=Y` 的为 **837**；若仅对这项研究应用其作者接受规则，保留集为 **2,986** 行。这个数**不是**其他研究也经过同一谐和度规则筛选后的统一质量控制结果，正式建模前须确定一致的颗粒级 QC 口径。

保留集按 DOI 的样品数／数据库年龄行数分别为 Wang 2022（6／406）、Li 2022（1／113）、Breitfeld 2020（9／1,025）、Quek 2023（5／410）、Breitfeld & Hall 2018（9／1,220）。两个 Breitfeld study 占保留样品 18/30、年龄行 2,245/3,174，故后续验证必须按 DOI/study 分组；30 个样品并非 30 个完全独立研究。

## 判定规则与地质理由

1. 仅审计原数据库中 E1、`MAIN_TRAIN_CANDIDATE`、`SEDIMENTARY_DETRITAL` 或 `MODERN_RIVER_OR_SAND` 的 39 个样品；床岩和原本标为 TARGET 重叠参照的记录不进入训练候选。
2. 保留西南婆罗洲/Schwaner 本体、其河流输出，以及原文明确与 SW Borneo/Schwaner 供源或再循环有联系的 Kuching/Kayan/Ketungau 沉积单元。允许混合来源；不要求“纯 Schwaner”谱，也不按年龄谱峰位、距离或模型性能筛选。
3. `STB74b` 原数据库 `formation` 为空。Breitfeld & Hall (2018) 正文 7.2.3 将其明确列在 **Bako–Mintu Sandstone**，地点为 Gunung Ngili 上部转石。地名“Ngili”不等于 Ngili Sandstone 地层。本审计在新增 `formation_audit` 列标注 Bako–Mintu，并保留样品；原数据库 `formation` 列不变。
4. Breitfeld et al. (2020) 正文说 8 个河砂，原始 Supplementary Table 3 却明确列出 **9 个不同样品编号和坐标**，分析合计 1,025、作者接受 837。故按附表可验证的 9 个物理样品保留；正文—附表数量不一致作为文献勘误风险记录，而非删除整组。原始样品编号见 `derived_data/Breitfeld2020_original_sample_ids.csv`。
5. 9 个未纳入 E1 的样品并未从数据库删除或改标。其不进入训练的原因是原文地层/构造物源与此处定义的 SW Borneo/Schwaner 输出不符或尚无直接证据，而**不是**年龄谱相似性：

| 样品 | 数量 | 本次角色 | 理由 |
|---|---:|---|---|
| `STB75`, `STB75b`, `STB77`, `TB60a` | 4 | 外部参照 | Ngili Sandstone 原文主要指向邻近 Sadong/Kuching 地层，尚无明确 Schwaner 输出联系。 |
| `SR09`, `SR30` | 2 | 外部参照 | Serabang 位于 West Sarawak 早期活动陆缘；原研究强调 Hainan–Borneo 亲缘，未证明该样品直接代表 Schwaner 输出。 |
| `17MY-60A1`, `17MY-61C1` | 2 | 外部参照 | Lupar/Lubok Antu 增生混杂体系及 greywacke，原文的构造解释和区域来源不等同于 Schwaner 集水输出。 |
| `IK22-8` | 1 | 外部参照 | Meratus/Manunggul 位于东南婆罗洲含金刚石沉积体系；非 SW Borneo/Schwaner 地理和沉积系统。 |

原 39 个碎屑候选的 `sample_instance_uid` 均唯一，年龄谱指纹也无完全重复。本轮**没有因重复或身份无法确认而从这 39 个中剔除任何样品**；`STB74b` 和 Breitfeld 2020 河砂的先前身份疑点已由原始出版资料核实。原库另有 19 行 `TARGET_OVERLAP_REFERENCE`，在本轮之前即不属主训练候选；当前 ZIP 不含可供重新比对的 TARGET 原始颗粒表，故未声称独立复核了 TARGET 重叠。

## 文件

- `E1_Schwaner_affinity_reaudit.xlsx`：总览、30 个保留样品、9 个外部参照样品、逐 study 统计、河砂原始编号。
- `E1_retained_full_metadata.csv`：保留的 30 行完整原数据库样品元数据，追加审计列；原字段未删。
- `E1_external_or_deferred_full_metadata.csv`：9 行外部参照完整元数据及理由。
- `derived_data/E1_all_39_candidate_audit.csv`：原 39 个训练候选的完整审计轨迹。
- `derived_data/E1_study_summary.csv` 和 `derived_data/E1_comparison.csv`：统计依据。
- `derived_data/Breitfeld2020_original_sample_ids.csv`：原始附表 9 个物理编号/坐标。
- `scripts/`：可复现代码；原始 CSV/ZIP 只读。

## 数据来源与文献依据

源包：`C:\Users\admin\Desktop\新\outputs\Borneo_DZ_database_FINAL_v1_2026-09-14.zip`，SHA-256 `67e378ead13f9c74ce429ea15bd9cc0bc3796b3eec23cea0bbd9dbb6f05b2d9b`。使用其 `MASTER/all_samples_metadata_full.csv` 和 `MASTER/all_endpoints_UPb_full.csv`。河砂身份及作者接受标记另以本地存档的 Breitfeld 2020 Supplementary Table 3 核对。

- [Breitfeld et al. (2020), Schwaner 山区及河砂](https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2020.568715/full)
- [Breitfeld & Hall (2018), Kuching/Kayan/Ketungau 沉积物源](https://www.sciencedirect.com/science/article/abs/pii/S1342937X18301588)
- [Wang et al. (2022), Western Kuching–South Schwaner](https://www.sciencedirect.com/science/article/abs/pii/S1367912022000347)
- [Quek et al. (2023), SW Borneo 碎屑锆石](https://doi.org/10.1130/G50966.1)
- [Zhou et al. (2026), Serabang/Hainan–Borneo 亲缘](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025GC012837)
- [Wang et al. (2021), Kuching/Lupar 增生体系](https://research.monash.edu/en/publications/cretaceous-kuching-accretionary-orogenesis-in-malaysia-sarawak-ge/)
- [Kueter et al. (2016), Meratus/Manunggul](https://www.sciencedirect.com/science/article/abs/pii/S0024493716300706)

## 使用边界

这是一份**统一 E1 训练端元建议清单**，不是整个 E1–E3 体系的训练端元冻结公告。它保留地质上合理的混合输出，因此年龄谱内部异质性是研究对象，不是删样依据。正式机器学习前还须确定统一颗粒级 QC，并对 E1 的 study 集中使用分组交叉验证。
