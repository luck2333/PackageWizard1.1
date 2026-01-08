# Side 视图参数推理改动记录

## 本次调整的函数
- `package_core/PackageExtract/get_pairs_data_present5_test.py::infer_side_high_pair`（新增）
  - 从 side 视图候选中按数值排序推断高度对：最大值 → A，次大值 → A1；若存在 `Absolutely == 'high'` 的标记则优先用其作为 A。
- `package_core/PackageExtract/QFP_extract.py::find_QFP_parameter`（调用新增逻辑）
  - 在生成 `QFP_parameter_list` 之后立即调用 `infer_side_high_pair`，用 side 视图的最大/次大值分别填入 A、A1 的 `maybe_data`，随后再执行去重整理。

## 设计思路
- **问题背景**：side 视图通常只包含主体高度 A 与支撑高度 A1 两个尺寸，原逻辑仅通过范围筛选和 `get_QFP_high` 选取单个高度，无法在候选众多时稳定区分 A/A1。
- **推理规则**：
  - 直接使用 side 视图所有候选值进行排序，无需额外阈值；最大值推为 A，次大值推为 A1，符合常见图纸标注习惯。
  - 若标注中明确标记了 `Absolutely == 'high'`，则优先将该标注作为 A，保持原有“绝对正确”信号的优先级。
- **兼容性**：
  - 新的推理结果以覆盖方式写入 A、A1 的 `maybe_data`，随后交由 `resort_parameter_list_2` 去重整理，避免影响后续 pitch 等参数处理。

## 之后的改进方向
- 将 side 视图的来源信息（是否主体/支撑标尺线、是否连接主体轮廓）一并保存在 `maybe_data` 中，用于进一步验证 A/A1 的物理位置关系。
- 在缺失 side 视图时，回退到原有的范围筛选 + `get_QFP_high` 逻辑。
