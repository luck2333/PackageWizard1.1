# 参数推断（maybe_data -> 参数名）现状与改进思路

## 现状：填参函数和推断函数的分工
- `parameter_fillers.py` 中的封装专属填参函数（BGA/QFP/QFN/SOP/SON）只是把已经确定的 `maybe_data` 写入 19 行参数表，没有再做推断或排序逻辑；它们直接复用 `function_tool.get_*_parameter_data`。现有填参代码默认 `maybe_data` 中的候选已经按优先级排好序。【F:package_core/PackageExtract/parameter_fillers.py†L1-L44】
- 主要的“类型推断”发生在 F4.9 进入填参之前：
  - 视图解析阶段根据框线匹配把数值塞进 `maybe_data` 列表（如 `QFP_parameter_list[4]['maybe_data']` 收集到的都是高 A/A1 相关的候选）。
  - 专用的筛选函数对候选再做一次“判别”：
    - `get_QFP_high(side_ocr_data)`：优先找 `Absolutely == 'high'` 的候选，否则取 `max_medium_min[0]` 最大的一个作为器件高度 A；
    - `get_QFP_pitch(...)`：用行/列 PIN 数约束区间 `(nx-1)*0.7*body < pitch < body` 过滤出行距/列距；
    - 部分封装再用 `resort_parameter_list_2` 等函数对候选顺序做微调（按长度、宽度与外围尺寸关系筛掉不合理值）。

## 最新调整：侧视图 A/A1 推断
- 新增 `infer_side_high_pair(side_ocr_data)`（位于 `get_pairs_data_present5_test.py`），直接对 side 视图候选按 `max_medium_min[0]` 由大到小排序：
  - 若存在 `Absolutely == 'high'` 标记，则优先把最大的高标签作为 A；
  - 否则使用数值最大者推断为 A，次大者推断为 A1。
- `QFP_extract.find_QFP_parameter` 在组装 `QFP_parameter_list` 后立即调用上述函数，用推断结果覆盖 A、A1 的 `maybe_data`，再交给 `resort_parameter_list_2` 处理去重/冲突。【F:package_core/PackageExtract/QFP_extract.py†L547-L560】【F:package_core/PackageExtract/get_pairs_data_present5_test.py†L1643-L1675】

## 现有推断的局限
- `get_QFP_high` 等函数只选取“最大值”或单一标记，无法区分 A 与 A1（主体高 vs. 支撑高）的相对关系，也没有利用多视图一致性。
- `get_QFP_pitch` 依赖 `nx`/`ny` 和单一 body 尺寸，缺少对“次大值是引脚高度/支撑高”这类模式的识别。
- `maybe_data` 内部没有保留统计特征（分布、来源视图、重叠度），导致填参阶段无法在缺少人工标记时自动分配 A/A1、L/b 等标签。

## 按封装拆分的改进思路（示例：侧视图 A/A1 判别）
- **侧视图尺寸排序策略**：
  - 将侧视图 `maybe_data` 中的候选按 `max_medium_min[0]` 从大到小排序，记录前两名。
  - 若第一名来自主体标尺线（Inside/Outside 信息或连到主体轮廓），第二名来自引脚/支撑标尺线，则自动推断 "A = 最大值"、"A1 = 次大值"。
  - 若缺少明确连线信息，可用长宽比或绝对位置（接近基板的更可能是 A1）做启发式判别。
- **封装特定约束**：
  - QFP/QFN/SOP：要求 `A > A1`，且 `(A - A1)` 大致接近引脚厚度区间；
  - BGA：若存在球高标注，允许 `A ≈ A1`，但基座厚度可通过球径差值反推；
  - SON/QFN 暴露焊盘：额外检查 `D2/E2` 与主体 `D/E` 的比例，避免把暴露焊盘厚度误当 A1。
- **数据结构改进**：
  - 在 `maybe_data` 中保留来源视图、连接的标尺线 ID、Inside/Outside 标记和排序名次（max/2nd/3rd）。
  - 为每个封装写独立的“推断器”函数，只读 `maybe_data` 进行标签判别，输出带标签的候选（如 `{'role': 'A', 'data': ...}`），再交给现有的填参函数写表，做到逻辑解耦。

## 可落地的迭代顺序
1) 在单独的推断文件中为 QFP/QFN/SOP 先实现“最大值 = A、次大值 = A1”启发式，并在日志中打印命中/冲突情况；
2) 扩展到行距/列距：结合 `nx`/`ny` 与主体尺寸，为 pitch 计算“预计区间”，若多个候选符合则选择靠近区间中值的；
3) 为 BGA/SON 增加特有规则（球径、暴露焊盘厚度），同时保留回退到现有 `get_*` 筛选的逻辑，确保兼容旧流程。
