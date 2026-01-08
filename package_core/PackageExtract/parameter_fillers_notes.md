# 填参函数解读与按封装改进思路

本文梳理 `parameter_fillers.py` 中新增的按封装填参入口，并总结各封装在 `function_tool.py` 中的参数映射规则，最后给出可按封装改进填参的方向。所有内容均在不修改现有流水线的前提下，可作为后续迭代的参考。

## 当前填参入口
- 位置：`package_core/PackageExtract/parameter_fillers.py`
- 作用：为 BGA/QFP/QFN/SOP/SON 暴露独立的填参函数，内部直接调用 `function_tool.py` 里的原始 `get_*_parameter_data` 映射逻辑，便于外部试验或替换，而不影响现有主流程。
- 入口函数：
  - `fill_bga_parameters(parameter_candidates, nx, ny)` → `get_BGA_parameter_data(...)`
  - `fill_qfp_parameters(parameter_candidates, nx, ny)` → `get_QFP_parameter_data(...)`
  - `fill_qfn_parameters(parameter_candidates, nx, ny)` → `get_QFN_parameter_data(...)`
  - `fill_sop_parameters(parameter_candidates, nx, ny)` → `get_SOP_parameter_data(...)`
  - `fill_son_parameters(parameter_candidates, nx, ny)` → `get_SON_parameter_data(...)`
  - `fill_parameters_by_package(package_type, parameter_candidates, nx, ny)`：按封装类型调度上述函数，未覆盖的封装会抛出 `ValueError`。

## 各封装的现有映射规则
以下条目均基于 `function_tool.py` 中的 `get_*_parameter_data` 函数，记录参数候选(`maybe_data`)写入最终参数表的位置及取值优先级。

### BGA
- 产出 19 行的参数表，目前填充：
  - 行/列间距：使用 `maybe_data[6]`、`maybe_data[7]` 的首个候选；
  - 实体高与支撑高：来自 `maybe_data[4]`、`maybe_data[5]`；
  - 实体长/宽：优先 `maybe_data[2]`，缺失时回退到 `maybe_data[3]`（或反之）。
- PIN 行列数的写入被注释掉，表中对应行暂为空。
- 位置：`get_BGA_parameter_data`。【F:package_core/PackageExtract/function_tool.py†L243-L302】

### QFP
- 产出 19 行表，填充实体长宽高、支撑高、外围长宽、引脚宽、行列数、行列间距、散热盘尺寸、端子厚度与角度等，其中多数字段在 `maybe_data[2/3/4/5/6/7/8/9/12/13/14/15]` 间有主/备优先级。
- 行列数 `nx/ny` 会直接落在表的第 10、11 行。
- 位置：`get_QFP_parameter_data`。【F:package_core/PackageExtract/function_tool.py†L304-L503】

### SOP
- 产出 17 行表，重点写入实体长宽高、支撑高、引脚宽、行列数、间距、散热盘尺寸，端子厚度/角度等条目预留但未填。
- 引脚宽取 `maybe_data[7]`；间距优先 `maybe_data[6]`，缺失时使用 `maybe_data[7]`。
- 位置：`get_SOP_parameter_data`。【F:package_core/PackageExtract/function_tool.py†L508-L635】

### SON
- 产出 12 行表，写入实体长宽高、支撑高、PIN 长宽、行列数、间距、散热盘尺寸；端子厚度/角度等条目被注释掉。
- 长宽高分别来自 `maybe_data[2/3/4]`；PIN 间距优先 `maybe_data[6]`，回退到 `maybe_data[7]`。
- 位置：`get_SON_parameter_data`。【F:package_core/PackageExtract/function_tool.py†L637-L760】

### QFN
- 产出 15 行表，写入实体长宽高、支撑高、PIN 宽、行列数、间距、散热盘尺寸，削角与圆角标识留空。
- PIN 宽取自 `maybe_data[7]`；间距优先 `maybe_data[6]`，回退到 `maybe_data[7]`。
- 位置：`get_QFN_parameter_data`。【F:package_core/PackageExtract/function_tool.py†L764-L865】

## 按封装的改进思路
- **BGA**：
  - 开启已注释的 PIN 行列数写入，以便下游表格完整；
  - 根据球阵规格，允许行/列间距分别取不同槽位或做均值，避免单一候选导致偏差。
- **QFP**：
  - 将端子厚度与多角度（θ、θ1、θ2、θ3）按 JEDEC 表格顺序拆行，同时补充缺失的 PIN 长度计算逻辑（源注释中已有草稿）；
  - 在外围尺寸与主体尺寸冲突时，可集成 `alter_QFP_parameter_data` 的判定流程，自动校正长宽与外围长宽的关系。
- **SOP**：
  - 明确 PIN 长度来源（目前缺失），并与 `maybe_data` 编号对齐；
  - 针对散热盘尺寸与封装宽度关系增加一致性检查，防止写入异常值。
- **SON**：
  - 将 PIN 间距与 PIN 长度、宽度进行自检（例如长 ≥ 间距 - 宽度），以过滤识别噪声；
  - 补齐端子厚度/角度映射，或在填参结果中显式标记“未提取”。
- **QFN**：
  - 针对端子高度（A3）与支撑高（A1）分离填充，便于判断焊盘抬升；
  - 削角与圆角标识可通过 OCR 标签或尺寸占比进行二次推断，填入预留列。

这些改进可以通过在 `parameter_fillers.py` 旁新增封装专属版本（例如 `fill_qfp_parameters_v2`）逐步实验，对比现有输出再决定是否替换主流程。
