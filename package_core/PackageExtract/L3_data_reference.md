# L3 数据结构说明

## 1. L3 的整体数据类型与组织方式

`L3` 是一个 **list[dict]** 的容器，每个元素都包含以下字段：

- `list_name`: 字符串，用来标识该数据列表的用途与所属视图（如 `top_dbnet_data`）。
- `list`: 实际的数据列表/数组（通常是检测框、OCR 结果、配对结果等）。

L3 的构造在 `common_pipeline.get_data_location_by_yolo_dbnet` 中完成，流程会遍历每个视图（`top/bottom/side/detailed`），将 YOLO 与 DBNet 的检测结果拆分并以 `list_name` 命名写入 L3。特殊情况是 bottom 视图还会追加 BGA 的序号信息。整体结构是一个扁平列表，不是嵌套字典。【F:package_core/PackageExtract/common_pipeline.py†L167-L235】

## 2. L3 中常见的 `list_name` 字段

以下键名是 L3 的核心组成，所有封装类型都会复用这些字段：

- `{view}_dbnet_data`: DBNet 检测出的文字框坐标。
- `{view}_yolox_pairs`: YOLOX 检测出的尺寸标尺线/箭头框。
- `{view}_yolox_num`: YOLOX 检测出的尺寸数字框。
- `{view}_yolox_serial_num`: YOLOX 检测出的序号框（如 PIN 序号）。
- `{view}_pin`, `{view}_pad`, `{view}_border`, `{view}_other`, `{view}_angle_pairs`: 其他结构框（引脚、焊盘、边框、OTHER、角度标注）。
- `bottom_BGA_serial_letter`, `bottom_BGA_serial_num`: 仅 bottom 视图用于 BGA 序号的字母/数字框。

这些字段在 L3 构建阶段被统一写入，之后所有的 F4.6-F4.9 流程都通过 `find_list(L3, list_name)` 来获取对应数据。【F:package_core/PackageExtract/common_pipeline.py†L167-L235】

## 3. OCR 数据在 L3 中的结构（`*_ocr_data`）

OCR 相关结果在 L3 中以 `top_ocr_data / bottom_ocr_data / side_ocr_data / detailed_ocr_data` 的形式出现。OCR 数据的结构由 `convert_Dic` 统一包装，形成如下 dict 结构：

- `location`: 对应框的坐标数组（与 DBNet 框一致）。
- `ocr_strings`: OCR 识别的原始字符串。
- `key_info`: 提取后的关键 token 列表（数字、±、Φ、max/min 等）。
- `Absolutely`: 用于标记语义（如 `angle`、`mb_pin_diameter` 等）。
- `max_medium_min`: 经过推理计算后的 [max, mid, min] 数值数组。

该结构在 `convert_Dic` 中完成初始化，并被后续的清洗/归一化流程反复读写。【F:package_core/PackageExtract/get_pairs_data_present5_test.py†L4035-L4063】

## 4. 参数推理所依赖的 L3 字段（重点部分）

参数推理（F4.9）阶段主要依赖以下 L3 字段：

### 4.1 视图 OCR 数据

参数候选主要来自四个 OCR 列表：

- `top_ocr_data`
- `bottom_ocr_data`
- `side_ocr_data`
- `detailed_ocr_data`

`get_QFP_parameter_list` 会遍历这些 OCR 列表，根据 `max_medium_min` 的区间范围将标注分配到不同参数的 `maybe_data` 中，同时也会参考 `Absolutely` 与 `key_info`（例如 angle 或 Φ）进行额外的语义分配。【F:package_core/PackageExtract/get_pairs_data_present5_test.py†L4334-L4565】

### 4.2 框与标尺线匹配结果

标尺线与标注的配对结果会被写回 L3，并在参数推理前用于计算尺寸线长度、推断 body 尺寸等。核心字段包括：

- `yolox_pairs_top` / `yolox_pairs_bottom`
- `top_yolox_pairs_length` / `bottom_yolox_pairs_length`
- `top_border` / `bottom_border`

这些字段会被 `compute_qfp_parameters` 读取，用于调用 `get_QFP_body` 计算封装主体尺寸，从而辅助筛选 D/D1/E/E1 等参数范围。【F:package_core/PackageExtract/common_pipeline.py†L645-L691】

### 4.3 序号提取结果

PIN 序号推断会先生成 `top_serial_numbers_data` 与 `bottom_serial_numbers_data`，随后在参数推理中用于推断 `nx/ny`（PIN 数量），再参与 pitch 相关参数的推断：

- `top_serial_numbers_data`
- `bottom_serial_numbers_data`

这些字段在 `compute_qfp_parameters` 中通过 `get_serial` 计算出 `nx`/`ny`，然后影响 pitch 的推断流程。【F:package_core/PackageExtract/common_pipeline.py†L649-L710】

### 4.4 参数推理中最关键的 OCR 字段

参数推理阶段最依赖 OCR 字段如下：

- `max_medium_min`: 提供尺寸数值区间，是各参数候选分配的核心依据。
- `key_info`: 用来捕获 `Φ`、`±` 等符号，进而决定特殊参数归类。
- `Absolutely`: 指示角度(`angle`)或 pin 直径(`pin_diameter`)等语义标签。

这些字段在 `get_QFP_parameter_list` 中直接参与参数归类（如角度、Φ 参数、A/A1 等）。【F:package_core/PackageExtract/get_pairs_data_present5_test.py†L4334-L4565】

## 5. 使用建议（便于后续维护）

- 若新增参数推理逻辑，应尽量复用 `max_medium_min` 与 `key_info` 字段，避免引入新的字段结构。
- 如果需要在推理阶段区分某类标注（如 Φ），应在 OCR 清洗/归一化流程中写入 `Absolutely` 或补充 `key_info` token，以保证推理时一致可用。
- 若要扩展 L3 字段，建议沿用 `list_name` + `list` 的统一格式，并保证 `find_list` 能检索到新字段。
