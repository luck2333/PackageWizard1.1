# F4 阶段函数速览

本文梳理封装提取流程中 F4（尺寸线与参数提取）阶段的主要函数与用法，重点覆盖 `f4_pipeline_runner.py` 串联的步骤。流程默认输入位于 `Result/Package_extract/data` 下的多视图图片，输出为参数列表。

## 入口：`run_f4_pipeline`
- 位置：`package_core/PackageExtract/BGA_Function/f4_pipeline_runner.py`
- 作用：串联 F4.1–F4.9 的主要步骤，返回经过封装的参数列表。
- 参数：
  - `image_root`：包含 `top/bottom/side/detailed` 视图图片的目录。
  - `package_class`：封装类别（如 `"BGA"`）。
  - `key`：影响 OCR 清洗策略的标志位。
  - `test_mode`：传递给尺寸线补齐模块的调试开关。
- 核心流程：
  1. 枚举视图并通过 `common_pipeline.get_data_location_by_yolo_dbnet` 汇总 YOLO/DBNet 检测结果为 `L3`。
  2. F4.1–F4.4：调用 `pre_extract` 下的匹配函数（其他框、PIN、角度、数字尺寸线）。
  3. F4.45：用 `num_direction.add_direction_field_to_yolox_nums` 为数字框补充朝向信息。
  4. F4.6：`common_pipeline.enrich_pairs_with_lines` 识别标尺线并附加长度信息。
  5. F4.7：`match_triple_factor.match_arrow_pairs_with_yolox` 生成尺寸线与箭头配对后，再经 `common_pipeline.preprocess_pairs_and_text` 整理候选，并运行 `common_pipeline.run_svtr_ocr` 与 `normalize_ocr_candidates` 完成 OCR 推理与清洗。
  6. F4.8：`common_pipeline.extract_pin_serials` 识别 PIN 序号，`match_pairs_with_text` 将尺寸线重新与文本匹配。
  7. F4.9：`common_pipeline.finalize_pairs` 清理配对结果，`compute_qfp_parameters` 生成参数结构，最后由 `function_tool.get_BGA_parameter_data` 产出列表。

## F4.6：尺寸线补齐
- `enrich_pairs_with_lines(L3, image_root, test_mode)`：
  - 输入：各视图的 YOLO 箭头框 `yolox_pairs`。
  - 处理：调用 `find_pairs_length` 识别标尺线两端的引线，生成包含箭头、引线与长度的 13 列数组。
  - 输出：将结果写回 `*_yolox_pairs_length`。

## F4.7：尺寸线与文本整理
- `preprocess_pairs_and_text(L3, key)`：融合 DBNet 与 YOLO 结果，生成尺寸线与文本的精炼集合，并保留副本用于后续回溯。
- `run_svtr_ocr(L3)`：对合并后的文本框跑 SVTR OCR，并合并重叠框，产出各视图 `*_ocr_data`。
- `normalize_ocr_candidates(L3, key)`：调用 `data_wrangling` 根据数值/方向清洗 OCR 候选，补全 `max/medium/min`。

## F4.8：PIN 与尺寸线配对
- `extract_pin_serials(L3, package_class)`：对 QFP/QFN/SOP/SON 等封装提取序号信息并写回 `top/bottom_serial_numbers_data`；BGA 的 PIN 逻辑保留在注释中。
- `match_pairs_with_text(L3, key)`：利用角度框、封装边框与 OCR 结果，通过 `MPD` 将尺寸线与文本重新绑定。
- `finalize_pairs(L3)`：调用 `get_better_data_2` 过滤/补全配对结果，并输出 `yolox_pairs_*` 供参数计算使用。

## F4.9：参数计算与输出
- `compute_qfp_parameters(L3)`：
  - 依赖：`finalize_pairs` 产生的配对结果、引线长度、OCR 数据及边框信息。
  - 步骤：
    1. `get_serial` 推断 PIN 阵列尺寸 `nx/ny`。
    2. `get_QFP_body` 根据引线与边框确定主体长宽（`body_x/body_y`）。
    3. `get_QFP_parameter_list` 整合各视图候选，构建参数列表；若存在多个候选，结合 `get_QFP_high`、`get_QFP_pitch` 与 `resort_parameter_list_2` 缩小范围。
  - 返回：参数列表与 `nx/ny`。
- `function_tool.get_BGA_parameter_data(parameters, nx, ny)`：将 `compute_qfp_parameters` 的结果映射为最终的 BGA 参数列表（包含 PIN 间距、实体尺寸等），作为 `run_f4_pipeline` 的返回值。
