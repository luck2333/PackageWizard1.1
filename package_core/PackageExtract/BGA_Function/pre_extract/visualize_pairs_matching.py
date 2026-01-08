"""辅助可视化：标注与标尺线的匹配关系.

该模块新增了独立函数，不会修改或覆盖原有流程，便于在调试阶段直观查看
标注框（红）与标尺线/箭头框（绿、蓝）的空间关系。
"""

from __future__ import annotations

import os
from typing import Iterable, List, Mapping, Optional, Sequence, Union

import cv2
import numpy as np

BBox = Sequence[float]
OCRItem = Mapping[str, object]


def _load_image(img_path: str) -> np.ndarray:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"无法找到图片: {img_path}")
    with open(img_path, "rb") as f:
        np_arr = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {img_path}")
    return img


def visualize_pairing(
    img_path: str,
    pairs: Iterable[Union[BBox, Sequence[float]]],
    ocr_items: List[OCRItem],
    arrow_pairs: Optional[Iterable[Union[BBox, Sequence[float]]]] = None,
    output_path: Optional[str] = None,
    window_name: str = "pairs_vs_ocr",
) -> np.ndarray:
    """将标注与标尺线/箭头的匹配结果可视化。

    Args:
        img_path: 原图路径。
        pairs: 标尺线或匹配到的尺寸线框（x1, y1, x2, y2）。
        ocr_items: 标注字典列表，需包含 `location` 和可选的 `matched_pairs_location`。
        arrow_pairs: 可选，箭头对信息，会以蓝色展示。
        output_path: 若指定则保存渲染后的图片。
        window_name: 可选窗口名称，便于区分不同视图。

    Returns:
        渲染完成的 BGR 图像矩阵。
    """

    img = _load_image(img_path)
    canvas = img.copy()

    # 绘制标尺线/尺寸线（绿色）
    for pair in pairs:
        x1, y1, x2, y2 = map(int, pair[:4])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2, lineType=cv2.LINE_4)

    # 绘制箭头信息（蓝色）
    if arrow_pairs is not None:
        for pair in arrow_pairs:
            x1, y1, x2, y2 = map(int, pair[:4])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 2, lineType=cv2.LINE_4)

    # 绘制 OCR 识别框（红色），若有匹配到的标尺线则同时描边绿色外框
    for item in ocr_items:
        loc = np.array(item.get("location", []), dtype=float)
        if loc.shape[0] >= 4:
            x1, y1, x2, y2 = map(int, loc[:4])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2, lineType=cv2.LINE_8)

        matched_loc = np.array(item.get("matched_pairs_location", []), dtype=float)
        if matched_loc.shape[0] >= 4:
            x1, y1, x2, y2 = map(int, matched_loc[:4])
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 1, lineType=cv2.LINE_AA)

        label = item.get("ocr_strings") or ""
        if len(label) > 0 and loc.shape[0] >= 2:
            cv2.putText(
                canvas,
                str(label).strip(),
                (int(loc[0]), int(loc[1]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 255),
                1,
                lineType=cv2.LINE_AA,
            )

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, canvas)

    cv2.imshow(window_name, canvas)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

    return canvas

