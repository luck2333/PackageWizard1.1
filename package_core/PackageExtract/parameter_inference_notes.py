"""Parameter inference mapping guide.

This file documents where the current inference steps live so developers can
jump directly to the implementation without re-reading the whole pipeline.
It is intentionally import-free and can be opened as plain text.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class InferenceStep:
    name: str
    description: str
    functions: List[str]
    references: List[str]


# Ordered list of the inference logic executed before values are written into
# the 19-row parameter tables (see ``parameter_fillers.py`` for the writers).
INFERENCE_STEPS: List[InferenceStep] = [
    InferenceStep(
        name="Candidate binning per view",
        description=(
            "OCR + ruler pairs are grouped into per-parameter buckets such as "
            "body length/width (D/E), height (A/A1) and pitches (e)."
        ),
        functions=[
            "get_BGA_parameter_list(top_ocr_data, bottom_ocr_data, side_ocr_data, detailed_ocr_data, body_x, body_y)",
        ],
        references=[
            "package_core/PackageExtract/get_pairs_data_present5_test.py::get_BGA_parameter_list",
        ],
    ),
    InferenceStep(
        name="Height selection (A vs. tallest candidate)",
        description=(
            "Chooses the tallest side-view candidate marked Absolutely=='high' "
            "or, as a fallback, the max of max_medium_min[0]."
        ),
        functions=[
            "get_QFP_high(side_ocr_data)",
        ],
        references=[
            "package_core/PackageExtract/get_pairs_data_present5_test.py::get_QFP_high",
            "package_core/PackageExtract/common_pipeline.py (compute_BGA_parameters)",
            "package_core/PackageExtract/QFP_extract.py / QFN_extract.py / SOP_extract.py / SON_extract.py",
        ],
    ),
    InferenceStep(
        name="Pitch filtering",
        description=(
            "Filters side-view distances into pitch_x/pitch_y using (nx-1)*0.7*body < pitch < body bounds."
        ),
        functions=[
            "get_QFP_pitch(side_ocr_data, body_x, body_y, nx, ny)",
        ],
        references=[
            "package_core/PackageExtract/get_pairs_data_present5_test.py::get_QFP_pitch",
            "package_core/PackageExtract/common_pipeline.py (compute_BGA_parameters)",
        ],
    ),
    InferenceStep(
        name="Outlier pruning / resorting",
        description=(
            "Drops or reorders candidates that contradict body dimensions before final fill."
        ),
        functions=[
            "resort_parameter_list_2(QFP_parameter_list)",
        ],
        references=[
            "package_core/PackageExtract/common_pipeline.py (compute_BGA_parameters)",
        ],
    ),
    InferenceStep(
        name="Parameter serialization into table rows",
        description=(
            "After inference, candidates are written into the 19-row table by package-specific fill helpers."
        ),
        functions=[
            "compute_BGA_parameters(...)",
            "get_serial(QFP_parameter_list)",
            "get_body(top_ocr_data, bottom_ocr_data, detailed_ocr_data)",
        ],
        references=[
            "package_core/PackageExtract/common_pipeline.py::compute_BGA_parameters",
            "package_core/PackageExtract/common_pipeline.py::get_serial",
            "package_core/PackageExtract/common_pipeline.py::get_body",
            "package_core/PackageExtract/parameter_fillers.py (package-specific writers)",
        ],
    ),
]

__all__ = ["InferenceStep", "INFERENCE_STEPS"]
