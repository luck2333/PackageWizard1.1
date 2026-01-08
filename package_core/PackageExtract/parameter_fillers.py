"""Dedicated parameter fill helpers for each package type.

These functions mirror the BGA fill pattern (`get_BGA_parameter_data`) and
provide a consistent entry point to convert the F4.9 candidate structures into
final parameter lists for each supported package. They do not modify the
original pipeline; callers can import these helpers to experiment with
package-specific output mapping without touching the existing code paths.
"""

from package_core.PackageExtract.function_tool import (
    get_BGA_parameter_data,
    get_QFP_parameter_data,
    get_QFN_parameter_data,
    get_SOP_parameter_data,
    get_SON_parameter_data,
)


def fill_bga_parameters(parameter_candidates, nx: int, ny: int):
    """Map BGA candidates to the standardized 19-row parameter table."""

    return get_BGA_parameter_data(parameter_candidates, nx, ny)


def fill_qfp_parameters(parameter_candidates, nx: int, ny: int):
    """Map QFP candidates to the standardized 19-row parameter table."""

    return get_QFP_parameter_data(parameter_candidates, nx, ny)


def fill_qfn_parameters(parameter_candidates, nx: int, ny: int):
    """Map QFN candidates to the standardized 19-row parameter table."""

    return get_QFN_parameter_data(parameter_candidates, nx, ny)


def fill_sop_parameters(parameter_candidates, nx: int, ny: int):
    """Map SOP candidates to the standardized 19-row parameter table."""

    return get_SOP_parameter_data(parameter_candidates, nx, ny)


def fill_son_parameters(parameter_candidates, nx: int, ny: int):
    """Map SON candidates to the standardized 19-row parameter table."""

    return get_SON_parameter_data(parameter_candidates, nx, ny)


def fill_parameters_by_package(package_type: str, parameter_candidates, nx: int, ny: int):
    """Dispatch to the package-specific filler following the BGA fill contract."""

    dispatch = {
        "BGA": fill_bga_parameters,
        "QFP": fill_qfp_parameters,
        "QFN": fill_qfn_parameters,
        "SOP": fill_sop_parameters,
        "SON": fill_son_parameters,
    }
    try:
        fill_fn = dispatch[package_type.upper()]
    except KeyError:
        raise ValueError(f"Unsupported package type: {package_type}")
    return fill_fn(parameter_candidates, nx, ny)
