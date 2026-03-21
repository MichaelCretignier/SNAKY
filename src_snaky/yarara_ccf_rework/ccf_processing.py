import glob as glob
import sys
from typing import NamedTuple, TypedDict

import matplotlib.pylab as plt
import numpy as np
import numpy.typing as npt

from scipy.ndimage import map_coordinates
from src_snaky.snaky_classes import tableXY
from src_snaky.yarara_ccf_rework.ccf_config import CCFConfig
from src_snaky.yarara_ccf_rework.output_config import OutputConfig
from src_snaky.yarara_ccf_rework.stellar_params import StellarParams

import logging

logger = logging.getLogger('snaky')

class Parabola_Vertex(NamedTuple):
    x: float
    y: float
def parabola_vertex(a: float, b: float, c: float) -> Parabola_Vertex:
    x = -b / (2.0 * a)
    return Parabola_Vertex(x, a * x**2 + b * x + c)

class Replace_None_Return(NamedTuple):
    y:float
    y_err: float
def replace_none(y: float, yerr: float | None) -> Replace_None_Return:
    if yerr is None:
        return Replace_None_Return(np.nan, 1e6)
    return Replace_None_Return(y, yerr)


def normalize_ccf_backup(ccf: tableXY) -> tableXY:
    ccf_backup = ccf.copy()
    p75 = np.nanpercentile(ccf_backup.y, 75)

    # Guard against degenerate continuum
    if p75 == 0:
        return ccf_backup

    ccf_backup.y /= p75
    ccf_backup.yerr /= p75
    return ccf_backup


def fit_ccf_model(
    ccf: tableXY,
    analytical_model: str,
    beta0: float,
    plot: bool = False,
) -> None:
    if analytical_model == "gaussian":
        ccf.fit_gaussian(Plot=plot)
    else:
        ccf.fit_GND(Plot=plot, beta_fixed=int(beta0))


def resolve_model_parametric(analytical_model: str, beta0: float) -> str:
    return "GND2.0" if analytical_model == "gaussian" else f"GND{beta0:.1f}"


def find_ccf_center(ccf: tableXY, fwhm: float) -> float:
    ccf.y *= -1
    ccf.yerr = np.sqrt(np.abs(ccf.y))
    ccf.find_max(vicinity=5)

    # Derivative peak detection — relax vicinity until two peaks are found
    ccf.diff(replace=False)
    assert ccf.deri is not None
    ccf.deri.y = np.abs(ccf.deri.y)
    for vicinity in (4, 3, 2):
        ccf.deri.find_max(vicinity=vicinity)
        assert ccf.deri.x_max is not None
        if len(ccf.deri.x_max) > 1:
            break

    assert ccf.deri.y_max is not None
    sorted_peaks = np.argsort(ccf.deri.y_max)

    assert ccf.deri.x_max is not None
    first_max: float = ccf.deri.x_max[sorted_peaks[-1]]
    second_max: float = ccf.deri.x_max[sorted_peaks[-2]]

    ccf.y *= -1

    # Prefer the local maximum closest to the midpoint of the two shoulders
    # Fall back to the global minimum for broad lines (fwhm >= 15 km/s)
    mid = 0.5 * (first_max + second_max)
    if (np.min(np.abs(ccf.x_max - mid)) < 5) and (fwhm < 15):
        center = ccf.x_max[np.argmin(np.abs(ccf.x_max - mid))]
    else:
        center = ccf.x[ccf.y.argmin()]

    return center


def trim_ccf(ccf: tableXY, rv_borders: float, del_outside_max: bool) -> None:

    if not del_outside_max:
        window = (ccf.x > -rv_borders) & (ccf.x < rv_borders)
        ccf.supress_mask(window)
    else:
        ccf.find_max(vicinity=10)
        ccf.index_max = np.sort(ccf.index_max)
        window = np.zeros(len(ccf.x), dtype=bool)
        window[ccf.index_max[0] : ccf.index_max[1] + 1] = True
        ccf.supress_mask(window)


def normalize_ccf(ccf: tableXY, normalisation: str) -> None:

    if normalisation == "left":
        norm = ccf.y[0]
    else:
        half = len(ccf.y) // 2
        max1 = np.argmax(ccf.y[:half])
        max2 = np.argmax(ccf.y[half:]) + half
        slope = (ccf.y[max2] - ccf.y[max1]) / (max2 - max1)
        norm = slope * (np.arange(len(ccf.y)) - max2) + ccf.y[max2]

    # Guard against degenerate continuum
    norm = np.where(norm != 0, norm, 1.0)

    ccf.y /= norm
    ccf.yerr /= norm


# ---------------------------------------------------------------------------
# SUB-BLOCK 5 — CONSISTENCY CHECK & PARAMETER EXTRACTION
# ---------------------------------------------------------------------------


def check_ccf_consistency(
    ccf: tableXY,
    ccf_backup: tableXY,
    center: float,
    check_non_transform: bool,
) -> None:

    ccf_backup.params["cen"].value -= center

    if not check_non_transform:
        return

    V1 = ccf_backup.params["cen"].value
    V2 = ccf.params["cen"].value

    if abs(V1 - V2) > 1:
        logger.warning(
            f"Discrepancy detected between CCFs (*{V1:.4f}*/*{V2:.4f}* km/s), "
            f"reverting to non-transformed fit"
        )
        ccf.params = ccf_backup.params


def extract_ccf_parameters(
    ccf: tableXY,
    i: int,
    center: float,
    calibrated_phot_noise: dict,
) -> dict:

    rv_ccf, rv_ccf_std = replace_none(
        ccf.params["cen"].value + center,
        ccf.params["cen"].stderr
    )
    contrast_ccf, contrast_ccf_std = replace_none(
        -ccf.params["amp"].value,
        ccf.params["amp"].stderr
    )
    wid_ccf, wid_ccf_std = replace_none(
        ccf.params["wid"].value,
        ccf.params["wid"].stderr
    )
    offset_ccf, offset_ccf_std = replace_none(
        ccf.params["offset"].value,
        ccf.params["offset"].stderr
    )

    return {
        "rv": rv_ccf,
        "rv_std": calibrated_phot_noise["rv"][i],  # override with calibrated noise
        "contrast": contrast_ccf,
        "contrast_std": calibrated_phot_noise["contrast"][i],
        "fwhm": wid_ccf,
        "fwhm_std": calibrated_phot_noise["fwhm"][i],
        "offset": offset_ccf,
        "offset_std": offset_ccf_std,
    }


_BISSPAN_FALLBACK_WINDOWS = (0.5, 2.0, 5.0)  # km/s — progressive fallback windows


def clip_ccf_for_bisspan(ccf: tableXY, bis_range: float) -> None:

    for window in (bis_range, *_BISSPAN_FALLBACK_WINDOWS):
        ccf.clip(min=[-window, None], max=[window, None], replace=False)
        if len(ccf.clipped.x) >= 5:
            if window != bis_range:
                logger.info(f"BISSPAN window expanded to *±{window}* km/s")
            return

    logger.warning("Could not find enough points for BISSPAN even at ±5 km/s")


def compute_parabolic_center(ccf: tableXY, center: float) -> tuple[float, float]:

    ccf.clipped.fit_poly()
    a, b, c = ccf.clipped.poly_coefficient
    x_v, y_v = parabola_vertex(a, b, c)
    return x_v + center, y_v


def compute_bisspan(ccf: tableXY, rv_ccf: float, dv: float, bis_range: float) -> float:

    ccf_core = ccf.copy()

    if rv_ccf == rv_ccf:  # guard NaN
        ccf_core.x -= rv_ccf

    step = dv / 1000  # m/s -> km/s
    half_grid = np.arange(0, bis_range + step * 0.99, step)
    vrad_center = np.concatenate([-half_grid[1:][::-1], half_grid])  # symmetric grid

    ccf_core.interpolate(new_grid=vrad_center, replace=True, method="cubic")
    ccf_core.fit_poly()

    a, b, _ = ccf_core.poly_coefficient
    x_v, _ = parabola_vertex(a, b, _)
    return x_v


def compute_equivalent_width(ccf: tableXY) -> float:

    return float(np.mean(1.0 - ccf.y))


def compute_ccf_diagnostics(
    ccf: tableXY,
    rv_ccf: float,
    center: float,
    dv: float,
    bis_range: float,
    i: int,
    calibrated_phot_noise: dict,
) -> dict:

    clip_ccf_for_bisspan(ccf, bis_range)
    para_center, para_depth = compute_parabolic_center(ccf, center)
    ew = compute_equivalent_width(ccf)
    bisspan = compute_bisspan(ccf, rv_ccf, dv, bis_range)

    return {
        "ew": ew,
        "ew_std": calibrated_phot_noise["ew"][i],
        "center": para_center,
        "center_std": calibrated_phot_noise["center"][i],
        "depth": 1.0 - para_depth,
        "depth_std": calibrated_phot_noise["depth"][i],
        "bisspan": bisspan,
        "bisspan_std": calibrated_phot_noise["vspan"][i],
    }

class CCFResult(TypedDict):
    rv:           np.float64
    rv_std:       np.float64
    contrast:     np.float64
    contrast_std: np.float64
    fwhm:         np.float64
    fwhm_std:     np.float64
    offset:       np.float64
    offset_std:   np.float64
    ew:           np.float64
    ew_std:       np.float64
    center:       np.float64
    center_std:   np.float64
    depth:        np.float64
    depth_std:    np.float64
    bisspan:      np.float64
    bisspan_std:  np.float64

# def process_single_ccf(
#     i: int,
#     vrad: np.ndarray,
#     ccf_power_col: np.ndarray,
#     ccf_power_std_col: np.ndarray,
#     dv: float,
#     stellar_params: StellarParams,
#     ccf_config: CCFConfig,
#     output_config: OutputConfig,
#     calibrated_phot_noise: dict,
#     beta0: float,
# ) -> dict:

#     ccf = tableXY(vrad / 1000, ccf_power_col, ccf_power_std_col)
#     ccf_backup = normalize_ccf_backup(ccf)

#     # Initial fit on raw CCF for consistency check reference
#     fit_ccf_model(
#         ccf_backup, ccf_config.analytical_model, beta0, plot=output_config.debug
#     )

#     if output_config.debug:
#         plt.figure("debug")
#         ccf_backup.plot()
#         plt.close("debug")

#     # Center detection
#     center = find_ccf_center(ccf, stellar_params.fwhm)
#     ccf.x -= center

#     # Trim, normalize, refit
#     trim_ccf(ccf, ccf_config.rv_borders, ccf_config.del_outside_max)
#     normalize_ccf(ccf, ccf_config.normalisation)
#     fit_ccf_model(ccf, ccf_config.analytical_model, beta0)

#     if output_config.debug:
#         ccf.plot(color=None)

#     # Consistency check — may replace ccf.params with backup
#     check_ccf_consistency(ccf, ccf_backup, center, ccf_config.check_non_transform)

#     # Parameter extraction
#     params = extract_ccf_parameters(ccf, i, center, calibrated_phot_noise)

#     # Diagnostics
#     diagnostics = compute_ccf_diagnostics(
#         ccf, params["rv"], center, dv, ccf_config.bis_range, i, calibrated_phot_noise
#     )

#     return {**params, **diagnostics}

class CCFResults(TypedDict):
    rv:           npt.NDArray[np.float64]
    rv_std:       npt.NDArray[np.float64]
    contrast:     npt.NDArray[np.float64]
    contrast_std: npt.NDArray[np.float64]
    fwhm:         npt.NDArray[np.float64]
    fwhm_std:     npt.NDArray[np.float64]
    offset:       npt.NDArray[np.float64]
    offset_std:   npt.NDArray[np.float64]
    ew:           npt.NDArray[np.float64]
    ew_std:       npt.NDArray[np.float64]
    center:       npt.NDArray[np.float64]
    center_std:   npt.NDArray[np.float64]
    depth:        npt.NDArray[np.float64]
    depth_std:    npt.NDArray[np.float64]
    bisspan:      npt.NDArray[np.float64]
    bisspan_std:  npt.NDArray[np.float64]
def stack_ccf_results(results: list[CCFResult]) -> CCFResults:
    return CCFResults({
        key: np.array([r[key] for r in results], dtype=np.float64)
        for key in CCFResults.__annotations__
    })

def process_all_ccf(
    vrad: npt.NDArray[np.float64],
    ccf_power: npt.NDArray[np.float64],
    dv: float,
    ccf_config: CCFConfig,
    calibrated_phot_noise: dict,
) -> CCFResults:

    vrad_kms = vrad / 1000                                                      # (N_vel,)
    N_vel = len(vrad_kms)
    N_spec = ccf_power.shape[1]
    dv_kms = np.median(np.diff(vrad_kms))                                      # uniform step

    # ---- Normalize all CCFs at once ----
    p75 = np.nanpercentile(ccf_power, 75, axis=0)                              # (N_spec,)
    p75 = np.where(p75 != 0, p75, 1.0)
    ccf_norm = ccf_power / p75[np.newaxis, :]                                  # (N_vel, N_spec)

    # ---- Find all centers at once ----
    centers = vrad_kms[np.argmin(ccf_norm, axis=0)]                            # (N_spec,)

    # ---- Center all CCFs ----
    x_centered = vrad_kms[:, np.newaxis] - centers[np.newaxis, :]              # (N_vel, N_spec)

    # ---- Trim all CCFs at once ----
    in_window = np.abs(x_centered) <= ccf_config.rv_borders                    # (N_vel, N_spec)

    # ---- Normalize continuum ----
    if ccf_config.normalisation == 'left':
        norm = ccf_norm[0, :]                                                   # (N_spec,)
    else:
        half = len(vrad_kms) // 2
        norm = np.max(ccf_norm[:half], axis=0)                                 # (N_spec,)
    norm = np.where(norm != 0, norm, 1.0)
    ccf_norm = ccf_norm / norm[np.newaxis, :]                                  # (N_vel, N_spec)

    # ---- Fit all CCFs (moment-based vectorized Gaussian) ----
    depth = np.where(
        in_window,
        np.maximum(np.max(ccf_norm, axis=0)[np.newaxis, :] - ccf_norm, 0),
        0,
    )                                                                           # (N_vel, N_spec)
    total = np.sum(depth, axis=0) + 1e-20                                      # (N_spec,)
    rvs = np.sum(depth * x_centered, axis=0) / total + centers                # (N_spec,)

    # Weighted variance -> sigma -> fwhm
    x_rv_centered = x_centered - (rvs - centers)[np.newaxis, :]               # (N_vel, N_spec)
    sigmas = np.sqrt(np.sum(depth * x_rv_centered**2, axis=0) / total)        # (N_spec,)
    fwhms = sigmas * 2.355                                                      # (N_spec,)

    # Contrast and offset
    continuum = np.percentile(ccf_norm, 75, axis=0)                            # (N_spec,)
    contrasts = (continuum - np.min(ccf_norm, axis=0)) / (continuum + 1e-20)  # (N_spec,)
    offsets = continuum                                                          # (N_spec,)

    # ---- Equivalent width ----
    ew = np.mean(1.0 - ccf_norm, axis=0)                                       # (N_spec,)

    # ---- Fractional pixel offset per spectrum ----
    # x_centered[:, i] = vrad_kms - centers[i]
    # pixel 0 of ccf_norm.T corresponds to vrad_kms[0]
    # so pixel index of position x is: x/dv_kms + N_vel/2 + pixel_offset
    pixel_offset = centers / dv_kms                                            # (N_spec,)

    # ---- Parabolic center — interpolate onto common bisspan grid ----
    x_common = np.linspace(
        -ccf_config.bis_range,
        ccf_config.bis_range,
        50,
    )                                                                           # (50,)

    x_common_pixels = (
        x_common[:, np.newaxis] / dv_kms
        + N_vel / 2
        + pixel_offset[np.newaxis, :]
    )                                                                           # (50, N_spec)

    row_coords = np.broadcast_to(
        np.arange(N_spec)[np.newaxis, :],
        x_common_pixels.shape,
    )                                                                           # (50, N_spec)

    y_common = map_coordinates(
        ccf_norm.T,                                                             # (N_spec, N_vel)
        [row_coords, x_common_pixels],
        order=3,
        mode='constant',
        cval=np.nan,
    )                                                                       # (50, N_spec)

    # Fit parabola to all spectra at once
    coeffs = np.polyfit(x_common, np.nan_to_num(y_common), 2)                 # (3, N_spec)
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    x_v = -b / (2 * a)                                                         # (N_spec,)
    y_v = a * x_v**2 + b * x_v + c                                            # (N_spec,)

    para_centers = x_v + centers                                               # (N_spec,)
    para_depths = y_v                                                           # (N_spec,)

    # ---- Bisspan — interpolate onto symmetric core grid centered on RV ----
    step = dv / 1000
    half_grid = np.arange(0, ccf_config.bis_range + step * 0.99, step)
    vrad_center = np.concatenate([-half_grid[1:][::-1], half_grid])            # (N_core,)

    x_bisspan_pixels = (
        vrad_center[:, np.newaxis] / dv_kms
        + N_vel / 2
        + pixel_offset[np.newaxis, :]
        + (rvs - centers)[np.newaxis, :] / dv_kms
    )                                                                           # (N_core, N_spec)

    row_coords_bis = np.broadcast_to(
        np.arange(N_spec)[np.newaxis, :],
        x_bisspan_pixels.shape,
    )                                                                           # (N_core, N_spec)

    y_bisspan = map_coordinates(
        ccf_norm.T,
        [row_coords_bis, x_bisspan_pixels],
        order=3,
        mode='constant',
        cval=np.nan,
    )                                                                         # (N_core, N_spec)

    # Fit parabola to bisspan grid for all spectra at once
    coeffs_bis = np.polyfit(vrad_center, np.nan_to_num(y_bisspan), 2)         # (3, N_spec)
    a_b, b_b = coeffs_bis[0], coeffs_bis[1]
    bisspan = -b_b / (2 * a_b)                                                 # (N_spec,)

    return CCFResults(
        rv           = rvs,
        rv_std       = calibrated_phot_noise['rv'],
        contrast     = contrasts,
        contrast_std = calibrated_phot_noise['contrast'],
        fwhm         = fwhms,
        fwhm_std     = calibrated_phot_noise['fwhm'],
        offset       = offsets,
        offset_std   = np.zeros(N_spec),
        ew           = ew,
        ew_std       = calibrated_phot_noise['ew'],
        center       = para_centers,
        center_std   = calibrated_phot_noise['center'],
        depth        = 1.0 - para_depths,
        depth_std    = calibrated_phot_noise['depth'],
        bisspan      = bisspan,
        bisspan_std  = calibrated_phot_noise['vspan'],
    )

# def _fit_single_spectrum(
#     i: int,
#     vrad: npt.NDArray[np.float64],
#     ccf_power_col: npt.NDArray[np.float64],
#     center: float,
#     ccf_config: CCFConfig,
#     calibrated_phot_noise: dict,
#     beta0: float,
# ) -> CCFResult:

#     vrad_kms = vrad / 1000
#     dv = float(np.median(np.diff(vrad)))

#     # ---- Backup fit on raw CCF before any transformation ----
#     ccf_backup = tableXY(vrad_kms, ccf_power_col, np.sqrt(np.abs(ccf_power_col)))
#     p75_backup = np.nanpercentile(ccf_backup.y, 75)
#     if p75_backup != 0:
#         ccf_backup.y /= p75_backup
#         ccf_backup.yerr /= p75_backup
#     fit_ccf_model(ccf_backup, ccf_config.analytical_model, beta0)

#     # ---- Build main CCF ----
#     ccf = tableXY(vrad_kms, ccf_power_col, np.sqrt(np.abs(ccf_power_col)))

#     # Center
#     ccf.x -= center

#     # Trim
#     trim_ccf(ccf, ccf_config.rv_borders, ccf_config.del_outside_max)

#     # Normalize — after trimming, same as original
#     normalize_ccf(ccf, ccf_config.normalisation)

#     # Fit
#     fit_ccf_model(ccf, ccf_config.analytical_model, beta0)

#     # Consistency check
#     check_ccf_consistency(ccf, ccf_backup, center, ccf_config.check_non_transform)

#     # Extract parameters
#     params = extract_ccf_parameters(ccf, i, center, calibrated_phot_noise)

#     # Diagnostics
#     diagnostics = compute_ccf_diagnostics(
#         ccf, params['rv'], center, dv, ccf_config.bis_range, i, calibrated_phot_noise
#     )

#     return CCFResult({**params, **diagnostics})
