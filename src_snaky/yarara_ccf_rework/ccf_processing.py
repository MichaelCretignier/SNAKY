from collections import namedtuple
import datetime
import logging
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os
from astropy.io import fits
from scipy.ndimage import map_coordinates
from tqdm import tqdm
import glob as glob
import time

from scipy.interpolate import interp1d

from src_snaky.yarara_ccf_rework.ccf_config import CCFConfig
from src_snaky.yarara_ccf_rework.mask_config import MaskConfig
from src_snaky.yarara_ccf_rework.output_config import OutputConfig
from src_snaky.yarara_ccf_rework.stellar_params import StellarParams

from .observation_context import ObservationContext

from .. import snaky_variables as myv
from .. import snaky_functions as myf
from ..snaky_classes import tableXY
from dataclasses import field


def parabola_vertex(a: float, b: float, c: float) -> tuple[float, float]:
    """
    Returns (x_vertex, y_vertex) of parabola ax² + bx + c.
    Used for both parabolic center and bisspan computation.
    """
    x = -b / (2.0 * a)
    return x, a * x**2 + b * x + c


def replace_none(y: float, yerr: float) -> tuple[float, float]:
    """Replace None stderr with a large uncertainty sentinel."""
    if yerr is None:
        return np.nan, 1e6
    return y, yerr

def normalize_ccf_backup(ccf: tableXY) -> tableXY:
    """
    Create a normalized copy of the raw CCF for the initial reference fit.
    Normalizes to the 75th percentile to bring the continuum to ~1.

    Parameters
    ----------
    ccf : tableXY   raw CCF object

    Returns
    -------
    ccf_backup : tableXY   normalized copy, original is untouched
    """
    ccf_backup = ccf.copy()
    p75 = np.nanpercentile(ccf_backup.y, 75)

    # Guard against degenerate continuum
    if p75 == 0:
        return ccf_backup

    ccf_backup.y    /= p75
    ccf_backup.yerr /= p75
    return ccf_backup

def fit_ccf_model(
    ccf:              tableXY,
    analytical_model: str,
    beta0:            float,
    plot:             bool = False,
) -> None:
    """
    Fit a Gaussian or GND model to a CCF object in place.
    Modifies ccf.params directly via lmfit.

    Parameters
    ----------
    ccf              : tableXY   CCF to fit (modified in place)
    analytical_model : str       'gaussian' or 'GND{beta}'
    beta0            : float     fixed beta for GND fit
    plot             : bool      whether to plot the fit
    """
    if analytical_model == 'gaussian':
        ccf.fit_gaussian(Plot=plot)
    else:
        ccf.fit_GND(Plot=plot, beta_fixed=int(beta0))


def resolve_model_parametric(analytical_model: str, beta0: float) -> str:
    """
    Return the string label for the fitted model.
    Used for persistence and logging.
    """
    return 'GND2.0' if analytical_model == 'gaussian' else f'GND{beta0:.1f}'


def find_ccf_center(ccf: tableXY, fwhm: float) -> float:
    """
    Locate the CCF minimum robustly using derivative peak detection
    to identify the two line shoulders.

    Inverts the CCF temporarily to use find_max on the absorption line,
    then restores it.

    Parameters
    ----------
    ccf  : tableXY   CCF object (restored to original sign on return)
    fwhm : float     expected FWHM in km/s — used to select center strategy

    Returns
    -------
    center : float   velocity of the CCF minimum in km/s
    """
    ccf.y    *= -1
    ccf.yerr  = np.sqrt(np.abs(ccf.y))
    ccf.find_max(vicinity=5)

    # Derivative peak detection — relax vicinity until two peaks are found
    ccf.diff(replace=False)
    ccf.deri.y = np.abs(ccf.deri.y)
    for vicinity in (4, 3, 2):
        ccf.deri.find_max(vicinity=vicinity)
        if len(ccf.deri.x_max) > 1:
            break

    sorted_peaks = np.argsort(ccf.deri.y_max)

    first_max    = ccf.deri.x_max[sorted_peaks[-1]]
    second_max   = ccf.deri.x_max[sorted_peaks[-2]]


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
    """
    Trim the CCF to the fitting window in place.

    Two strategies:
    - Standard : keep ±rv_borders around the center
    - del_outside_max : keep only between the two shoulder maxima

    Parameters
    ----------
    ccf            : tableXY   CCF centered on 0 (modified in place)
    rv_borders     : float     half-width of the fitting window in km/s
    del_outside_max: bool      if True, trim between shoulder maxima instead
    """
    if not del_outside_max:
        window = (ccf.x > -rv_borders) & (ccf.x < rv_borders)
        ccf.supress_mask(window)
    else:
        ccf.find_max(vicinity=10)
        ccf.index_max = np.sort(ccf.index_max)
        window        = np.zeros(len(ccf.x), dtype=bool)
        window[ccf.index_max[0]: ccf.index_max[1] + 1] = True
        ccf.supress_mask(window)


def normalize_ccf(ccf: tableXY, normalisation: str) -> None:
    """
    Normalize the CCF continuum to 1 in place.

    Two strategies:
    - 'left'  : divide by the leftmost (bluest) continuum value
    - 'slope' : fit a linear slope between the two continuum peaks

    Parameters
    ----------
    ccf           : tableXY   trimmed CCF (modified in place)
    normalisation : str       'left' or 'slope'
    """
    if normalisation == 'left':
        norm = ccf.y[0]
    else:
        half  = len(ccf.y) // 2
        max1  = np.argmax(ccf.y[:half])
        max2  = np.argmax(ccf.y[half:]) + half
        slope = (ccf.y[max2] - ccf.y[max1]) / (max2 - max1)
        norm  = slope * (np.arange(len(ccf.y)) - max2) + ccf.y[max2]

    # Guard against degenerate continuum
    norm = np.where(norm != 0, norm, 1.0)

    ccf.y    /= norm
    ccf.yerr /= norm


# ---------------------------------------------------------------------------
# SUB-BLOCK 5 — CONSISTENCY CHECK & PARAMETER EXTRACTION
# ---------------------------------------------------------------------------

def check_ccf_consistency(
    ccf:                tableXY,
    ccf_backup:         tableXY,
    center:             float,
    check_non_transform: bool,
) -> None:
    """
    Compare fit results before and after centering/normalisation.
    If they disagree by more than 1 km/s, fall back to the backup fit.
    Modifies ccf.params in place if fallback is triggered.

    Parameters
    ----------
    ccf                 : tableXY   fitted transformed CCF
    ccf_backup          : tableXY   fitted raw CCF (reference)
    center              : float     velocity offset applied to ccf.x
    check_non_transform : bool      whether to perform the check
    """
    ccf_backup.params['cen'].value -= center

    if not check_non_transform:
        return

    V1 = ccf_backup.params['cen'].value
    V2 = ccf.params['cen'].value

    if abs(V1 - V2) > 1:
        logger.warning(
            f'Discrepancy detected between CCFs (*{V1:.4f}*/*{V2:.4f}* km/s), '
            f'reverting to non-transformed fit'
        )
        ccf.params = ccf_backup.params


def extract_ccf_parameters(
    ccf:                    tableXY,
    i:                      int,
    center:                 float,
    calibrated_phot_noise:  dict,
) -> dict:
    """
    Extract RV, contrast, FWHM and offset from the fitted CCF params.
    Stderrs are replaced by calibrated photon noise uncertainties.

    Parameters
    ----------
    ccf                   : tableXY   fitted CCF with .params populated
    i                     : int       spectrum index into calibrated_phot_noise
    center                : float     velocity offset to add back to RV
    calibrated_phot_noise : dict      per-observable calibrated uncertainties

    Returns
    -------
    dict with keys: rv, rv_std, contrast, contrast_std,
                    fwhm, fwhm_std, offset, offset_std
    """
    rv_ccf,       rv_ccf_std       = replace_none(ccf.params['cen'].value + center,  ccf.params['cen'].stderr)
    contrast_ccf, contrast_ccf_std = replace_none(-ccf.params['amp'].value,           ccf.params['amp'].stderr)
    wid_ccf,      wid_ccf_std      = replace_none(ccf.params['wid'].value,             ccf.params['wid'].stderr)
    offset_ccf,   offset_ccf_std   = replace_none(ccf.params['offset'].value,          ccf.params['offset'].stderr)

    return {
        'rv':           rv_ccf,
        'rv_std':       calibrated_phot_noise['rv'][i],        # override with calibrated noise
        'contrast':     contrast_ccf,
        'contrast_std': calibrated_phot_noise['contrast'][i],
        'fwhm':         wid_ccf,
        'fwhm_std':     calibrated_phot_noise['fwhm'][i],
        'offset':       offset_ccf,
        'offset_std':   offset_ccf_std,
    }


_BISSPAN_FALLBACK_WINDOWS = (0.5, 2.0, 5.0)  # km/s — progressive fallback windows

def clip_ccf_for_bisspan(ccf: tableXY, bis_range: float) -> None:
    """
    Clip the CCF to ±bis_range for bisspan computation.
    Falls back to progressively wider windows if fewer than 5 points remain.
    Populates ccf.clipped in place.

    Parameters
    ----------
    ccf       : tableXY   centered, normalized CCF
    bis_range : float     initial half-width in km/s
    """
    for window in (bis_range, *_BISSPAN_FALLBACK_WINDOWS):
        ccf.clip(min=[-window, None], max=[window, None], replace=False)
        if len(ccf.clipped.x) >= 5:
            if window != bis_range:
                logger.info(f'BISSPAN window expanded to *±{window}* km/s')
            return

    logger.warning('Could not find enough points for BISSPAN even at ±5 km/s')


def compute_parabolic_center(ccf: tableXY, center: float) -> tuple[float, float]:
    """
    Fit a parabola to the clipped CCF and return the vertex.

    Parameters
    ----------
    ccf    : tableXY   CCF with .clipped populated
    center : float     velocity offset to add back to the parabolic center

    Returns
    -------
    para_center : float   parabolic RV in km/s
    para_depth  : float   CCF depth at the parabolic center
    """
    ccf.clipped.fit_poly()
    a, b, c     = ccf.clipped.poly_coefficient
    x_v, y_v   = parabola_vertex(a, b, c)
    return x_v + center, y_v


def compute_bisspan(ccf: tableXY, rv_ccf: float, dv: float, bis_range: float) -> float:
    """
    Compute the bisector velocity span by fitting a parabola
    to the CCF core centered on the RV.

    Parameters
    ----------
    ccf      : tableXY   centered, normalized CCF
    rv_ccf   : float     fitted RV in km/s
    dv       : float     CCF velocity step in m/s
    bis_range: float     half-width of the bisspan window in km/s

    Returns
    -------
    bisspan : float   bisector velocity span in km/s
    """
    ccf_core = ccf.copy()

    if rv_ccf == rv_ccf:  # guard NaN
        ccf_core.x -= rv_ccf

    step        = dv / 1000                                              # m/s -> km/s
    half_grid   = np.arange(0, bis_range + step * 0.99, step)
    vrad_center = np.concatenate([-half_grid[1:][::-1], half_grid])     # symmetric grid

    ccf_core.interpolate(new_grid=vrad_center, replace=True, method='cubic')
    ccf_core.fit_poly()

    a, b, _ = ccf_core.poly_coefficient
    x_v, _  = parabola_vertex(a, b, _)
    return x_v


def compute_equivalent_width(ccf: tableXY) -> float:
    """
    Compute the equivalent width of the CCF as the mean line depth.

    EW = mean(1 - ccf.y)

    Parameters
    ----------
    ccf : tableXY   normalized CCF

    Returns
    -------
    ew : float
    """
    return float(np.mean(1.0 - ccf.y))


def compute_ccf_diagnostics(
    ccf:                   tableXY,
    rv_ccf:                float,
    center:                float,
    dv:                    float,
    bis_range:             float,
    i:                     int,
    calibrated_phot_noise: dict,
) -> dict:
    """
    Compute bisspan, equivalent width and parabolic center for a single CCF.

    Parameters
    ----------
    ccf                   : tableXY   centered, normalized, trimmed CCF
    rv_ccf                : float     fitted RV in km/s
    center                : float     velocity offset applied earlier
    dv                    : float     CCF velocity step in m/s
    bis_range             : float     bisspan half-width in km/s
    i                     : int       spectrum index
    calibrated_phot_noise : dict      per-observable calibrated uncertainties

    Returns
    -------
    dict with keys: ew, ew_std, center, center_std,
                    depth, depth_std, bisspan, bisspan_std
    """
    clip_ccf_for_bisspan(ccf, bis_range)
    para_center, para_depth = compute_parabolic_center(ccf, center)
    ew                      = compute_equivalent_width(ccf)
    bisspan                 = compute_bisspan(ccf, rv_ccf, dv, bis_range)

    return {
        'ew':          ew,
        'ew_std':      calibrated_phot_noise['ew'][i],
        'center':      para_center,
        'center_std':  calibrated_phot_noise['center'][i],
        'depth':       1.0 - para_depth,
        'depth_std':   calibrated_phot_noise['depth'][i],
        'bisspan':     bisspan,
        'bisspan_std': calibrated_phot_noise['vspan'][i],
    }


def process_single_ccf(
    i:                     int,
    vrad:                  np.ndarray,
    ccf_power_col:         np.ndarray,
    ccf_power_std_col:     np.ndarray,
    dv:                    float,
    stellar_params:        StellarParams,
    ccf_config:            CCFConfig,
    output_config:         OutputConfig,
    calibrated_phot_noise: dict,
    beta0:                 float,
) -> dict:
    """
    Full processing pipeline for a single CCF spectrum.

    Parameters
    ----------
    i                     : int          spectrum index
    vrad                  : (V,)         velocity axis in m/s
    ccf_power_col         : (V,)         CCF flux for this spectrum
    ccf_power_std_col     : (V,)         CCF flux uncertainty
    dv                    : float        velocity step in m/s
    stellar_params        : StellarParams
    ccf_config            : CCFConfig
    output_config         : OutputConfig
    calibrated_phot_noise : dict
    beta0                 : float        fitted GND beta

    Returns
    -------
    dict merging parameter extraction and diagnostics results
    """
    ccf        = tableXY(vrad / 1000, ccf_power_col, ccf_power_std_col)
    ccf_backup = normalize_ccf_backup(ccf)

    # Initial fit on raw CCF for consistency check reference
    fit_ccf_model(ccf_backup, ccf_config.analytical_model, beta0, plot=output_config.debug)

    if output_config.debug:
        plt.figure('debug')
        ccf_backup.plot()
        plt.close('debug')

    # Center detection
    center  = find_ccf_center(ccf, stellar_params.fwhm)
    ccf.x  -= center

    # Trim, normalize, refit
    trim_ccf(ccf, ccf_config.rv_borders, ccf_config.del_outside_max)
    normalize_ccf(ccf, ccf_config.normalisation)
    fit_ccf_model(ccf, ccf_config.analytical_model, beta0)

    if output_config.debug:
        ccf.plot(color=None)

    # Consistency check — may replace ccf.params with backup
    check_ccf_consistency(ccf, ccf_backup, center, ccf_config.check_non_transform)

    # Parameter extraction
    params      = extract_ccf_parameters(ccf, i, center, calibrated_phot_noise)

    # Diagnostics
    diagnostics = compute_ccf_diagnostics(
        ccf, params['rv'], center, dv, ccf_config.bis_range, i, calibrated_phot_noise
    )

    return {**params, **diagnostics}
