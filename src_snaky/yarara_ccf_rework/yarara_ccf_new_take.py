from collections import namedtuple
import datetime
import logging
import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.pylab as plt
import os
from astropy.io import fits
from scipy.ndimage import map_coordinates
from scipy.interpolate import interp1d
import glob as glob
import time

from scipy.interpolate import interp1d

from src_snaky.snaky_main import replace_none
from src_snaky.yarara_ccf_rework.ccf_config import CCFConfig
from src_snaky.yarara_ccf_rework.ccf_processing import process_all_ccf, stack_ccf_results
from src_snaky.yarara_ccf_rework.mask_config import MaskConfig
from src_snaky.yarara_ccf_rework.output_config import OutputConfig
from src_snaky.yarara_ccf_rework.stellar_params import StellarParams
from src_snaky.yarara_ccf_rework.observation_context import ObservationContext


from src_snaky import snaky_variables as myv
from src_snaky import snaky_functions as myf
from src_snaky import snaky_classes as myc

from dataclasses import field

import inspect

logger = logging.getLogger('snaky')

PHOT_NOISE_CALIBRATION = {
    'rv': (0.98, -3.08),
    'contrast': (0.98, -3.58),
    'fwhm': (0.98, -2.94),
    'center': (0.98, -2.83),
    'depth': (0.97, -3.62),
    'ew': (0.97, -3.47),
    'vspan': (0.98, -2.95),
}

FALLBACK_NOISE = 0.01  # module-level constant

''' This should be computed before calling the function. The function  always require a complete CCFConfig to work properly. The caller should ensure to pass all data
T = TypeVar('T')
def coalesce(value: Optional[T], default: T, on_default_message: str) -> T:
    if value is not None:
        return value
    else:
        logger.info(on_default_message.format(default=default))
        return default

def get_analytical_model(model: str, star_beta_gnd: float):
    if model != 'auto':
        return model

    if star_beta_gnd > 2.5:
        return f'GND{star_beta_gnd:.1f}'

    return 'gaussian'

logger.info(f'FWHM: {star.fwhm:.2f} kms')
rv_range = coalesce(rv_range, 3 * star.fwhm, 'RV range updated to : {default:.1f} kms')
rv_borders = coalesce(rv_borders, int(2*star.fwhm), 'RV borders updated to : {default:.1f} kms')
bis_range = coalesce(bid_range, np.round(0.33*star.fwhm,1), 'BISSPAN borders updated to : {default:.1f} kms')
analytical_model = get_analytical_model('mymodel', 2.1)
logger.info(f'CCF analytical model :{analytical_model}')'''

def doppler_r(lamb, v):
    """Relativistic Doppler. Takes (wavelength, velocity in [m/s]) and returns lambda observed and lambda source."""
    c = 299.792e6
    factor = np.where(v != 0, np.sqrt((1 + v / c) / (1 - v / c)), 1.0)
    return lamb * factor, lamb / factor

def interpolate_rv_shift(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    rv_shift: npt.NDArray[np.float64],
    xnew: npt.NDArray[np.float64] | None = None,
    fill_value: float = 0,
    kind: str = 'linear',
) -> npt.NDArray[np.float64]:

    if xnew is None:
        xnew = x.copy()

    # Shift all grids at once — (N_spec, N_pix)
    shifted_grids = doppler_r(x[np.newaxis, :], rv_shift[:, np.newaxis])[1]

    # Find insertion indices for all spectra at once — (N_spec, N_pix)
    idx = np.searchsorted(xnew, shifted_grids)
    idx = np.clip(idx, 1, len(xnew) - 1)

    # Linear interpolation weights
    lo = xnew[idx - 1]                              # (N_spec, N_pix)
    hi = xnew[idx]                                  # (N_spec, N_pix)
    t  = (shifted_grids - lo) / (hi - lo + 1e-20)  # (N_spec, N_pix)

    # Gather values at lo and hi indices for all spectra
    # y shape: (N_spec, N_pix)
    spec_idx = np.arange(len(y))[:, np.newaxis]     # (N_spec, 1)
    y_lo = y[spec_idx, idx - 1]                     # (N_spec, N_pix)
    y_hi = y[spec_idx, idx]                         # (N_spec, N_pix)

    # Interpolated values
    result = (1 - t) * y_lo + t * y_hi              # (N_spec, N_pix)

    # Apply fill_value outside bounds
    out_of_bounds = (shifted_grids < xnew[0]) | (shifted_grids > xnew[-1])
    result[out_of_bounds] = fill_value

    return result.astype(y.dtype)

    return map_coordinates(
        y,
        [row_coords, pixel_coords],
        order=1 if kind == 'linear' else 3,
        mode='constant',
        cval=fill_value,
    ).astype(y.dtype)

ImportSpectrumReturn = namedtuple("ImportSpectrumReturn", ("grid", "flux"))
def import_spectrums(files, rv_shift:np.ndarray, scale=True) -> ImportSpectrumReturn:
    if scale:
        wave_grid = np.round(files[0]/100.,2)
        spectrums = (files[1]/10000.).astype('float32')
    else:
        wave_grid = files[0]
        spectrums = files[1]

    if rv_shift is not None and np.any(rv_shift != 0):
        spectrums = interpolate_rv_shift(wave_grid, spectrums, rv_shift=rv_shift, fill_value=1, kind='linear')

    return ImportSpectrumReturn(wave_grid, spectrums)

CCF_GRID_MARGIN_KMS = 30_000 # safety margin

def filter_mask_to_grid(
    lines: np.ndarray,
    grid: np.ndarray,
    wave_min: float,
    wave_max: float,
    margin_kms: float = CCF_GRID_MARGIN_KMS
) -> np.ndarray:
    blue_shifted, red_shifted = doppler_r(lines[:, 0], margin_kms)

    within_grid = (blue_shifted < grid.max()) & (red_shifted > grid.min())
    within_range = (lines[:, 0] > wave_min) & (lines[:, 0] < wave_max)

    return lines[within_grid & within_range]

GridTrimmingReturn = namedtuple("GridTrimmingReturn", ("grid", "flux"))
def trim_grid_to_mask(
        grid: np.ndarray,
        flux: np.ndarray,
        lines: np.ndarray,
        margin_kms: float = 100_000
) -> GridTrimmingReturn:
    low = doppler_r(lines[:, 0].min(), -margin_kms)[0]
    high = doppler_r(lines[:, 0].max(),  margin_kms)[0]
    logger.info(f'Wave min : {low:.0f} AA | Wave max :{high:.0f} AA')

    #supress useless part of the spectra to speed up the CCF
    keep = (grid >= low) & (grid <= high)
    return GridTrimmingReturn(grid[keep], flux[:, keep])

StaticMaskReturn = namedtuple("StaticMaskReturn", ("log_grid_mask", "log_mask"))
def generate_static_mask(
    mask: np.ndarray,
    weighted: bool,
    delta_window: int,
    grid: np.ndarray,
    dgrid: np.float64,
) -> StaticMaskReturn:
    mask_wave = np.log10(mask[:,0])
    mask_contrast = mask[:,1]*weighted + (1-weighted)

    grid_mask = np.arange(grid.min()-10*dgrid,grid.max()+10*dgrid+dgrid/10,dgrid/11)

    # region handles log mask
    log_mask = np.zeros(len(grid_mask))
    match = myf.identify_nearest(mask_wave,grid_mask)
    offsets = np.arange(-delta_window, delta_window + 1, dtype=int)
    indices = (match[:, np.newaxis] + offsets[np.newaxis, :]).ravel()
    values = np.broadcast_to(mask_contrast[:, np.newaxis], (len(match), len(offsets))).ravel()
    log_mask[indices] = values
    # endregion

    return StaticMaskReturn(grid_mask, log_mask)

def save_static_mask(
    log_grid_mask: np.ndarray,
    log_mask: np.ndarray,
    path: str,
) -> None:
    hdu = fits.PrimaryHDU(np.array([log_grid_mask, log_mask]).T)
    list = fits.HDUList([hdu])
    list.writeto(path)
    logger.info(f'CCF mask saved under : { path }')

def yarara_ccf(
    observations: ObservationContext,
    star: StellarParams,
    mask_config: MaskConfig,
    ccf_config: CCFConfig,
    output_config: OutputConfig = field(default_factory=OutputConfig),
):

    start = time.time()

    mask_config.mask[:,0] = doppler_r(mask_config.mask[:,0], star.rv_sys)[0]

    if observations.spectra is None:
        grid_base, flux_base = import_spectrums(observations.files, rv_shift=observations.rv_shift)
    else:
        grid_base, flux_base = observations.spectra
    elapsed = time.perf_counter() - start

    flux_err = None

    logger.info('Reference color : flat normalised continuum')

    mask = filter_mask_to_grid(
        mask_config.mask,
        grid_base,
        mask_config.wave_min,
        mask_config.wave_max
    )
    logger.info(f'Nb lines in the mask : { len(mask) }')

    grid, flux = trim_grid_to_mask(
            grid_base,
            flux_base,
            mask
    )

    log_grid = np.log10(np.geomspace(grid[0], grid[-1], len(grid)))
    dgrid = log_grid[1] - log_grid[0]
    #dv = (10**(dgrid)-1)*299.792e6

    #computation of region free of spectral line to increase code speed
    #used_region = ((10**log_grid)>=mask_shifted[1][:,np.newaxis])&((10**log_grid)<=mask_shifted[0][:,np.newaxis])
    #used_region = (np.sum(used_region,axis=0)!=0).astype('bool')
    #logger.info('Percentage of the spectrum used : %.1f [%%] (%.0f)'%(100*sum(used_region)/len(grid),len(grid)))

    ccf_mask_path = f'{observations.dir_root}CCF_MASK/CCF_{mask_config.name.split('.')[0]}.fits'

    if not os.path.exists(ccf_mask_path):
        logger.info('CCF mask reduced for the first time, wait for the static mask production...')
        log_grid_mask, log_mask = generate_static_mask(
            mask,
            mask_config.weighted,
            mask_config.delta_window,
            log_grid,
            dgrid
        )

        if output_config.debug:
            plt.figure()
            plt.plot(10**log_grid_mask,log_mask)

        save_static_mask(log_grid_mask, log_mask, ccf_mask_path)

    else:
        logger.debug(f'CCF mask found : { ccf_mask_path }')
        log_grid_mask, log_mask = fits.open(ccf_mask_path)[0].data.T

    log_template = interp1d(
        log_grid_mask,
        log_mask ** (1.0 + float(mask_config.squared)),
        bounds_error=False,
        fill_value=0,
    )(log_grid)

    amplitudes = []
    amplitudes_std = []
    rvs = []
    rvs_std = []
    fwhms = []
    fwhms_std = []
    ew = []
    ew_std = []
    centers = []
    centers_std = []
    depths = []
    depths_std = []
    bisspan = []
    bisspan_std = []

    now = datetime.datetime.now()
    logger.info(f'Computing CCFs (Current time {now.strftime('%H:%M:%S')})')

    # Replaces the chunking Might be wrong as the workflow is totally different
    grid_log10 = np.log10(grid)

    pixel_coords = np.interp(log_grid, grid_log10, np.arange(len(grid_log10)))
    row_coords = np.arange(len(flux))[:, np.newaxis] * np.ones(len(log_grid))
    col_coords = np.ones(len(flux))[:, np.newaxis] * pixel_coords[np.newaxis, :]

    flux = map_coordinates(
        flux,
        [row_coords, col_coords],
        order=myv.INTERP_ORDER[myv.interp_degree],
        mode='constant',
        cval=0.0,
    ).astype(flux.dtype)

    gravity_center_wave = np.sum(10**log_grid*log_template)/np.sum(log_template)

    logger.info(f'Gravity center wavelength = {gravity_center_wave:.0f} AA')

    vrad, ccf_power, ccf_power_std = myf.ccf(
        log_grid,
        flux,
        log_template,
        rv_range = ccf_config.rv_range,
        oversampling = ccf_config.oversampling,
        spec1_std = flux_err
    ) #to compute on all the ccf simultaneously

    end = time.time()

    logger.debug(f"Line number: {inspect.currentframe().f_lineno}")
    #logger.debug(f"Execution time {counter_dev}: {end - start:.3f} seconds")

    now = datetime.datetime.now()
    dv = np.median(np.diff(vrad))

    logger.debug(f'CCFs computed (Current time {now.strftime('%H:%M:%S')})')
    logger.info(f'CCF velocity step : {dv:.0f} m/s')

    ccf_ref = np.median(ccf_power,axis=1)

    if ccf_config.continuum_method=='flux':
        continuum_idx = np.argmax(ccf_ref)
        n_top = len(ccf_ref) // 2
        top_ccf = np.sort(np.argpartition(ccf_ref, -n_top)[-n_top:]) #roughly half of a CCF is made of the continuum
    else:
        continuum_idx = np.argmax(abs(vrad))
        n_top = len(ccf_ref) // 2
        top_ccf = np.sort(np.argpartition(np.abs(vrad), -n_top)[-n_top:]) #roughly half of a CCF is made of the continuum

    master_ccf = ccf_ref/np.max(ccf_ref)
    master_ccf = myc.tableXY(vrad/1000, master_ccf, 0.01*np.ones(len(master_ccf)))

    try:
        master_ccf.fit_GND(beta_fixed=0,Plot=False)
        beta_param = master_ccf.params['beta'].value
        beta0 = beta_param if type(beta_param) is float else 2.0
    except:
        beta0 = 2.0

    logger.info(f'Beta value of GND = { beta0:.2f}')

    if ((beta0 > 2.5) and (ccf_config.analytical_model=='gaussian')):
        logger.warning('Significant Kurtosis detected.')

    continuum_idx = np.argmax(ccf_ref)
    continuum_level = np.mean(ccf_power[continuum_idx])

    ccf_continuum_residuals = (ccf_power[top_ccf] - ccf_ref[top_ccf, np.newaxis]) / continuum_level * 100
    ccf_continuum_residuals -= np.median(ccf_continuum_residuals, axis=0)

    ccf_signal_noise_ratio = 100.0 / np.std(ccf_continuum_residuals, axis=0)

    logger.info(f'SNR CCF continuum median : {np.median(ccf_signal_noise_ratio):.0f}')

    # Noise profile — normalised by continuum level and CCF SNR
    ccf_norm_sqrt = np.sqrt(ccf_ref / np.max(ccf_ref)) * ccf_ref[continuum_idx]
    noise_ccf = ccf_norm_sqrt[:, np.newaxis] / ccf_signal_noise_ratio

    # RV uncertainty via optimal weighting — steeper gradient = better precision
    vrad_step = np.gradient(vrad)
    ccf_gradient = np.abs(np.gradient(ccf_ref)) / vrad_step
    sigma_rv = noise_ccf / (ccf_gradient[:, np.newaxis] + 1e-20)

    # Photon noise RV precision
    w_rv = (ccf_gradient[:, np.newaxis] / noise_ccf) ** 2
    svrad_phot = 1.0 / np.sqrt(np.sum(w_rv, axis=0))

    # Penalize oversampling in vrad
    svrad_phot *= np.sqrt(820 / np.mean(vrad_step))
    svrad_phot[svrad_phot==0] = 2*np.max(svrad_phot)

    logger.info(f'Photon noise RV median : {np.median(svrad_phot):.2f} m/s\n ')

    # Compute all calibrated uncertainties in one pass
    log_svrad = np.log10(svrad_phot)
    calibrated_phot_noise = {
        obs: 10 ** (slope * log_svrad + intercept)
        for obs, (slope, intercept) in PHOT_NOISE_CALIBRATION.items()
    }

    logger.info(f'Photon noise RV from calibration : {np.median(calibrated_phot_noise['rv'])*1000:.2f} m/s ')

    logger.info(f'Number of velocity bin ={len(vrad)}')

    nonzero_mask = noise_ccf != 0
    nonzero_mean = np.mean(noise_ccf[nonzero_mask]) if nonzero_mask.any() else FALLBACK_NOISE
    noise_ccf = np.where(nonzero_mask, noise_ccf, nonzero_mean)

    noise_75th = np.percentile(noise_ccf, 75, axis=0)
    factor = 1.0 / noise_75th ** 2

    ccf_power = ccf_power * factor[np.newaxis, :]
    ccf_power_std = ccf_power_std * factor[np.newaxis, :]

    # TBD optimize take 9s for N=360
    # results = stack_ccf_results([
    #     process_single_ccf(
    #         i,
    #         vrad,
    #         ccf_power[:,i],
    #         ccf_power_std[:, i],
    #         dv,
    #         star,
    #         ccf_config,
    #         output_config,
    #         calibrated_phot_noise,
    #         beta0
    #     )
    #     for i, _ in enumerate(observations.files[-1])
    # ])

    results = process_all_ccf(
        vrad,
        ccf_power,
        dv,
        ccf_config,
        calibrated_phot_noise
    )

    if ccf_config.analytical_model=='gaussian':
        model_parametric = 'GND2.0'
    else:
        model_parametric = f'GND{beta0}.1f'

    rvs_std = calibrated_phot_noise['rv']
    fwhms = results["fwhm"] * 2.355
    fwhms_std = results["fwhm_std"] * 2.355

    warning_rv_borders = False
    if np.median(fwhms)>(ccf_config.rv_borders/1.5):
        logger.warning('The CCF is larger than the RV borders for the fit')
        warning_rv_borders = True

    jdb = observations.jdb if observations.jdb is not None else np.arange(len(mask_config.files[-1]))
    ccf_rv = myc.tableXY(jdb,np.array(results["rv"])*1000,np.array(results["rv_std"])*1000)
    ccf_centers = myc.tableXY(jdb,np.array(results["center"])*1000,np.array(results["center_std"])*1000)
    ccf_contrast = myc.tableXY(jdb,np.array(results["contrast"])*100,np.array(results["contrast_std"])*100)
    ccf_depth = myc.tableXY(jdb,depths,depths_std)
    ccf_fwhm = myc.tableXY(jdb,fwhms,fwhms_std)
    ccf_vspan = myc.tableXY(jdb,np.array(results["bisspan"])*1000,np.array(results["bisspan_std"])*1000)
    ccf_ew = myc.tableXY(jdb,np.array(results["ew"]),np.array(results["ew_std"]))
    ccf_timeseries = np.array([
        results["ew"],
        results["ew_std"],
        results["contrast"],
        results["contrast_std"],
        results["rv"],
        results["rv_std"],
        calibrated_phot_noise['rv'],
        results["fwhm"],
        results["fwhm_std"],
        results["center"],
        results["center_std"],
        results["depth"],
        results["depth_std"],
        results["bisspan"],
        results["bisspan_std"]
    ])
    ccf_infos = pd.DataFrame(ccf_timeseries.T,columns=['ew','ew_std','contrast','contrast_std','rv','rv_std','rv_std_phot','fwhm','fwhm_std','center','center_std','depth','depth_std','bisspan','bisspan_std'])
    ccf_infos['jdb'] = jdb
    ccf_infos['filename'] = observations.files[-1]

    #Update to remove nan value in RV 02.05.25
    ccf_rv.yerr[ccf_rv.y!=ccf_rv.y] = np.nanmedian(ccf_rv.yerr[ccf_rv.y!=ccf_rv.y])
    offset = np.nanmedian(ccf_centers.y - ccf_rv.y)
    ccf_rv.y[ccf_rv.y!=ccf_rv.y] = ccf_centers.y[ccf_rv.y!=ccf_rv.y] - offset

    ccf_infos = {'table':ccf_infos,'model_parametric':model_parametric,'weighting':1.0+float(mask_config.squared),'creation_date':datetime.datetime.now().isoformat()}

    file_summary_ccf = myf.touch_pickle(observations.dir_root+'WORKSPACE/Analyse_ccf.p')
    file_summary_ccf['CCF_'+mask_config.name.split('.')[0]] = ccf_infos

    myf.pickle_dump(file_summary_ccf,open(observations.dir_root+'WORKSPACE/Analyse_ccf.p','wb'))

    ccf_norm = (ccf_power.T/np.percentile(ccf_power,75,axis=0)[:,np.newaxis]).T
    ccf_shifted = ccf_norm.copy()
    rvs = ccf_rv.y
    #rvs = rvs - np.nanmedian(rvs)
    for n,rv in enumerate(rvs):
        if rv==rv:
            profile = myc.tableXY(vrad-rv,ccf_shifted[:,n],0*vrad)
            profile.interpolate(new_grid=vrad,method='linear')
            ccf_shifted[:,n] = profile.y
    master_ccf = np.nanmean(ccf_shifted,axis=1)
    ccf_res = ccf_norm - np.nanmedian(ccf_norm,axis=1)[:,np.newaxis]

    ccf_name = mask_config.name

    export = myf.touch_pickle(observations.dir_root+'WORKSPACE/Analyse_ccf_saved.p')
    export['CCF_'+ccf_name] = {}
    export['CCF_'+ccf_name][observations.sub_dico] = {'ccf_vrad':vrad,'ccf_flux':ccf_norm,'ccf_shifted':ccf_shifted,'ccf_master':master_ccf,'filename':observations.files[-1]}
    myf.pickle_dump(export,open(observations.dir_root+'WORKSPACE/Analyse_ccf_saved.p','wb'))

    warning = 0
    if ccf_name=='mask_telluric_o2':
        fwhm_ins = np.nanmedian(ccf_fwhm.y)
        if observations.ins.split('_')[0] in myv.instrument_res_kms.keys():
            ref = myv.instrument_res_kms[observations.ins.split('_')[0]]
            logger.info(f'Reference value for {observations.ins} is {ref:.1f} km/_s')
            if abs(ref - fwhm_ins)>1:
                warning = 1
                logger.warning(f'Instrumental resolution is not usual ({fwhm_ins:.1f} km/s)')
        else:
            ref = np.nan

    if (ccf_name=='G2')&(np.nanstd(ccf_rv.y)>1000): # SB flag
        warning = 1

    plt.figure(figsize=(9,8))
    plt.axes([0.1,0.72,0.6,0.22])
    med = np.nanmedian(ccf_rv.y)
    ccf_rv.plot() ; plt.ylabel('RV [m/s]') ; plt.axhline(y=med,color='r',label='%.1f'%(med)) ; plt.tick_params(labelbottom=False)
    plt.axes([0.1,0.50,0.6,0.22])
    med = np.nanmedian(ccf_fwhm.y)
    ccf_fwhm.plot() ; plt.ylabel('FWHM [km/s]') ; plt.axhline(y=med,color='r',label='%.2f km/s'%(med)) ; plt.tick_params(labelbottom=False)
    if ccf_name=='mask_telluric_o2':
        plt.axhline(y=ref,color='k',ls='-.',lw=1)
    plt.legend(loc=3)
    plt.axes([0.1,0.28,0.6,0.22])
    med = np.nanmedian(ccf_contrast.y)
    ccf_contrast.plot() ; plt.ylabel('CT [%]') ; plt.axhline(y=med,color='r',label='%.1f %%'%(med)) ; plt.tick_params(labelbottom=False)
    plt.legend(loc=3)
    plt.axes([0.1,0.06,0.6,0.22])
    med = np.nanmedian(ccf_vspan.y)
    ccf_vspan.plot() ; plt.ylabel('VSPAN [m/s]') ; plt.axhline(y=med,color='r',label='%.1f'%(med))
    plt.axes([0.75,0.06,0.22,0.66])
    plt.imshow(ccf_res.T,vmin=-0.005,vmax=0.005,aspect='auto',cmap='seismic') ;
    plt.axvline(x=len(vrad)*0.5,color='k',ls='-.',lw=1)
    plt.axes([0.75,0.72,0.22,0.22])
    plt.plot(vrad/1000,master_ccf,color='k')
    plt.plot(vrad/1000,ccf_norm,alpha=0.2,color='k')
    plt.axvline(x=0,color='k',ls='-.',lw=1)
    plt.tick_params(top=True,labeltop=True,labelbottom=False)
    plt.savefig(observations.dir_root+'IMAGES/CCF_summary_%s'%(ccf_name)+myv.PRD_EXT+'.png')
    if warning:
        plt.savefig(observations.dir_root+'WARNING/CCF_summary_%s'%(ccf_name)+myv.PRD_EXT+'.png')

    output = {
        'rv':ccf_rv,
        'contrast':ccf_contrast,
        'fwhm':ccf_fwhm,
        'vspan':ccf_vspan}

    if output_config.save:
        summary = import_summary(observations.dir_root)
        mask = myf.in1d(np.array(summary['filename']),observations.files[-1])
        summary['ccf_rv_'+ccf_name] = np.nan ; summary.loc[mask,'ccf_rv_'+ccf_name] = np.round(ccf_rv.y,0) # DONT USE RV FROM SNAKY, PRECISION NOT BETTER THAN 3 M/S
        summary['ccf_ct_'+ccf_name] = np.nan ; summary.loc[mask,'ccf_ct_'+ccf_name] = np.round(ccf_contrast.y,4)
        summary['ccf_fwhm_'+ccf_name] = np.nan ; summary.loc[mask,'ccf_fwhm_'+ccf_name] = np.round(ccf_fwhm.y,4)
        summary.to_csv(observations.dir_root+'WORKSPACE/Analyse_summary.csv')

    if output_config.return_ccf:
        return output, vrad, ccf_shifted
    else:
        return output

# observationContext = ObservationContext(dir_root='/data/HD189733/', files=[])
# stellarParams = StellarParams(rv_sys=-2.3, fwhm=7.2, beta_gnd=2.1)
# maskConfig = MaskConfig(mask_input='G2', wave_min=5000, wave_max=6800)


def import_summary(dir_root):
    material = pd.read_csv(dir_root+'WORKSPACE/Analyse_summary.csv',index_col=0)
    return material
