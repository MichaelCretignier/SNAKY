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

from src_snaky.yarara_ccf_rework.ccf_config import CCFConfig
from src_snaky.yarara_ccf_rework.mask_config import MaskConfig
from src_snaky.yarara_ccf_rework.output_config import OutputConfig
from src_snaky.yarara_ccf_rework.stellar_params import StellarParams

from .observation_context import ObservationContext

from .. import snaky_variables as myv
from .. import snaky_functions as myf
from .. import snaky_classes as myc

from dataclasses import field
import numpy as np

logger = logging.getLogger('snaky')

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
    c      = 299.792e6
    factor = np.where(v != 0, np.sqrt((1 + v / c) / (1 - v / c)), 1.0)
    return lamb * factor, lamb / factor

def interpolate_rv_shift(x,y, rv_shift:np.ndarray, xnew=None, fill_value=0,kind='linear'):
    if xnew is None:
        xnew = x.copy()
    if rv_shift is None:
        rv_shift = np.zeros(len(y))

    shifted_grids = doppler_r(x[np.newaxis, :], rv_shift[:, np.newaxis])[1]

    pixel_coords = np.interp(shifted_grids, xnew, np.arange(len(xnew)))

    row_coords = np.arange(len(y))[:, np.newaxis] * np.ones(len(xnew))

    y = map_coordinates(
        y,
        [row_coords, pixel_coords],
        order=1 if kind == 'linear' else 3,
        mode='constant',
        cval=fill_value,
    ).astype(y.dtype)

    return y

ImportSpectrumReturn = namedtuple("ImportSpectrumReturn", ("grid", "flux"))
def import_spectrums(files, rv_shift:np.ndarray, scale=True) -> ImportSpectrumReturn:
    "rv_shift in m/s"

    if scale:
        wave_grid = np.round(files[0]/100.,2)
        spectrums = (files[1]/10000.).astype('float32')
    else:
        wave_grid = files[0]
        spectrums = files[1]

    if rv_shift is not None:
            sts = interpolate_rv_shift(wave_grid, spectrums, rv_shift=rv_shift, fill_value=1, kind='linear')

    return ImportSpectrumReturn(wave_grid, sts)

CCF_GRID_MARGIN_KMS = 30_000 # safety margin

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
        grid, flux = import_spectrums(observations.files, rv_shift=observations.rv_shift)
    else:
        grid, flux = observations.spectra

    flux_err = None

    logger.info('Reference color : flat normalised continuum')

    blue_shifted, red_shifted = doppler_r(mask_config.mask[:,0], CCF_GRID_MARGIN_KMS)
    within_grid = (blue_shifted < grid.max()) & (red_shifted > grid.min())
    mask = mask_config.mask[within_grid] # supresses line farther than 30kms
    mask = mask[mask[:,0] > mask_config.wave_min,:]
    mask = mask[mask[:,0] < mask_config.wave_max,:]

    mask_min = np.min(mask[:,0])
    mask_max = np.max(mask[:,0])

    logger.info(f'Nb lines in the mask : { len(mask):.0f }')
    logger.info(f'Wave min : { mask_min:.0f } AA | Wave max :{ mask_max:.0f } AA')

    #supress useless part of the spectra to speed up the CCF
    grid_min = int(myf.find_nearest(grid,myf.doppler_r(mask_min,-100000)[0])[0][0])
    grid_max = int(myf.find_nearest(grid,myf.doppler_r(mask_max,100000)[0])[0][0])
    grid = grid[grid_min:grid_max]
    flux = flux[:,grid_min:grid_max]
    if flux_err is not None:
        flux_err = flux_err[:,grid_min:grid_max]

    log_grid = np.linspace(np.log10(grid[0]),np.log10(grid[-1]),len(grid))
    dgrid = log_grid[1] - log_grid[0]
    #dv = (10**(dgrid)-1)*299.792e6

    #computation of region free of spectral line to increase code speed
    #used_region = ((10**log_grid)>=mask_shifted[1][:,np.newaxis])&((10**log_grid)<=mask_shifted[0][:,np.newaxis])
    #used_region = (np.sum(used_region,axis=0)!=0).astype('bool')
    #logger.info('Percentage of the spectrum used : %.1f [%%] (%.0f)'%(100*sum(used_region)/len(grid),len(grid)))
    time.sleep(1)

    ccf_mask_path = dir_root+'CCF_MASK/CCF_'+mask_name.split('.')[0]+'.fits'
    if not os.path.exists(ccf_mask_path):
        logger.info('CCF mask reduced for the first time, wait for the static mask production...')
        time.sleep(1)
        mask_wave = np.log10(mask[:,0])
        mask_contrast = mask[:,1]*weighted + (1-weighted)

        log_grid_mask = np.arange(log_grid.min()-10*dgrid,log_grid.max()+10*dgrid+dgrid/10,dgrid/11)
        log_mask = np.zeros(len(log_grid_mask))

        #mask_contrast /= np.sqrt(np.nansum(mask_contrast**2)) #UPDATE 04.05.21 (DOES NOT WORK)

        match = myf.identify_nearest(mask_wave,log_grid_mask)
        for j in np.arange(-delta_window,delta_window+1,1):
            log_mask[match+j] = mask_contrast

        if debug:
            plt.figure()
            plt.plot(10**log_grid_mask,log_mask)

        hdu = fits.PrimaryHDU(np.array([log_grid_mask, log_mask]).T)
        hdul = fits.HDUList([hdu])
        hdul.writeto(ccf_mask_path)
        logger.info(f'CCF mask saved under : { ccf_mask_path }')

        del hdu
        del hdul
    else:
        logger.info(f'CCF mask found : { ccf_mask_path }')
        log_grid_mask, log_mask = fits.open(ccf_mask_path)[0].data.T

    log_template = myf.interpolate_rv_shift(log_grid_mask,log_mask**(1.0+float(squared)),xnew=log_grid,fill_value=0)

    amplitudes = [] ; amplitudes_std = []
    rvs = [] ; rvs_std = []
    fwhms = [] ; fwhms_std = []
    ew = [] ; ew_std = []
    centers = [] ; centers_std = []
    depths = [] ; depths_std = []
    bisspan = []  ; bisspan_std = []

    now = datetime.datetime.now()
    logger.info(f'Computing CCFs (Current time {now.strftime('%H:%M:%S')})')

    chunks = np.array_split(np.arange(len(log_grid)), 5)

    if True:
        grid_log10 = np.log10(grid)
        for idx in tqdm(chunks):
            idx2 = (grid_log10>log_grid[idx[0]])&(grid_log10<log_grid[idx[-1]])
            for j,i in enumerate(files[-1]):
                flux[j][idx] = myf.interpolate_rv_shift(grid_log10[idx2], flux[j][idx2], xnew=log_grid[idx], rv=0, fill_value=0, kind=interp_degree)
        del grid_log10
    else:
        logger.debug('ALGO1')
        # TBD optimize take 9s for N=360
        for j,i in enumerate(files[-1]):
            if flux_err is None:
                f_err = 0*flux[j]
            else:
                f_err = flux_err[j]
            temp = myc.tableXY(np.log10(grid), flux[j], f_err)
            temp.interpolate(new_grid=log_grid,method=interp_degree)
            flux[j] = temp.y
            if flux_err is not None:
                flux_err[j] = temp.yerr

        del f_err

    gravity_center_wave = np.sum(10**log_grid*log_template)/np.sum(log_template)

    logger.info(f'Gravity center wavelength = {gravity_center_wave:.0f} AA')
    #flux = flux[:,used_region]
    #log_grid = log_grid[used_region]
    #log_template = log_template[used_region]
    #if flux_err is not None:
    #    flux_err = flux_err[:,used_region]

    start3 = time.time()
    vrad, ccf_power, ccf_power_std = myf.ccf(log_grid, flux, log_template,
                                                rv_range = rv_range, oversampling = ccf_oversampling, spec1_std = flux_err) #to compute on all the ccf simultaneously

    del log_grid
    del log_mask
    del log_template

    end = time.time()
    if myv.DEV:
        counter_dev+=1
        logger.debug(f"Line number: {inspect.currentframe().f_lineno}")
        logger.debug(f"Execution time {counter_dev}: {end - start:.3f} seconds")

    del flux
    del flux_err

    now = datetime.datetime.now()
    dv = np.median(np.diff(vrad))

    logger.debug(f'CCFs computed (Current time {now.strftime('%H:%M:%S')})')
    logger.info(f'CCF velocity step : {dv:.0f} m/s')

    all_ccf_saved = {ccf_name:(vrad, ccf_power, ccf_power_std)}

    ccf_ref = np.median(ccf_power,axis=1)

    if continuum_method=='flux':
        continuum_ccf = np.argmax(ccf_ref)
        top_ccf = np.sort(np.argsort(ccf_ref)[-int(len(ccf_ref)/2):]) #roughly half of a CCF is made of the continuum
    else:
        continuum_ccf = np.argmax(abs(vrad))
        top_ccf = np.sort(np.argsort(abs(vrad))[-int(len(ccf_ref)/2):]) #roughly half of a CCF is made of the continuum

    master_ccf = ccf_ref/np.max(ccf_ref)
    master_ccf = myc.tableXY(vrad/1000, master_ccf, 0.01*np.ones(len(master_ccf)))

    try:
        master_ccf.fit_GND(beta_fixed=0,Plot=False)
        beta0 = master_ccf.params['beta']
    except:
        beta0 = 2.0

    logger.info(f'Beta value of GND = { beta0:.2f}')
    if (beta0>2.5)&(analytical_model=='gaussian'):
        logger.warning('Significant Kurtosis detected.')

    dccf2 = (ccf_power-ccf_ref[:,np.newaxis])[top_ccf]/np.mean(ccf_power[continuum_ccf])*100
    dccf2 -= np.median(dccf2,axis=0)
    ccf_snr = 1/np.std(dccf2,axis=0)*100
    logger.info(f'SNR CCF continuum median : {np.median(ccf_snr).0f}')

    noise_ccf = (np.sqrt(ccf_ref/np.max(ccf_ref))*ccf_ref[continuum_ccf])[:,np.newaxis]/ccf_snr #assume that the noise in the continuum is white (okay for matching_mad but wrong when tellurics are still there)
    sigma_rv = noise_ccf/(abs(np.gradient(ccf_ref))/np.gradient(vrad))[:,np.newaxis]
    w_rv = (1/sigma_rv)**2
    svrad_phot = 1/np.sqrt(np.sum(w_rv,axis=0))
    scaling = np.sqrt(820/np.mean(np.gradient(vrad))) #to penalize oversampling in vrad
    svrad_phot*=scaling

    svrad_phot[svrad_phot==0] = 2*np.max(svrad_phot) #in case of null values

    logger.info(f'Photon noise RV median : {np.median(svrad_phot):.2f} m/s\n ')

    svrad_phot2 = {}
    svrad_phot2['rv'] = 10**(0.98*np.log10(svrad_phot)-3.08) # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2['contrast'] = 10**(0.98*np.log10(svrad_phot)-3.58) # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2['fwhm'] = 10**(0.98*np.log10(svrad_phot)-2.94) # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2['center'] = 10**(0.98*np.log10(svrad_phot)-2.83) # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2['depth'] = 10**(0.97*np.log10(svrad_phot)-3.62) # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2['ew'] = 10**(0.97*np.log10(svrad_phot)-3.47) # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2['vspan'] = 10**(0.98*np.log10(svrad_phot)-2.95) # from photon noise simulations Photon_noise_CCF.py

    logger.info(f'Photon noise RV from calibration : {np.median(svrad_phot2['rv'])*1000:.2f} m/s ')

    logger.info(f'Number of velocity bin ={len(vrad):%.0f}')

    if np.sum(noise_ccf!=0)>0:
        noise_ccf[noise_ccf==0] = np.mean(noise_ccf[noise_ccf!=0])
    else:
        noise_ccf *= 0
        noise_ccf += 0.01
    ccf_power_std = noise_ccf
    factor = 1/(np.percentile(noise_ccf,75,axis=0))**2
    ccf_power = ccf_power*factor
    ccf_power_std = ccf_power*factor

    end = time.time()
    if myv.DEV:
        counter_dev+=1
        logger.debug(f"Line number: {inspect.currentframe().f_lineno}")
        logger.debug(f"Execution time {counter_dev}: {end - start:.3f} seconds")

    # TBD optimize take 9s for N=360
    for j,i in enumerate(files[-1]):
        ccf_power_old = ccf_power[:,j]
        ccf_power_old_std = ccf_power_std[:,j]
        ccf = myc.tableXY(vrad/1000,ccf_power_old,ccf_power_old_std)

        ccf_backup = ccf.copy()
        ccf_backup.yerr/=np.nanpercentile(ccf_backup.y,75)
        ccf_backup.y/=np.nanpercentile(ccf_backup.y,75)

        if debug:
            plt.figure('debug')
            ccf_backup.plot()

        if analytical_model=='gaussian':
            ccf_backup.fit_gaussian(Plot=debug)
            model_parametric = 'GND2.0'
        else:
            ccf_backup.fit_GND(Plot=debug,beta_fixed=beta0)
            model_parametric = 'GND%.1f'%(beta0)

        plt.close('debug')

        ccf.yerr = np.sqrt(abs(ccf.y))

        ccf.y *= -1
        ccf.find_max(vicinity=5)

        ccf.diff(replace=False)
        ccf.deri.y = np.abs(ccf.deri.y)
        for jj in range(3):
            ccf.deri.find_max(vicinity=4-jj)
            if len(ccf.deri.x_max)>1:
                break

        first_max = ccf.deri.x_max[np.argsort(ccf.deri.y_max)[-1]]
        second_max = ccf.deri.x_max[np.argsort(ccf.deri.y_max)[-2]]

        ccf.y *= -1
        if (np.min(abs(ccf.x_max-0.5*(first_max+second_max)))<5)&(fwhm<15):
            center=ccf.x_max[np.argmin(abs(ccf.x_max-0.5*(first_max+second_max)))]
        else:
            center=ccf.x[ccf.y.argmin()]
        ccf.x -= center

        if not del_outside_max:
            mask = (ccf.x>-rv_borders)&(ccf.x<rv_borders)
            ccf.supress_mask(mask)
        else:
            ccf.find_max(vicinity=10)
            ccf.index_max = np.sort(ccf.index_max)
            mask = np.zeros(len(ccf.x)).astype('bool')
            mask[ccf.index_max[0]:ccf.index_max[1]+1]=True
            ccf.supress_mask(mask)

        if normalisation=='left':
            norm = ccf.y[0]
        else:
            max1 = np.argmax(ccf.y[0:int(len(ccf.y)/2)])
            max2 = np.argmax(ccf.y[int(len(ccf.y)/2):])+int(len(ccf.y)/2)
            fmax1 = ccf.y[max1]
            fmax2 = ccf.y[max2]
            norm = (fmax2-fmax1)/(max2-max1)*(np.arange(len(ccf.y))-max2)+fmax2
        ccf.yerr /= norm
        ccf.y /= norm

        if debug:
            ccf.plot(color=None)

        if analytical_model=='gaussian':
            ccf.fit_gaussian(Plot=False)
        else:
            ccf.fit_GND(Plot=False,beta_fixed=beta0)

        ccf_backup.params['cen'].value -= center

        if check_non_transform:
            V1,V2 = ccf_backup.params['cen'].value,ccf.params['cen'].value
            if abs(V1-V2)>1:
                logger.warning(f'Discrepancy detected between CCFs ({V1:.4f}/{V2:.4f}), value reset to non-transformed one')
                ccf.params = ccf_backup.params

        rv_ccf = ccf.params['cen'].value+center
        rv_ccf_std = ccf.params['cen'].stderr
        rv_ccf,rv_ccf_std = replace_none(rv_ccf,rv_ccf_std)
        rv_ccf_std = svrad_phot2['rv'][j]

        contrast_ccf = -ccf.params['amp'].value
        contrast_ccf_std = ccf.params['amp'].stderr
        contrast_ccf,contrast_ccf_std = replace_none(contrast_ccf,contrast_ccf_std)
        contrast_ccf_std = svrad_phot2['contrast'][j]

        wid_ccf = ccf.params['wid'].value
        wid_ccf_std = ccf.params['wid'].stderr
        wid_ccf,wid_ccf_std = replace_none(wid_ccf,wid_ccf_std)
        wid_ccf_std = svrad_phot2['fwhm'][j]

        offset_ccf = ccf.params['offset'].value
        offset_ccf_std = ccf.params['offset'].stderr
        offset_ccf,offset_ccf_std = replace_none(offset_ccf,offset_ccf_std)

        amplitudes.append(contrast_ccf)
        amplitudes_std.append(contrast_ccf_std)
        rvs.append(rv_ccf)
        rvs_std.append(rv_ccf_std)
        fwhms.append(wid_ccf)
        fwhms_std.append(wid_ccf_std)

        ccf.clip(min=[-bis_range,None],max=[bis_range,None],replace=False)
        if len(ccf.clipped.x)<5:
            ccf.clip(min=[-0.5,None],max=[0.5,None],replace=False)
            logger.info('BISSPAN updated to +/- 0.5 km/s')
        if len(ccf.clipped.x)<5:
            ccf.clip(min=[-2,None],max=[2,None],replace=False)
            logger.info('BISSPAN updated to +/- 2 km/s')
        if len(ccf.clipped.x)<5:
            ccf.clip(min=[-5,None],max=[5,None],replace=False)
            logger.info('BISSPAN updated to +/- 5 km/s')

        ccf.clipped.fit_poly()
        a,b,c = ccf.clipped.poly_coefficient
        para_center = -b/(2*a)+center
        para_depth = a*(-b/(2*a))**2+b*(-b/(2*a))+c
        centers.append(para_center)
        depths.append(1-para_depth)

        EW = np.sum(1-ccf.y)/len(ccf.y)
        ew.append(EW)
        save_ccf = {'ccf_flux':ccf.y,'ccf_flux_std':ccf.yerr,'ccf_rv':ccf.x+center,'ew':EW}

        para_ccf = {'para_rv':para_center,'para_depth':para_depth}

        ccf_core = ccf.copy()
        if rv_ccf==rv_ccf:
            ccf_core.x += center
            ccf_core.x -= rv_ccf

        vrad_center = np.arange(0,bis_range+(dv/1000)*0.99,dv/1000)
        vrad_center = np.hstack([-vrad_center[1:][::-1],vrad_center])

        ccf_core.interpolate(new_grid=vrad_center,replace=True,method='cubic')
        ccf_core.fit_poly()
        a,b,c = ccf_core.poly_coefficient
        vs = -b/(2*a)

        bisspan.append(vs)
        bisspan_ccf_std = svrad_phot2['vspan'][j]
        bisspan_std.append(bisspan_ccf_std)

        ew_std.append(svrad_phot2['ew'][j])
        centers_std.append(svrad_phot2['center'][j])
        depths_std.append(svrad_phot2['depth'][j])

        save_ccf['ew_std'] = svrad_phot2['ew'][j]
        para_ccf['para_rv_std'] = svrad_phot2['center'][j]
        para_ccf['para_depth_std'] = svrad_phot2['depth'][j]

        save_gauss = {'contrast':contrast_ccf,'contrast_std':contrast_ccf_std,
                                'rv':rv_ccf,'rv_std':rv_ccf_std, 'rv_std_phot':svrad_phot2['rv'][j],
                                'fwhm':wid_ccf,'fwhm_std':wid_ccf_std,
                                'offset':offset_ccf,'offset_std':offset_ccf_std,
                                'vspan':rv_ccf - para_center,'vspan_std':bisspan_ccf_std}

    end = time.time()
    if myv.DEV:
        counter_dev+=1
        logger.debug(f"Line number: {inspect.currentframe().f_lineno}")
        logger.debug(f"Execution time {counter_dev}: {end - start:.3f} seconds")

    rvs_std = svrad_phot2['rv']
    fwhms = np.array(fwhms).astype('float')*2.355
    fwhms_std = np.array(fwhms_std).astype('float')*2.355

    warning_rv_borders = False
    if np.median(fwhms)>(rv_borders/1.5):
        logger.warning('The CCF is larger than the RV borders for the fit')
        warning_rv_borders = True

    if jdb is None:
        jdb = np.arange(len(files[-1]))
    ccf_rv = myc.tableXY(jdb,np.array(rvs)*1000,np.array(rvs_std)*1000)
    ccf_centers = myc.tableXY(jdb,np.array(centers)*1000,np.array(centers_std)*1000)
    ccf_contrast = myc.tableXY(jdb,np.array(amplitudes)*100,np.array(amplitudes_std)*100)
    ccf_depth = myc.tableXY(jdb,depths,depths_std)
    ccf_fwhm = myc.tableXY(jdb,fwhms,fwhms_std)
    ccf_vspan = myc.tableXY(jdb,np.array(bisspan)*1000,np.array(bisspan_std)*1000)
    ccf_ew = myc.tableXY(jdb,np.array(ew),np.array(ew_std))
    ccf_timeseries = np.array([ew,ew_std,amplitudes,amplitudes_std,rvs,rvs_std,svrad_phot2['rv'],fwhms,fwhms_std,centers,centers_std,depths,depths_std,bisspan,bisspan_std])
    ccf_infos = pd.DataFrame(ccf_timeseries.T,columns=['ew','ew_std','contrast','contrast_std','rv','rv_std','rv_std_phot','fwhm','fwhm_std','center','center_std','depth','depth_std','bisspan','bisspan_std'])
    ccf_infos['jdb'] = jdb
    ccf_infos['filename'] = files[-1]

    #Update to remove nan value in RV 02.05.25
    ccf_rv.yerr[ccf_rv.y!=ccf_rv.y] = np.nanmedian(ccf_rv.yerr[ccf_rv.y!=ccf_rv.y])
    offset = np.nanmedian(ccf_centers.y - ccf_rv.y)
    ccf_rv.y[ccf_rv.y!=ccf_rv.y] = ccf_centers.y[ccf_rv.y!=ccf_rv.y] - offset

    ccf_infos = {'table':ccf_infos,'model_parametric':model_parametric,'weighting':1.0+float(squared),'creation_date':datetime.datetime.now().isoformat()}

    file_summary_ccf = myf.touch_pickle(dir_root+'WORKSPACE/Analyse_ccf.p')
    file_summary_ccf['CCF_'+mask_name.split('.')[0]] = ccf_infos

    myf.pickle_dump(file_summary_ccf,open(dir_root+'WORKSPACE/Analyse_ccf.p','wb'))

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

    export = myf.touch_pickle(dir_root+'WORKSPACE/Analyse_ccf_saved.p')
    export['CCF_'+ccf_name] = {}
    export['CCF_'+ccf_name][sub_dico] = {'ccf_vrad':vrad,'ccf_flux':ccf_norm,'ccf_shifted':ccf_shifted,'ccf_master':master_ccf,'filename':files[-1]}
    myf.pickle_dump(export,open(dir_root+'WORKSPACE/Analyse_ccf_saved.p','wb'))

    warning = 0
    if ccf_name=='mask_telluric_o2':
        fwhm_ins = np.nanmedian(ccf_fwhm.y)
        if ins.split('_')[0] in myv.instrument_res_kms.keys():
            ref = myv.instrument_res_kms[ins.split('_')[0]]
            logger.info(f'Reference value for {ins} is {ref:.1f} km/_s')
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
    plt.imshow(ccf_res.T,vmin=-0.02,vmax=0.02,aspect='auto',cmap='seismic') ;
    plt.axvline(x=len(vrad)*0.5,color='k',ls='-.',lw=1)
    plt.axes([0.75,0.72,0.22,0.22])
    plt.plot(vrad/1000,master_ccf,color='k')
    plt.plot(vrad/1000,ccf_norm,alpha=0.2,color='k')
    plt.axvline(x=0,color='k',ls='-.',lw=1)
    plt.tick_params(top=True,labeltop=True,labelbottom=False)
    plt.savefig(dir_root+'IMAGES/CCF_summary_%s'%(ccf_name)+myv.PRD_EXT+'.png')
    if warning:
        plt.savefig(dir_root+'WARNING/CCF_summary_%s'%(ccf_name)+myv.PRD_EXT+'.png')

    output = {
        'rv':ccf_rv,
        'contrast':ccf_contrast,
        'fwhm':ccf_fwhm,
        'vspan':ccf_vspan}

    if save:
        summary = import_summary(dir_root)
        mask = myf.in1d(np.array(summary['filename']),files[-1])
        summary['ccf_rv_'+ccf_name] = np.nan ; summary.loc[mask,'ccf_rv_'+ccf_name] = np.round(ccf_rv.y,0) # DONT USE RV FROM SNAKY, PRECISION NOT BETTER THAN 3 M/S
        summary['ccf_ct_'+ccf_name] = np.nan ; summary.loc[mask,'ccf_ct_'+ccf_name] = np.round(ccf_contrast.y,4)
        summary['ccf_fwhm_'+ccf_name] = np.nan ; summary.loc[mask,'ccf_fwhm_'+ccf_name] = np.round(ccf_fwhm.y,4)
        summary.to_csv(dir_root+'WORKSPACE/Analyse_summary.csv')

    if return_ccf:
        return output, vrad, ccf_shifted
    else:
        return output

observationContext = ObservationContext(dir_root='/data/HD189733/', files=[])
stellarParams = StellarParams(rv_sys=-2.3, fwhm=7.2, beta_gnd=2.1)
maskConfig = MaskConfig(mask_input='G2', wave_min=5000, wave_max=6800)

test = yarara_ccf(observationContext, stellarParams, maskConfig)
