from doctest import master
import getopt
import datetime
import pandas as pd
import numpy as np 
import matplotlib.pylab as plt
import pickle
import os
from astropy.io import fits
from tqdm import tqdm
import glob as glob
import time
import sys

from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from sklearn.neighbors import KNeighborsRegressor #saved in the pickle calibration file
from colorama import Fore
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d,RectBivariateSpline
from scipy.optimize import minimize

from . import snaky_variables as myv
from . import snaky_functions as myf
from . import snaky_classes as myc

sys.path.append(myv.ROOT_DIR.replace('SNAKY','FINCH')) #Until FINCH is pip installable

try:
    import finch as Finch
    installation = 'complete'
    print('[IMPORT] FINCH module is found')
except:
    installation = 'partial'
    print(Fore.YELLOW+'[IMPORT] FINCH module is missing'+Fore.RESET)

MATERIAL_DIR = myv.MATERIAL_DIR
interp_degree = myv.interp_degree

import inspect

"""

SNAKY — Spectroscopic Novel Analysis Kit of Yarara

"""

__version__ = '1.6.1'

print(Fore.GREEN+"""\n[INFO SNAKY]
[INFO USER] SNAKY version = """+__version__ +""" 
[INFO USER] READ ME CAREFULLY 
[INFO USER] Vsini only validated for solar analogs
[INFO USER] Continuum normalisation by RASSINE (see Cretignier et al. 2020b)
[INFO USER] Atmospheric parameters see Cretignier et al. 2024a
[INFO USER] The MHK activity index see Cretignier et al. 2024b
[INFO USER] An issue or an upgrade? Contact me at:  michael.cretignier@physics.ox.ac.uk
      """+Fore.RESET)

cwd = os.getcwd()
root = '/'.join(cwd.split('/')[:-1])

#INTERMEDIATE FUNCTIONS

def get_berv(ra_deg, dec_deg, obstime_utc, instrument):
    ins = instrument.split('_')[0].split('-')[0]
    obs_lat = myv.instrument_lat_deg[ins.upper()]
    obs_lon = myv.instrument_lon_deg[ins.upper()]
    obs_alt = myv.instrument_altm[ins.upper()]

    t = Time(obstime_utc, scale='utc')
    star = SkyCoord(ra=ra_deg*u.deg, dec=dec_deg*u.deg)
    loc = EarthLocation.from_geodetic(obs_lon*u.deg, obs_lat*u.deg, obs_alt*u.m)
    #loc2 = myf.observatory(instrument='HARPS')
    berv = star.radial_velocity_correction(obstime=t, location=loc)
    return berv.to(u.km/u.s)

def snaky_help():
    print(Fore.GREEN +'\n [SNAKY ADVICE] Here is some help!'+Fore.RESET)

def snaky_print_sequence():
    names = [
        "force_pre",
        "force_summary",
        "force_rvsys",
        "force_ccf",
        "force_master",
        "force_atmos",
        "force_resolution",
        "force_vsini",
        "force_abs_continuum",
        "force_activity",
        "force_mhk",
        "force_spectroscopy",
        "force_magcycle",
        "force_cleaning",
    ]

    for i, name in enumerate(names, start=1):
        print(f"{i:2d}  {name}")

##### CONTROL CHECK FUNCTIONS (DB PROCESSING)

def check_force_pre(dir_root,step_nb=''):
    try:
        test = pd.read_pickle(glob.glob(dir_root+'WORKSPACE/RASSINE*')[0])
        os.system('touch '+dir_root+'REDUCTION_INFO/force_pre.txt')
        print(Fore.GREEN+' [INFO] Recipe PRE done! %s'%(step_nb)+Fore.RESET) ; QC=1
    except:
        print(Fore.YELLOW+' [INFO] Recipe PRE not launched or crashed! %s'%(step_nb)+Fore.RESET) ; QC=0
    return QC

def check_force_summary(dir_root,step_nb=''):
    try:
        os.system('rm -f '+dir_root+'REDUCTION_INFO/force_summary.txt')
        test = pd.read_csv(dir_root+'WORKSPACE/Analyse_summary.csv',index_col=0)['flag1']
        os.system('touch '+dir_root+'REDUCTION_INFO/force_summary.txt')
        print(Fore.GREEN+' [INFO] Recipe SUMMARY done! %s'%(step_nb)+Fore.RESET) ; QC=1
    except:
        print(Fore.YELLOW+' [INFO] Recipe SUMMARY not launched or crashed! %s'%(step_nb)+Fore.RESET) ; QC=0
    return QC

def check_force_rvsys(dir_root,step_nb=''):
    try:
        os.system('rm -f '+dir_root+'REDUCTION_INFO/force_rvsys.txt')
        test = pd.read_pickle(glob.glob(dir_root+'STAR_INFO/Stellar_info*.p')[0])['Rv_sys']['SNAKY']
        os.system('touch '+dir_root+'REDUCTION_INFO/force_rvsys.txt')
        print(Fore.GREEN+' [INFO] Recipe RVSYS done! %s'%(step_nb)+Fore.RESET) ; QC=1
    except:
        print(Fore.YELLOW+' [INFO] Recipe RVSYS not launched or crashed! %s'%(step_nb)+Fore.RESET) ; QC=0
    return QC

def check_force_ccf(dir_root,step_nb=''):
    try:
        os.system('rm -f '+dir_root+'REDUCTION_INFO/force_ccf.txt')
        test = pd.read_pickle(dir_root+'WORKSPACE/Analyse_ccf.p')['CCF_G2']['table']
        os.system('touch '+dir_root+'REDUCTION_INFO/force_ccf.txt')
        print(Fore.GREEN+' [INFO] Recipe CCF done! %s'%(step_nb)+Fore.RESET) ; QC=1
    except:
        print(Fore.YELLOW+' [INFO] Recipe CCF not launched or crashed! %s'%(step_nb)+Fore.RESET) ; QC=0
    return QC

def check_force_master(dir_root,step_nb=''):
    try:
        os.system('rm -f '+dir_root+'REDUCTION_INFO/force_master.txt')
        test = pd.read_pickle(dir_root+'WORKSPACE/Analyse_material.p')['reference_spectrum']
        os.system('touch '+dir_root+'REDUCTION_INFO/force_master.txt')
        print(Fore.GREEN+' [INFO] Recipe MASTER done! %s'%(step_nb)+Fore.RESET) ; QC=1
    except:
        print(Fore.YELLOW+' [INFO] Recipe MASTER not launched or crashed! %s'%(step_nb)+Fore.RESET) ; QC=0
    return QC

def check_force_atmos(dir_root,step_nb=''):
    try:
        os.system('rm -f '+dir_root+'REDUCTION_INFO/force_atmos.txt')
        test = pd.read_pickle(glob.glob(dir_root+'STAR_INFO/Stellar_info*.p')[0])['Mstar']['SNAKY']
        os.system('touch '+dir_root+'REDUCTION_INFO/force_atmos.txt')
        print(Fore.GREEN+' [INFO] Recipe ATMOS done! %s'%(step_nb)+Fore.RESET) ; QC=1
    except:
        print(Fore.YELLOW+' [INFO] Recipe ATMOS not launched or crashed! %s'%(step_nb)+Fore.RESET) ; QC=0
    return QC

def check_force_resolution(dir_root,step_nb=''):
    try:
        os.system('rm -f '+dir_root+'REDUCTION_INFO/force_resolution.txt')
        test = pd.read_pickle(glob.glob(dir_root+'STAR_INFO/Stellar_info*.p')[0])['FWHM']['O2']
        os.system('touch '+dir_root+'REDUCTION_INFO/force_resolution.txt')
        print(Fore.GREEN+' [INFO] Recipe RESOLUTION done! %s'%(step_nb)+Fore.RESET) ; QC=1
    except:
        print(Fore.YELLOW+' [INFO] Recipe RESOLUTION not launched or crashed! %s'%(step_nb)+Fore.RESET) ; QC=0
    return QC

def check_force_vsini(dir_root,step_nb=''):
    try:
        os.system('rm -f '+dir_root+'REDUCTION_INFO/force_vsini.txt')
        test = pd.read_pickle(glob.glob(dir_root+'STAR_INFO/Stellar_info*.p')[0])['Vsini']['SNAKY']
        if test!=test:
            pouet
        os.system('touch '+dir_root+'REDUCTION_INFO/force_vsini.txt')
        print(Fore.GREEN+' [INFO] Recipe VSINI done! %s'%(step_nb)+Fore.RESET) ; QC=1
    except:
        print(Fore.YELLOW+' [INFO] Recipe VSINI not launched or crashed! %s'%(step_nb)+Fore.RESET) ; QC=0
    return QC

def check_force_abs_continuum(dir_root,step_nb=''):
    try:
        os.system('rm -f '+dir_root+'REDUCTION_INFO/force_abs_continuum.txt')
        test = pd.read_pickle(dir_root+'WORKSPACE/Analyse_material.p')['correction_factor']
        os.system('touch '+dir_root+'REDUCTION_INFO/force_abs_continuum.txt')
        print(Fore.GREEN+' [INFO] Recipe ABSCONTINUUM done! %s'%(step_nb)+Fore.RESET) ; QC=1
    except:
        print(Fore.YELLOW+' [INFO] Recipe ABSCONTINUUM not launched or crashed! %s'%(step_nb)+Fore.RESET) ; QC=0
    return QC

def check_force_activity(dir_root,step_nb=''):
    try:
        os.system('rm -f '+dir_root+'REDUCTION_INFO/force_activity.txt')
        test = pd.read_pickle(glob.glob(dir_root+'STAR_INFO/Stellar_info*.p')[0])['Contrast']['Ha']
        os.system('touch '+dir_root+'REDUCTION_INFO/force_activity.txt')
        print(Fore.GREEN+' [INFO] Recipe ACTIVITY done! %s'%(step_nb)+Fore.RESET) ; QC=1
    except:
        print(Fore.YELLOW+' [INFO] Recipe ACTIVITY not launched or crashed! %s'%(step_nb)+Fore.RESET) ; QC=0
    return QC

def check_force_mhk(dir_root,step_nb=''):
    try:
        os.system('rm -f '+dir_root+'REDUCTION_INFO/force_mhk.txt')
        test = pd.read_pickle(glob.glob(dir_root+'STAR_INFO/Stellar_info*.p')[0])['MHK']['SNAKY']
        os.system('touch '+dir_root+'REDUCTION_INFO/force_mhk.txt')
        print(Fore.GREEN+' [INFO] Recipe MHK done! %s'%(step_nb)+Fore.RESET) ; QC=1
    except:
        print(Fore.YELLOW+' [INFO] Recipe MHK not launched or crashed! %s'%(step_nb)+Fore.RESET) ; QC=0
    return QC

def check_force_spectroscopy(dir_root,step_nb=''):
    try:
        os.system('rm -f '+dir_root+'REDUCTION_INFO/force_spectroscopy.txt')
        test = pd.read_pickle(dir_root+'WORKSPACE/Analyse_spectroscopy.p')['flux_corrected']
        os.system('touch '+dir_root+'REDUCTION_INFO/force_spectroscopy.txt')
        print(Fore.GREEN+' [INFO] Recipe SPECTROSCOPY done! %s'%(step_nb)+Fore.RESET) ; QC=1
    except:
        print(Fore.YELLOW+' [INFO] Recipe SPECTROSCOPY not launched or crashed! %s'%(step_nb)+Fore.RESET) ; QC=0
    return QC

def check_force_magcycle(dir_root,step_nb=''):
    try:
        os.system('rm -f '+dir_root+'REDUCTION_INFO/force_magcycle.txt')
        test = pd.read_pickle(glob.glob(dir_root+'STAR_INFO/Stellar_info*.p')[0])['Pmag']['SNAKY']
        os.system('touch '+dir_root+'REDUCTION_INFO/force_magcycle.txt')
        print(Fore.GREEN+' [INFO] Recipe MAGCYCLE done! %s'%(step_nb)+Fore.RESET) ; QC=1
    except:
        print(Fore.YELLOW+' [INFO] Recipe MAGCYCLE not launched or crashed! %s'%(step_nb)+Fore.RESET) ; QC=0
    return QC

#

def create_snaky_dir(output_dir,star,ins):
    """
    starname: string, name of the star without space(e.g. HD12345)
    ins: string, name of the instrument and DRS version (e.g. ESPRESSO_3.3.1, HARPS_3.5)
    """
    
    if len(star.split(' '))>1:
        print(Fore.YELLOW + '\n [WARNING] The specified star (%s) contains spaces'%(star)+Fore.RESET)
        star = star.replace(' ','')
        print(Fore.YELLOW + ' [WARNING] Spaces have been removed, new starname: %s'%(star)+Fore.RESET)

    if len(ins.split('_'))!=2:
        print(Fore.YELLOW + '\n [WARNING] The specified instrument (%s) is wrong'%(ins)+Fore.RESET)
        print(Fore.YELLOW + ' [WARNING] The format should follow: SPECTRO_DRS (ESPRESSO_3.3.1, HARPS_3.5)'+Fore.RESET)
    if len(ins.split('_'))==1:
        ins = ins+'_1.0'
        print(Fore.YELLOW + ' [WARNING] The instrument DRS version was set to 1.0 (%s)'%(ins)+Fore.RESET)

    if not os.path.exists(output_dir+'/'+star+'/data/s1d/ALLINS_MERGED'):
        os.makedirs(output_dir+'/'+star+'/data/s1d/ALLINS_MERGED', exist_ok=True)

    if os.path.exists(output_dir+'/'+star+'/data/s1d/'+ins+'/WORKSPACE'):
        myv.vprint(' [INFO] SNAKY directory found!\n')
        
    directories = ['RAW','IMAGES','WORKSPACE','EXPORT','CCF_MASK','DACE_TABLE','DETECTION_LIMIT','FILM','KEPLERIAN','KITCAT','MASTER','PCA','REDUCTION_INFO','TEMP','STAR_INFO','WARNING']
    myv.vprint(' [INFO] Star and instrument defined as %s and %s'%(star,ins))
    for d in directories:
        base = output_dir+'/'+star+'/data/s1d/'+ins+'/'+d
        os.makedirs(base, exist_ok=True)
    return star, ins

def clean_light_dir(dir_root):
    os.system('rm -f '+dir_root+'CCF_MASK/CCF*.fits')
    material = import_material(dir_root)
    if 'activity_proxies' in material.keys():
        del material['activity_proxies']
    if 'stellar_template' in material.keys():
        del material['stellar_template']
    pickle.dump(material,open(dir_root+'WORKSPACE/Analyse_material.p','wb'))

    print(Fore.GREEN+' [INFO] The final cleaning of the output products was done.'+Fore.RESET) ; QC=1

def plot_master(dir_root, srf=False, color='k', alpha=1.0, offset=0, print_info=True, figname='master', branch='Snaky', debug=False):
    """srf: Stellar-rest-frame, will use the Analyse_spectroscopy file"""
    plt.figure(figname,figsize=(16,6))
    
    dir_root = dir_root.replace('Snaky',branch) #to plot YARARA
    dir_root = np.array(glob.glob(dir_root))

    if len(dir_root)>1:
        print_info=False
        all_teff = []
        for d in dir_root:
            try:
                teff = pd.read_pickle(glob.glob(d+'STAR_INFO/Ste*')[0])['Teff'][branch.upper()]
            except:
                teff = 0
            all_teff.append(teff)
        all_teff = np.array(all_teff)
        dir_root = dir_root[np.argsort(all_teff)]

    if debug:
        for d in dir_root:
            print(d)


    for n in tqdm(np.arange(len(dir_root))):
        droot = dir_root[n]
        print(droot)
        if srf:
            spectro = import_spectroscopy(droot)
            master = myc.tableXY(spectro['wave'],spectro['flux_corrected'],0*spectro['wave'])
            master.plot(color=color, offset=offset*n, alpha=alpha)
            if print_info:
                plt.title('Teff = %.0f K  |  Logg = %.2f dex  |  [Fe/H] = %.2f dex  |  Vsini = %.1f kms  |  RHK = %.2f dex  |  MHK = %.1f %%'%(spectro['Teff']['SNAKY'],spectro['Log_g']['SNAKY'],spectro['FeH']['SNAKY'],spectro['Vsini']['SNAKY'],spectro['RHK']['SNAKY'],spectro['MHK']['SNAKY']))
        else:
            master = import_master(droot)
            master.plot(color=color, offset=offset*n, alpha=alpha)
    
    if n ==0:
        plt.ylim(-0.1,1.1)
    plt.xlabel(r'Wavelength [$\AA$]',fontsize=14)
    plt.ylabel('Flux normalised',fontsize=14)
    return dir_root

def plot_mhk(dir_root, hide_outliers=True, daily_binned=True, debug=False, rhk_ref=None):
    directory = '/'.join(dir_root.split('/')[0:-2])
    ins = dir_root.split('/')[-2]
    summaries = glob.glob(directory+'/*/WORKSPACE/Analyse_summary.csv')
    plt.figure(figsize=(15,6))
    plt.axes([0.05,0.09,0.75,0.84])
    count=-1
    samples = []
    for s in summaries:
        instrument = s.split('/WORKSPACE')[0].split('/')[-1]
        if debug:
            star_info = glob.glob(s.split('/WORKSPACE')[0]+'/STAR_INFO/Stellar*')[0]
            teff = np.round(pd.read_pickle(star_info)['Teff']['SNAKY'],-1)
            instrument = instrument+'(%.0f)'%(teff)
        tab = pd.read_csv(s,index_col=0) 
        if ('jdb' in tab.keys())&('MHK' in tab.keys()):
            count+=1
            proxy = myc.tableXY(np.array(tab['jdb']),np.array(tab['MHK']),np.array(tab['MHK_std']))
            proxy.supress_nan()
            if daily_binned:
                proxy.night_stack(replace=True)
            if hide_outliers:
                valid = proxy.yerr<20
            else:
                valid = proxy.yerr>0
            proxy.x[proxy.x<43850.0] = 43850.0   #minimum of the solar plot
            plt.errorbar(proxy.x[valid],proxy.y[valid],yerr=proxy.yerr[valid],label=instrument,marker=['o','s','^'][int(count//10)],ls='',capsize=0,color='C%.0f'%(count),mec='k')
            plt.scatter(proxy.x[~valid],proxy.y[~valid],marker='x',color='C%.0f'%(count))
            samples.append(np.ravel(np.random.randn(5000,len(proxy.y))*proxy.yerr+proxy.y))

    plt.legend()
    plt.ylabel('M-index [%]',fontsize=14)
    plt.xlabel('Jdb - 2,400,000 [days]',fontsize=14)

    solar_cycle = pd.read_csv(MATERIAL_DIR+'/Solar_Mg2.csv',index_col=0)
    sun_mag = myc.tableXY(solar_cycle['jdb'],solar_cycle['plage_fill'],0*solar_cycle['jdb'])
    sun_mag.smooth(box_pts=100,shape='savgol')
    plt.plot(sun_mag.x,sun_mag.y,color='gold',lw=1,alpha=0.7)
    plt.fill_between(sun_mag.x,0,sun_mag.y,color='gold',alpha=0.25)
    if rhk_ref is not None:
        plt.axhline(y=rhk_mhk(rhk_ref),color='k',ls='-.')

    ax = plt.gca()
    x_ticks = ax.get_xticks()[1:-1]
    xlim = ax.get_xlim()
    y_ticks = ax.get_yticks()[1:-1]
    ylim = ax.get_ylim()
    ax.twiny()
    plt.xlim(xlim)
    plt.xticks(x_ticks,np.round(2000+(x_ticks-51544.5)/365,1))
    plt.xlabel('Date [year]',fontsize=14)
    plt.axes([0.83,0.09,0.10,0.84])
    plt.tick_params(labelleft=False)
    ax = plt.gca()
    ax.twinx()
    plt.ylim(ylim)
    for n,s in enumerate(samples):
        a,b = np.histogram(s,np.linspace(ylim[0],ylim[1],100),density=True)
        b = 0.5*(b[1:]+b[0:-1])
        plt.fill_betweenx(b,0*a,a,alpha=0.3,color='C%.0f'%(n))
        plt.plot(a,b,color='C%.0f'%(n),lw=1)
    a,b = np.histogram(np.hstack(samples),np.linspace(ylim[0],ylim[1],100),density=True)
    b = 0.5*(b[1:]+b[0:-1])
    if rhk_ref is not None:
        plt.axhline(y=rhk_mhk(rhk_ref),color='k',ls='-.')
    plt.plot(a,b,alpha=1.0,color='k',lw=1.5)
    plt.xlim(0,None)
    plt.yticks(y_ticks,np.round(mhk_rhk(y_ticks),2))
    plt.ylabel(r'$\log$ $R_{HK}$ [dex]',fontsize=14)
    plt.subplots_adjust(top=0.93)
    plt.savefig(dir_root+'IMAGES/MHK'+myv.PRD_EXT+'.png')
    plt.savefig(dir_root.replace(ins,'ALLINS_MERGED')+'MHK'+myv.PRD_EXT+'.png')

def yarara_lithium_age(dir_root, teff=None, teff_std=70, ref_age=None):
    master = import_master(dir_root)
    star_info = import_star_info(dir_root)
    rv_sys = star_info['Rv_sys']['SNAKY']

    try:
        fwhm = star_info['FWHM']['G2']
    except:
        fwhm = star_info['FWHM']['fixed']

    sigma = fwhm/2.35

    samples = pd.read_csv(dir_root+'WORKSPACE/Analyse_samples.csv.gz')

    if teff is None:
        teff_samples = np.array(samples['teff'])
        teff_std = np.std(teff_samples)
    else:
        teff_samples = np.random.randn(5000)*teff_std+teff
    teff = np.mean(teff_samples)

    wc = myf.doppler_r(6707.84,rv_sys*1000)[0]
    windows = (3*sigma)*wc/3e5
    master.clip(min=[wc-4*windows,None],max=[wc+4*windows,None])

    if np.sum(master.y)==0:
        return

    ew_samples = []
    for wid in np.linspace(2,3,10):
        windows = (wid*sigma)*wc/3e5
        mask_line = abs(master.x-wc)<windows
        error = myf.mad(master.y[~mask_line])
        master.yerr = np.ones(len(master.x))*error

        continuum = np.nanpercentile(master.y[~mask_line],90)
        master.y = master.y/continuum

        dl = np.gradient(master.x[mask_line])
        ew = np.trapz(1 - master.y[mask_line]/continuum, master.x[mask_line])*1000
        ew_std = np.sqrt(np.nansum((master.yerr[mask_line] * dl)**2))*1000

        if ew<10:
            samples2 = np.random.uniform(0,1,500)*20
        else:
            samples2 = np.random.randn(500)*ew_std + ew

        ew_samples.append(samples2)

    ew_samples = np.ravel(ew_samples)
    ew_samples[ew_samples<0] = 0
    ew = np.mean(ew_samples)
    ew_std = np.std(ew_samples)

    ew_curve = np.random.rand(20000)*295+5

    print(' [INFO] Li1 EW = %.2f +/- %.2f mA'%(ew,ew_std))

    warn = 0
    age_curve,warn = myf.lithium_age(teff_samples, ew_curve, nsamples=len(ew_curve))
    if np.mean(teff)<6700: #outside calibration
        age,warn = myf.lithium_age(teff_samples, ew_samples, nsamples=5000)
    else:
        age = np.nan*ew_samples

    if np.sum(age!=13):
        age[age==13] = np.nan

    if np.sum(age!=0):
        age[age==0] = np.nan

    print(' [INFO] Age = %.2f +/- %.2f Gyr'%(np.nanmean(age),np.nanstd(age)))

    samples['age'] = np.random.choice(age,len(samples),replace=True)
    samples.to_csv(dir_root+'WORKSPACE/Analyse_samples.csv.gz',index=False)

    plt.figure(figsize=(15,6))
    plt.axes([0.08,0.09,0.55,0.87])
    master.plot()
    plt.axhline(y=1,color='g')
    plt.axvline(x=wc,color='k',ls=':',label='%.2f A'%(wc))
    plt.fill_between(master.x[mask_line],master.y[mask_line],master.y[mask_line]*0+1,alpha=0.3,color='g')
    plt.title(r'EW = %.2f +/- %.2f [$m\AA$]'%(ew,ew_std),fontsize=13)
    plt.xlim(wc-5*windows,wc+5*windows)
    plt.ylabel('Flux normalised',fontsize=14)
    plt.xlabel(r'Wavelength [$\lambda$]',fontsize=14)
    plt.ylim(0.7,1.05)

    plt.axes([0.70,0.59,0.25,0.37])    
    plt.scatter(np.log10(age_curve*1.e9),ew_curve,alpha=0.1,color='k',s=1)
    plt.scatter(np.log10(age*1.e9),ew_samples,alpha=0.1,color='C0',s=1)
    plt.scatter(np.mean(np.log10(age*1.e9)), np.mean(ew_samples), color='C0', s=50, ec='k')
    plt.title('Teff = %.0f +/- %.0f'%(np.mean(teff_samples),np.std(teff_samples)),fontsize=13)
    plt.xlim(6,10)
    plt.ylim(0,300)
    plt.grid()
    plt.ylabel('EW [mA]',fontsize=14)
    plt.xlabel('Log Age [yr]',fontsize=14)

    plt.axes([0.70,0.09,0.25,0.37])    

    if np.mean(teff)<6700: 
        plt.hist(age[age==age],bins=30,alpha=0.3,color='C0')
        plt.hist(age[age==age],bins=30,alpha=1.0,color='C0',histtype='step')
        plt.xlim(0,None)

    plt.xlabel('Age [Gyr]',fontsize=14)
    plt.axvline(x=ref_age,color='k',ls='-.')
    
    scale = 1
    unit = 'Gyr'
    precision = '%.2f'
    if np.nanmedian(age)<1:
        scale = 1000
        unit = 'Myr'
        precision = '%.0f'

    if warn == 0:
        plt.title((f"Age = {precision} ± {precision} %s") % (np.nanmedian(age)*scale, myf.mad(age)*scale, unit), fontsize=13)
    elif warn == 1:
        plt.title((f"Age > {precision} %s") % (np.nanpercentile(age, 2.5)*scale, unit), fontsize=13)
    elif warn == 2:
        plt.title((f"Age < {precision} %s") % (np.nanpercentile(age, 97.5)*scale, unit), fontsize=13)

    plt.savefig(dir_root+'IMAGES/Age_Lithium.png')

    return np.mean(age)

def yarara_finch(dir_root, proxy_name='MHK',ext='',trend_degree=0, harm=0, offset_instrument='yes', automatic_fit=False, x_unit='years',predict='today', predict_samples=None,print_reference=True, rm_source=['DACE','Yu+23'], rm_ins=[], add_source=[], add_ins=[], offset_fixed=['SNAKY','HYDRA']):

    if myv.VERBOSE:
        myf.print_box('\n---- RECIPE : FINCH MAGNETIC CYCLE PERIOD ----\n')

    starname = dir_root.split('/')[-5]
    ins = dir_root.split('/')[-2]
    star_info = import_star_info(dir_root)

    teff = int(star_info['Teff']['SNAKY'])
    logg = star_info['Log_g']['SNAKY']
    feh = star_info['FeH']['SNAKY']

    x=[] ; y=[] ; yerr=[] ; instrument = [] ; reference = [] ; flag = []

    folder = '/'.join(dir_root.split('/')[0:-2])
    files = list(glob.glob(dir_root.replace(ins,'*')+'WORKSPACE/Analyse_Finch_table.csv'))
    yarara = folder.replace('Snaky','Yarara')+'/INS_MERGED/WORKSPACE/Analyse_Finch_table.csv'
    if os.path.exists(yarara):
        files = [yarara]+ files

    print('[INFO] The following FINCH tables were found:\n')
    for file in files:
        print(file)
        branch = file.split('/'+starname)[0].split('/')[-1]
        table = pd.read_csv(file,index_col=0)
        if branch=='Yarara':
            yerr.append(np.array(table['mhk_cleaned_std']))
            y.append(np.array(table['mhk_cleaned']))
        else:
            yerr.append(np.array(table[proxy_name.lower()+'_std']))
            y.append(np.array(table[proxy_name.lower()]))
        x.append(np.array(table['jdb']))
        instrument.append(np.array(table['ins']))   
        reference.append(np.array(table['source']))   
        flag.append(np.array(table['flag']))
    print('\n')   

    folder = dir_root.split('/Snaky')[0]
    files = glob.glob(MATERIAL_DIR+'/Activity_MHK_*.csv')
    for file in files:
        print('[INFO] Table %s loaded'%(file))             
        proxy_name_is = file.split('Activity_MHK_')[1].split('_')[0]
        table = pd.read_csv(file,index_col=0)
        table = table.loc[table['star']==starname]
        if len(table):
            yerr.append(table[proxy_name_is+'_std'].values)
            y.append(table[proxy_name_is].values)
            x.append(table['jdb'].values)
            instrument.append(table['ins'].values)   
            reference.append(table['source'].values)   
            flag.append(table['flag'].values)   

    db_finch = Finch.get_star(
        starname.split('_')[0],
        finch_offset = True,
        rm_source = rm_source,
        add_source = add_source,
        rm_ins = rm_ins,
        add_ins = add_ins,
        )
    
    if db_finch is not None:
        db_finch.masked(db_finch.y!=0)
        if proxy_name.split('_')[0]=='MHK':
            db_finch.convert_smw_mhk(teff)
        x.append(db_finch.x)
        y.append(db_finch.y)
        yerr.append(db_finch.yerr)
        instrument.append(db_finch.instrument)
        reference.append(db_finch.reference)
        flag.append(db_finch.mask_flag)

    print("\n")
    jdb = np.hstack(x).astype('float')
    sindex = np.hstack(y).astype('float')
    sindex_std = np.hstack(yerr).astype('float')
    proxy = myc.tableXY(jdb,sindex,sindex_std)
    instrument = np.hstack(instrument)
    reference = np.hstack(reference)
    flag = np.hstack(flag)
    ext = '_'+proxy_name+ext

    instru = np.array([i.split('_')[0] for i in instrument])
    #reference[(instru=='NEID')|(instru=='NEID-HE')] = 'snaky'

    vec = Finch.tableXY(proxy.x, proxy.y, proxy.yerr, proxy_name = proxy_name) 
    vec.set_instrument(instru)
    vec.set_reference(reference)

    vec.set_ins_uncertainties(null_yerr=False)
    vec.set_flag(flag)

    vec.set_star(
        starname = starname, 
        teff = teff,
        logg = logg,
        feh = feh,
        )
    if not print_reference:
        vec.print_reference = False

    vec.create_hydra()

    vec.mask_flag[vec.yerr>20] = True
    vec.mask_flag[vec.x==0] = True      #remove to time datapoint

    #self.debug = vec,trend_degree,harm,automatic_fit,automatic_fit,offset_instrument,predict,x_unit

    if ((np.max(vec.x)-np.min(vec.x))/365.25)>4: #at least 4 years baseline to fit

        vec.fit_period_cycle(
            trend_degree = trend_degree, 
            harm = harm,
            automatic_fit = automatic_fit, 
            data_driven_std = False, 
            offset_instrument = offset_instrument, 
            offset_fixed = offset_fixed,
            predict = 'today',
            x_unit = x_unit)

        plt.savefig(dir_root.replace(ins,'ALLINS_MERGED')+'Finch_magnetic_cycle'+ext+myv.PRD_EXT+'.png')
        if not vec.out_convergence_flag:
            vec.out_pmag = 0.00
        vec.remove_ins_offset()

    else:
        dust = vec.prepare_data(debug=False, data_driven_std=True)
        vec.out_convergence_flag = False
        vec.out_output_table = {'P_computed':np.array([0.00,0.00,0.00])}
        vec.out_pmag = 0.00

    exportation1 = myc.tableXY(vec.x,vec.y,vec.yerr)
    exportation1.mask_qc = ~vec.mask_flag
    exportation1.export(dir_root.replace(ins,'ALLINS_MERGED')+'Finch_%s.csv'%(proxy_name),format='csv',columns=['jdb','proxy','proxy_std','qc'],species=vec.instrument)

    exportation2 = myc.tableXY(vec.bin.x,vec.bin.y,vec.bin.yerr)
    exportation2.mask_qc = ~vec.bin.mask_flag
    exportation2.export(dir_root.replace(ins,'ALLINS_MERGED')+'Finch_%s_binned.csv'%(proxy_name),format='csv',columns=['jdb','proxy','proxy_std','qc'],species=vec.bin.instrument)

    FINCH_Pmag = np.round(vec.out_pmag,1)     
    
    predict= 2026.0
    fig_gp = vec.fit_gp(baseline_factor=1, runalgo=bool(vec.out_convergence_flag), predict=predict, print_legend=False)

    solar_cycle = pd.read_csv(MATERIAL_DIR+'/Solar_Mg2.csv',index_col=0)
    sun_mag = myc.tableXY(solar_cycle['deciyear'],solar_cycle['plage_fill'],0*solar_cycle['deciyear'])
    sun_mag.smooth(box_pts=100,shape='savgol')
    plt.plot(sun_mag.x,sun_mag.y,color='gold',lw=1,alpha=0.7)
    plt.fill_between(sun_mag.x,0,sun_mag.y,color='gold',alpha=0.25)
    ax = plt.gca()
    ylim = ax.get_ylim()
    if ylim[0]>0:
        plt.ylim(0,None)
    if ylim[1]<10:
        plt.ylim(None,10)

    FINCH_Pmag_GP = np.round(vec.out_gp_pmag,1)   
    FINCH_Mmag_GP = np.round(vec.out_gp_meanmag,1)   
    FINCH_Kmag_GP = np.round(vec.out_gp_ampmag,1)   

    plt.savefig(dir_root.replace(ins,'ALLINS_MERGED')+'Finch_magnetic_cycle_GP'+ext+myv.PRD_EXT+'.png')

    fig_gp.set_figwidth(10)
    plt.title('Teff = %.0f K   |    Logg = %.2f dex   |    Fe/H = %.2f dex   |    Pmag = %.2f years   |    < M > = %.1f %% (A = %.1f %%)    '%(vec.star_teff, vec.star_logg, vec.star_feh, vec.out_gp_pmag, vec.out_gp_meanmag, vec.out_gp_ampmag))
    plt.xlim(1965,2040)
    plt.ylim(-15,90)
    plt.legend()
    plt.axhline(y=0,color='k',ls='-',alpha=0.7,lw=1)
    plt.savefig(dir_root.replace(ins,'ALLINS_MERGED')+'Finch_magnetic_cycle_GP_fixed_axis'+ext+myv.PRD_EXT+'.png')

    if (not vec.out_convergence_flag)&(len(vec.bin.x)>3):
        vec.bin.fit_line()
        trend = myc.tableXY(myf.conv_time(vec.bin.x)[1]-2000,vec.bin.y,vec.bin.yerr)
        trend.fit_line(recenter=False)
        model = (vec.out_gp_model[0]-2000)*trend.lin_slope_w + trend.lin_intercept_w
        model[vec.out_gp_model[0]<=(np.min(trend.x)+2000)] = vec.out_gp_model[1][vec.out_gp_model[0]<=(np.min(trend.x)+2000)]
        model[vec.out_gp_model[0]>=(np.max(trend.x)+2000)] = vec.out_gp_model[1][vec.out_gp_model[0]>=(np.max(trend.x)+2000)]
        vec.out_gp_model[1] = model

    export3 = myc.tableXY(myf.conv_time(vec.out_gp_model[0])[0],vec.out_gp_model[1],vec.out_gp_model[2])
    export3.export(dir_root.replace(ins,'ALLINS_MERGED')+'Finch_%s_GP_model.csv'%(proxy_name),format='csv',columns=['jdb','proxy','proxy_std','qc'])

    if predict_samples is not None:
        export3.x = myf.conv_time(export3.x)[1]
        export3.interpolate(new_grid=np.arange(predict_samples[0],predict_samples[1],0.25))
        plt.figure(figsize=(10,6))
        plt.subplot(2,1,1)
        plt.title('Predicted activity level')
        plt.plot(export3.x,export3.y,color='k',ls='-',alpha=0.7)
        plt.fill_between(export3.x,export3.y-export3.yerr,export3.y+export3.yerr,color='k',alpha=0.2)
        samples_mhk = []
        nb = int(99999/len(export3.x))
        for i,j in zip(export3.y,export3.yerr):
            samples_mhk.append(np.random.randn(nb)*j+i)
        samples_mhk = np.ravel(samples_mhk)
        plt.xlim(predict_samples[0],predict_samples[1])
        plt.ylabel('MHK [%]',fontsize=13)
        plt.xlabel('Date [year]',fontsize=13)

        plt.scatter(export3.x[::4],export3.y[::4],color='k')
        for i,j in zip(export3.x[::4],export3.y[::4]):
            plt.text(i,j,'%.1f %%'%(j),color='k',ha='left',va='bottom')

        plt.subplot(2,1,2)
        pby,pbx = np.histogram(samples_mhk,bins=np.arange(-41,200,1),density=True)
        pbx = 0.5*(pbx[1:]+pbx[0:-1])
        plt.fill_between(pbx,pby,alpha=0.2,color='k',label=r'MHK = %.1f $\pm$ %.1f'%(np.nanmean(samples_mhk),myf.mad(samples_mhk))+'%')
        plt.plot(pbx,pby,color='k')
        plt.xlabel('MHK [%]',fontsize=13)
        plt.ylim(0,None)
        plt.xlim(-40,200)
        plt.axvline(x=0,ls=':',color='k')
        plt.axvline(x=50,ls='-.',color='k',label='active stars',lw=1)
        plt.legend()
        plt.xticks(np.arange(-25,200,25))
        plt.subplots_adjust(hspace=0.35)
        plt.savefig(dir_root.replace(ins,'ALLINS_MERGED')+'MHK_samples_%.1f_%.1f'%(predict_samples[0],predict_samples[1])+ext+myv.PRD_EXT+'.png')

    output = [
        FINCH_Pmag,
        FINCH_Pmag_GP,
        FINCH_Mmag_GP,
        FINCH_Kmag_GP]+vec.out_gp_predict
    
    return output


def import_spectrum(file,sub_dico='matching_diff'):
    file = pd.read_pickle(file)
    spec = myc.tableXY(file['wave'],file['flux']/file[sub_dico]['continuum_linear'],0*file['wave'])
    spec.filename = file
    return spec

def master_spectrum(files, rv_shift, rv_sys, plot=False, sub_dico='matching_diff'):
    
    #wave_grid, sts, sts_err = import_sts(files, sub_dico=sub_dico)
    wave_grid = files[0]/100.

    rv_syst = rv_sys*1000
    shift_ms = rv_shift + rv_syst
        
    master = np.zeros_like(wave_grid, dtype='float32')
    chunks = np.array_split(np.arange(len(wave_grid)), 5)

    for idx in chunks:
        wave = wave_grid[idx]
        sts = np.empty((len(files[1]), len(wave)), dtype='float32')
        for m, rv in enumerate(shift_ms):
            sts[m] = myf.interpolate_rv_shift(wave,files[1][:,idx][m] / 10000., rv=rv, fill_value=0, kind='linear')
        sts[sts==0] = np.nan
        master[idx] = np.nanmedian(sts, axis=0)

    master[master!=master] = 0
    master = myc.tableXY(wave_grid, master, np.zeros_like(wave_grid))

    if plot:
        plt.figure('master')
        master.plot()
    
    return master

def import_sts(files, rv_shift=None, err=False, sub_dico='matching_diff', scale=True):
    "rv_shift in m/s"

    if scale:
        wave_grid = np.round(files[0]/100.,2)
        sts = (files[1]/10000.).astype('float32')
    else:
        wave_grid = files[0]
        sts = files[1]
    sts_err = None
    if rv_shift is None:
        rv_shift = np.zeros(len(sts))

    chunks = np.array_split(np.arange(len(wave_grid)), 5)
    for idx in chunks:
        count = -1
        for f, rv in zip(sts,rv_shift):
            count+=1
            if rv!=0:
                sts[count][idx] = myf.interpolate_rv_shift(wave_grid[idx],f[idx], rv=rv, fill_value=1, kind='linear')

    return wave_grid, sts, sts_err    

def create_sts(files, grid = np.round(np.arange(3900,6830.001,0.01),2), sub_dico='matching_diff', material=None, rv_sys=0):
    sts = []
    myv.vprint('\n [INFO] Creating the npy Spectrum time-series...\n')
    if material is not None:
        myv.vprint(' [INFO] Using material for extra continuum correction...\n')
        grid = material['wave']
    
    mask_activity = np.zeros(len(grid)).astype('bool')
    for line in [myv.Ca2K,myv.Ca2H,myv.Ha,myv.Hb,myv.Hc,myv.Hd]:
        mask_activity[abs(grid-myf.doppler_r(line[0],rv_sys*1000)[0])<20] = True

    for f in files:
        rassine_file = pd.read_pickle(f)
        spec_norm = rassine_file['flux']/rassine_file[sub_dico]['continuum_linear']
        wave = rassine_file['wave']
        spec = myc.tableXY(wave,spec_norm,0*spec_norm)
        spec.interpolate(new_grid=grid,method='linear',fill_value=0)
        if material is not None:
            spec.y = spec.y*material['correction_factor']
        
        spec.y[(spec.y>2)&(~mask_activity)] = 1
        spec.y[spec.y<0] = 0
        sts.append((10000*spec.y).astype('uint16'))
    
    return (grid*100).astype('int'),np.array(sts)


def read_hermes(file,dir_root,force=False,debug=False):
    fname = file.split('/')[-1]
    outname = dir_root+'WORKSPACE/RASSINE_Stacked_spectrum_B0.00_'+fname.replace('.fits','.p')
    if (not os.path.exists(outname))|(force):
        t = fits.open(file)
        cdelt1 = t[0].header['CDELT1']
        cval1 = t[0].header['CRVAL1']
        flux = t[0].data[0]
        wave = np.arange(cval1,cval1+cdelt1*len(flux)-0.0001,cdelt1)
        flux_std = 0*flux
        wave_grid = np.round(np.arange(3800,6900.001,0.01),2) #don-t use redder than 6900 (lighter file)
        spec = myc.tableXY(wave,flux,flux_std)
        spec.interpolate(new_grid=wave_grid,method='linear',fill_value=0)
        spec = rassine_normalise(spec)
        if debug:
            plt.plot(spec.x,spec.y/spec.rassine_continuum.y)

        if len(spec.x)==len(spec.rassine_continuum.y):
            export = {
                'wave':spec.x,
                'flux':spec.y,
                'matching_diff':{'continuum_linear':spec.rassine_continuum.y},
                'parameters':{'arcfiles':[file]}}
            pickle.dump(export,open(outname,'wb'))
        else:
            print('[WARNING] RASSINE continuum size wrong, potential multiprocessing issue')
    return 1

def read_neid(file,dir_root,force=False,debug=False):
    fname = file.split('/')[-1]
    outname = dir_root+'WORKSPACE/RASSINE_Stacked_spectrum_B0.00_'+fname.replace('.fits','.p')
    if (not os.path.exists(outname))|(force):
        t = fits.open(file)
        berv = t[0].header['SSBRV100']
        wave = t[7].data #check 8 or 9 too
        flux = t[1].data #check 8 or 9 too
        flux_err = np.sqrt(t[4].data) #check 8 or 9 too
        
        grid = np.round(np.arange(3800,6900.001,0.01),2) #don-t use redder than 6900 (lighter file)
        merged = np.ones(len(grid))*np.nan
        merged_weight = np.zeros(len(grid))
        for i in tqdm(np.arange(len(wave))):
            w = wave[i] ; f = flux[i] ; f2 = flux_err[i]
            if (np.max(w)>np.min(grid))&(np.min(w)<np.max(grid)):
                s = myc.tableXY(myf.conv_void_air(w),f,f2)
                s.x = myf.doppler_r(s.x,berv*1000)[0]
                s2 = s.copy()
                for k in range(3):
                    s2.find_max()
                    s2.max_extremum.smooth(box_pts=5)
                    s2.max_extremum.interpolate(s2.x,method='linear')
                    s2.y[s2.y<s2.max_extremum.y] = 0
                blaze = s2.max_extremum
                left = blaze.y[0]
                right = blaze.y[-1]
                mini = np.nanmin(blaze.y)
                maxi = np.nanmax(blaze.y)
                blaze.interpolate(s.x,method='linear')
                s.yerr /= blaze.y
                s.y /= blaze.y

                s.interpolate(new_grid=grid,fill_value=np.nan,method='linear')
                s_weight = 1/s.yerr**2
                merged = np.nansum(np.array([merged*merged_weight,s.y*s_weight]),axis=0)/np.nansum(np.array([merged_weight,s_weight]),axis=0)
                merged_weight = np.nansum(np.array([merged_weight,s_weight]),axis=0)

        merged[merged!=merged] = 0
        spec = myc.tableXY(grid,merged,0*merged)

        if debug:
            plt.plot(spec.x,spec.y/spec.rassine_continuum.y)

        export = {
            'wave':spec.x,
            'flux':spec.y,
            'matching_diff':{'continuum_linear':np.ones(len(spec.y))},
            'parameters':{'arcfiles':[file]}}
        pickle.dump(export,open(outname,'wb'))
    return 1

def read_yarara(file,dir_root,force=False,debug=False):
    fname = file.split('/')[-1]
    outname = dir_root+'WORKSPACE/RASSINE_'+fname
    if (not os.path.exists(outname))|(force):
        t = pd.read_pickle(file)
        wave = np.round(np.array(t['wave']),2)
        flux = np.array(t['reference_spectrum'])

        export = {
            'wave':wave,
            'flux':flux,
            'matching_diff':{'continuum_linear':np.ones(len(wave))},
            'parameters':{'arcfiles':[None],'jdb':0,'berv':0,'SNR_5500':9999}}
        pickle.dump(export,open(outname,'wb'))    
    return 1

def read_espresso(file,dir_root,force=False,debug=False):
    fname = file.split('/')[-1]
    outname = dir_root+'WORKSPACE/RASSINE_Stacked_spectrum_B0.00_'+fname.replace('.fits','.p')
    if (not os.path.exists(outname))|(force):
        t = fits.open(file)
        wave = t[1].data['wavelength_air']
        flux = t[1].data['flux']
        flux_std = t[1].data['error']
        wave_grid = np.arange(np.round(np.min(wave),2),np.round(np.max(wave),2),0.01)
        spec = myc.tableXY(wave,flux,flux_std)
        spec.interpolate(new_grid=wave_grid,method='linear')
        spec = rassine_normalise(spec)
        if debug:
            plt.plot(spec.x,spec.y/spec.rassine_continuum.y)
        
        if len(spec.x)==len(spec.rassine_continuum.y):
            export = {
                'wave':spec.x,
                'flux':spec.y,
                'matching_diff':{'continuum_linear':spec.rassine_continuum.y},
                'parameters':{'arcfiles':[file]}}
            pickle.dump(export,open(outname,'wb'))
        else:
            print('[WARNING] RASSINE continuum size wrong, potential multiprocessing issue')
    return 1

def read_eso(file, dir_root, ins, rv_shift=0, force=False, debug=False):
    fname = file.split('/')[-1]
    outname = dir_root+'WORKSPACE/RASSINE_Stacked_spectrum_B0.00_'+fname.replace('.fits','.p')
    if (not os.path.exists(outname))|(force):
        t = fits.open(file)
        data = t[1].data
        wave = data['wave'][0]
        if ins[0:4]=='ESPR': 
            wave = myf.conv_void_air(wave) # new drs in the void

        if (ins[0:4]=='UVES')&(np.max(wave)<1000):
            wave*=10 #nm instead of angstrom

        if rv_shift!=0:
            wave = myf.doppler_r(wave,rv_shift)[0]
        
        try:
            flux = data['flux'][0]
        except:
            flux = data['flux_reduced'][0]

        flux[0:2] = 0 ; flux[-2:] = 0
        borders = myf.clustering(wave,10,1)[-1] #in case of hole, set 0 around the hole to avoid interpolation
        for b in borders[:-1,1]:
            flux[b-5:b+6] = 0

        flux_std = flux*0 #data['err'][0]
        wave_grid = np.round(np.arange(3800,6900.001,0.01),2)
        #wave_grid = np.arange(np.round(np.min(wave),2),np.round(np.max(wave),2),0.01)
        spec = myc.tableXY(wave,flux,flux_std)
        spec.interpolate(new_grid=wave_grid,method='linear')

        spec = rassine_normalise(spec)
        if debug:
            plt.plot(spec.x,spec.y/spec.rassine_continuum.y)
        
        if len(spec.x)==len(spec.rassine_continuum.y):
            export = {
                'wave':spec.x,
                'flux':spec.y,
                'matching_diff':{'continuum_linear':spec.rassine_continuum.y},
                'parameters':{'arcfiles':[file]}}
            pickle.dump(export,open(outname,'wb'))
        else:
            print('[WARNING] RASSINE continuum size wrong, potential multiprocessing issue')
    return 1

def read_ia2(file,dir_root,force=False,debug=False):
    fname = file.split('/')[-1]
    outname = dir_root+'WORKSPACE/RASSINE_Stacked_spectrum_B0.00_'+fname.replace('.fits','.p')
    if (not os.path.exists(outname))|(force):
        t = fits.open(file)
        data = t[1].data
        if fname[0:2]=='r.': #new DRS
            wave = myf.conv_void_air(data['wavelength'])
        else:
            wave = data['wavelength']
        flux = data['flux_cal']
        flux_std = data['error_cal']
        wave_grid = np.arange(np.round(np.min(wave),2),np.round(np.max(wave),2),0.01)
        spec = myc.tableXY(wave,flux,flux_std)
        spec.interpolate(new_grid=wave_grid,method='linear')
        spec = rassine_normalise(spec)
        if debug:
            plt.plot(spec.x,spec.y/spec.rassine_continuum.y)
        
        if len(spec.x)==len(spec.rassine_continuum.y):
            export = {
                'wave':spec.x,
                'flux':spec.y,
                'matching_diff':{'continuum_linear':spec.rassine_continuum.y},
                'parameters':{'arcfiles':[file]}}
            pickle.dump(export,open(outname,'wb'))
        else:
            print('[WARNING] RASSINE continuum size wrong, potential multiprocessing issue')
    return 1

def read_static(file, dir_root, cval1, cdelt1, force=False, debug=False):
    fname = file.split('/')[-1]
    outname = dir_root+'WORKSPACE/RASSINE_Stacked_spectrum_B0.00_'+fname.replace('.fits','.p')
    if (not os.path.exists(outname))|(force):
        t = fits.open(file)
        flux = t[0].data
        wave = np.arange(cval1,cval1+cdelt1*len(flux)-0.0001,cdelt1)
        flux_std = 0*flux
        w0 = np.where(np.cumsum(flux)!=0)[0][0]
        flux = flux[w0:] ; wave = wave[w0:]

        wave_grid = np.round(np.arange(3800,6900.001,0.01),2)

        spec = myc.tableXY(wave,flux,flux_std)
        spec.interpolate(new_grid=wave_grid,method='linear')
        spec = rassine_normalise(spec)
        
        if debug:
            plt.plot(spec.x,spec.y/spec.rassine_continuum.y)

        if len(spec.x)==len(spec.rassine_continuum.y):
            export = {
                'wave':spec.x,
                'flux':spec.y,
                'matching_diff':{'continuum_linear':spec.rassine_continuum.y},
                'parameters':{'arcfiles':[file]}}
            pickle.dump(export,open(outname,'wb'))
        else:
            print('[WARNING] RASSINE continuum size wrong, potential multiprocessing issue')
    return 1


def read_gr8(file, dir_root, instrument, force=False, debug=False):
    fname = file.split('/')[-1]
    outname = dir_root+'WORKSPACE/RASSINE_Stacked_spectrum_B0.00_'+fname.replace('.fits','.p')
    if (not os.path.exists(outname))|(force):
        t = fits.open(file)
        if (instrument=='FIES')|(instrument=='HERMES'):
            flux = np.round(t[1].data['flux'],8).astype('float')
            wave = np.round(t[1].data['wavelength'],2).astype('float')
        elif (instrument=='UVES')|(instrument=='FEROS'):
            flux = np.round(t[1].data['flux'],8).astype('float')
            wave = np.round(t[1].data['wavelength']*10,2).astype('float')
        wave_first = np.where(flux!=0)[0][0]
        wave = wave[wave_first:]
        flux = flux[wave_first:]
        if wave[0]>3700:
            flux = flux[wave>3700]
            wave = wave[wave>3700]
        flux_std = 0*flux
        spec = myc.tableXY(wave,flux,flux_std)

        wmin = np.round(wave[0],0)+1
        wmax = np.round(wave[-1],0)-1
        grid_static = np.round(np.arange(wmin,wmax,0.01),2)

        spec.interpolate(new_grid=grid_static,replace=True,method='cubic')
        spec = rassine_normalise(spec)
        if debug:
            plt.plot(spec.x,spec.y/spec.rassine_continuum.y)

        if len(spec.x)==len(spec.rassine_continuum.y):
            export = {
                'wave':spec.x,
                'flux':spec.y,
                'matching_diff':{'continuum_linear':spec.rassine_continuum.y},
                'parameters':{'arcfiles':[file]}}
            pickle.dump(export,open(outname,'wb'))
        else:
            print('[WARNING] RASSINE continuum size wrong, potential multiprocessing issue')
    return 1

def pepsi_summary(files,output_dir):
    summary = []
    for f in files:
        file = fits.open(f)
        w0 = int(file[1].data['Arg'][0])
        w1 = int(file[1].data['Arg'][-1])
        snr = np.nanmedian(1/np.sqrt(file[1].data['Var']))
        summary.append([f,file[0].header['JD-OBS']-2400000,file[0].header['ORDER1'],file[0].header['ORDER2'],w0,w1,file[0].header['SNR']])
    summary = pd.DataFrame(summary,columns=['fileroot','jdb','order1','order2','w0','w1','SNR'])
    summary['iso'] = myf.conv_time(list(summary['jdb']))[2]
    summary.sort_values(by=['jdb','order1'],inplace=True)
    summary = summary.reset_index(drop=True)
    summary['night'] = summary['jdb'].astype('int')

    grid_pepsi = np.round(np.arange(3850,9000.001,0.01),2)
    
    new_summary = []
    for m,n in enumerate(np.unique(summary['night'])):
        sub = summary.loc[summary['night']==n].reset_index(drop=True)
        jdb = np.mean(sub['jdb'])
        matrix = []
        matrix_err = []
        snr = []
        for s in sub['fileroot']:
            wave = fits.open(s)[1].data['Arg']
            flux = fits.open(s)[1].data['Fun']
            flux_err = fits.open(s)[1].data['Var']
            flux[0:2] = 0 ; flux[-2:] = 0
            snr.append(1/np.sqrt(np.nanmedian(flux_err)))
            spec = myc.tableXY(wave,flux)
            spec.interpolate(new_grid=grid_pepsi,method='linear',fill_value=0)
            matrix.append(spec.y)
            matrix_err.append(spec.y*0+snr[-1]**4)
            #plt.plot(wave,flux)
        matrix = np.array(matrix)
        matrix_err = np.array(matrix_err)
        matrix_err = 1/np.array(matrix_err)**2
        matrix_err[matrix==0] = 0
        matrix_err /= np.sum(matrix_err,axis=0)
        snr = np.array(snr)
        stack = np.sum(matrix*matrix_err,axis=0)
        stack[stack!=stack] = 0
        #plt.plot(grid_pepsi,stack+m*2,color='k')
        snr_stack = np.sqrt(np.sum(snr**2))

        ref_fits = sub['fileroot'][0]
        ref_iso = sub['iso'][0]
        fits_file = fits.open(ref_fits)
        fits_file[0].header['SNR'] = snr_stack
        fits_file[0].header['JD-OBS'] = jdb+2400000

        cols = [
            fits.Column(name='Arg',  format='D', array=grid_pepsi),
            fits.Column(name='Fun',  format='D', array=stack),
            fits.Column(name='Var',  format='D', array=0*stack),
            fits.Column(name='Mask', format='L', array=(0*stack).astype(bool))
        ]

        hdu = fits.BinTableHDU.from_columns(cols)
        fits_file[1] = hdu

        output_file = output_dir+'RAW/PEPSI.'+ref_iso+'.fits'
        fits_file.writeto(output_file, overwrite=True)
        new_summary.append([output_file,jdb,snr_stack])
        print(' [INFO] Spectrum %s created!'%(output_file))
    new_summary = pd.DataFrame(new_summary,columns=['fileroot','jdb','snr'])
    return new_summary

def read_pepsi(file,dir_root,force=False,debug=False):
    fname = file.split('/')[-1]
    outname = dir_root+'WORKSPACE/RASSINE_Stacked_spectrum_B0.00_'+fname.replace('.fits','.p')
    if (not os.path.exists(outname))|(force):
        t = fits.open(file)
        flux = t[1].data['Fun']
        wave = t[1].data['Arg']
        flux_std = 0*flux
        spec = myc.tableXY(wave,flux,flux_std)
        spec = rassine_normalise(spec)
        if debug:
            plt.plot(spec.x,spec.y/spec.rassine_continuum.y)

        if len(spec.x)==len(spec.rassine_continuum.y):
            export = {
                'wave':spec.x,
                'flux':spec.y,
                'matching_diff':{'continuum_linear':spec.rassine_continuum.y},
                'parameters':{'arcfiles':[file]}}
            pickle.dump(export,open(outname,'wb'))
        else:
            print('[WARNING] RASSINE continuum size wrong, potential multiprocessing issue')
    return 1

def read_sophie(file,dir_root,force=False,debug=False):
    fname = file.split('/')[-1]
    outname = dir_root+'WORKSPACE/RASSINE_Stacked_spectrum_B0.00_'+fname.replace('.fits','.p')
    if (not os.path.exists(outname))|(force):
        try:
            t = fits.open(file)
            cdelt1 = t[0].header['CDELT1']
            cval1 = t[0].header['CRVAL1']
            flux = t[0].data
        except:
            print('[WARNING] Fits file %s could not be open correctly'%(file))
            return 0
        wave = np.arange(cval1,cval1+cdelt1*len(flux)-0.0001,cdelt1)
        flux_std = 0*flux
        spec = myc.tableXY(wave,flux,flux_std)
        spec = rassine_normalise(spec)
        if debug:
            plt.plot(spec.x,spec.y/spec.rassine_continuum.y)

        if len(spec.x)==len(spec.rassine_continuum.y):
            export = {
                'wave':spec.x,
                'flux':spec.y,
                'matching_diff':{'continuum_linear':spec.rassine_continuum.y},
                'parameters':{'arcfiles':[file]}}
            pickle.dump(export,open(outname,'wb'))
        else:
            print('[WARNING] RASSINE continuum size wrong, potential multiprocessing issue')
    return 1

def check_and_update_path(dir_root):
    if os.path.exists(dir_root+'WORKSPACE/Analyse_summary.csv'):
        processed = glob.glob(dir_root+'WORKSPACE/RASSINE*.p')
        if len(processed)!=0:
            summary = pd.read_csv(dir_root+'WORKSPACE/Analyse_summary.csv',index_col=0)
            path = processed[0].split('/RASSINE_')[0]
            summary['filename'] = np.array([path+'/'+f.split('WORKSPACE/')[-1] for f in summary['filename']])
            summary.to_csv(dir_root+'WORKSPACE/Analyse_summary.csv')

def query_value(header,kws):
    output = []
    for kw in kws:
        try:
            output.append(header[kw])
        except:
            output.append(np.nan)
    return output

def ra_to_deg(ra,ra_ref=None):
    if ra==ra:
        try:
            ra = float(ra)
            h  = int(ra // 10000)
            m  = int((ra % 10000) // 100)
            s  = ra % 100
            return (h + m/60 + s/3600) * 15
        except:
            if ra_ref is not None:
                return ra_ref
    else:
        return np.nan

def dec_to_deg(dec,dec_ref=None):
    if dec==dec:
        try:
            dec = float(dec)
            sign = -1 if dec < 0 else 1
            dec = abs(dec)
            d  = int(dec // 10000)
            m  = int((dec % 10000) // 100)
            s  = dec % 100
            return sign * (d + m/60 + s/3600)
        except:
            if dec_ref is not None:
                return dec_ref
    else:
        return np.nan

#@myf.time_step
def get_vmacro(teff,logg,feh,source='Cretignier+26'):
    """This is the gaussian width (not FWHM) for a Gaussian macroturbulence approximation"""
    if source=='Doyle+14': #only valid between teff = [5200-6400], logg = [4.0-4.6]
        Teff = np.arange(5250,6400,200)
        Vmacro = 3.21 + 2.33e-3*(Teff-5777)+2.0e-6*(Teff-5777)**2-2.0*(logg-4.44)
        Teff = [4200,4400,4600,4800,5000] + list(Teff)
        Vmacro = [0.45,0.60,0.90,1.00,1.20] + list(Vmacro/2)
        #Teff = [4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000]
        #vmacro = [1.00, 1.00, 1.00, 1.00, 1.19, 1.77, 2.56, 3.20, 3.79]
        calib = myc.tableXY(Teff,Vmacro)
        calib.interpolate(new_grid=np.array([teff]),method='linear')
        value = np.round(calib.y[0],3)
        value = (value,value,value)
    elif source=='Bruntt+10':
        value = np.round(myf.find_turbulence(teff,logg)[1],3)
        value = (value,value,value)
    elif source=='Cretignier+26': # obtained with GARFIELD on instrument PSF removed
        vmacro_teff = myc.tableXY([3500,3750,4000,4250,4500,4750,5000,5250,5500,5750,6000,6250,6500,6750],[3.00,3.50,3.90,4.20,4.20,3.90,3.70,3.70,3.90,4.20,4.70,5.80,7.5,7.5]) ; vmacro_teff.null()
        vmacro_feh = myc.tableXY([-1.0,-0.75,-0.50,-0.25,0.00,0.25],[0.00,0.00,0.00,0.00,0.25,0.60]) ; vmacro_feh.null()
        vmacro_logg = myc.tableXY([3.6, 3.8, 4.0, 4.2, 4.4, 4.6],[1.00,0.75,0.40,0.10,-0.15,-0.25]) ; vmacro_logg.null()
        vmacro_teff.interpolate(new_grid=np.array([teff]),method='linear',replace=False)
        vmacro_feh.interpolate(new_grid=np.array([feh]),method='linear',replace=False)
        vmacro_logg.interpolate(new_grid=np.array([logg]),method='linear',replace=False)
        value = vmacro_teff.y_interp[0] + vmacro_feh.y_interp[0] + vmacro_logg.y_interp[0]
        value = (value,value,value)
    value = {'G2':value[0],'Garfield':value[1],'Kitty':value[2]}
    return value

    
def extract_header(files, instru, debug=False, ra=None, dec=None, sources=None):
    instrument = instru.split('_')[0]
    ins = instrument[0:5]
    if files[0].split('/')[-1][0:7]=='RASSINE':
        ins = 'RASSINE'

    if (ins=='HARPS')&(instru.split('_')[-1]=='3.5'):
        ins = 'harps'

    if (ins=='HARPS')&(instru.split('_')[-1]=='3.8'):
        ins = 'harps'

    if ins=='NEID-':
        ins='NEID'

    all_infos = []
    kws = {'SOPHI':{'HIERARCH OHP DRS BJD':'rjd', 'HIERARCH OHP DRS BERV':'berv', 'HIERARCH OHP DRS CAL EXT SN30':'snr', 'HIERARCH OHP TARG ALPHA':'RA', 'HIERARCH OHP TARG DELTA':'DEC'},
           'NEID':{'OBSJD':'rjd', 'SSBRV100':'berv', 'EXTSNR':'snr', 'QRA':'RA', 'QDEC':'DEC'},
           'harps':{'ESO DRS BJD':'rjd', 'ESO DRS BERV':'berv', 'ESO DRS SPE EXT SN50':'snr', 'ESO TEL TARG ALPHA':'RA', 'ESO TEL TARG DELTA':'DEC'}, #old DRS (3.5)
           'HARPS':{'ESO QC BJD':'rjd', 'ESO QC BERV':'berv', 'ESO QC ORDER50 SNR':'snr', 'ESO TEL TARG ALPHA':'RA', 'ESO TEL TARG DELTA':'DEC'}, #new DRS (3.3.6)
           'HARPN':{'MJD-OBS':'rjd', 'TNG QC BERV':'berv', 'TNG QC ORDER50 SNR':'snr', 'TNG TEL TARG ALPHA':'RA', 'TNG TEL TARG DELTA':'DEC'}, #new DRS (3.0.1)
           'PEPSI':{'JD-TDB':'rjd', 'SSBVEL':'berv', 'SNR':'snr', 'RA':'RA', 'DEC':'DEC'},
           'CORAL':{'ESO DRS BJD':'rjd', 'ESO DRS BERV':'berv', 'ESO DRS SPE EXT SN50':'snr', 'ESO TEL TARG ALPHA':'RA', 'ESO TEL TARG DELTA':'DEC'},
           'ESPRE':{'HIERARCH ESO QC BJD':'rjd', 'HIERARCH ESO QC BERV':'berv', 'HIERARCH ESO QC ORDER100 SNR':'snr', 'HIERARCH ESO TEL1 TARG ALPHA':'RA', 'HIERARCH ESO TEL1 TARG DELTA':'DEC'},
           'FEROS':{'MJD-OBS':'rjd', 'HIERARCH ESO DRS BARYCORR':'berv', 'SNR':'snr', 'RA':'RA', 'DEC':'DEC'},
           'FIES':{'I-HJD':'rjd', 'I-VBAR':'berv', 'I-SNR':'snr','I-RA':'RA', 'I-DEC':'DEC'},
           'HERME':{'I-HJD':'rjd', 'I-VBAR':'berv', 'I-SNR':'snr','I-RA':'RA', 'I-DEC':'DEC'},
           'UVES':{'MJD-OBS':'rjd', 'HIERARCH ESO DRS BARYCORR':'berv', 'SNR':'snr', 'RA':'RA', 'DEC':'DEC'},
           'GR8':{'ESTSNR':'snr','RA':'RA', 'DEC':'DEC'},
           'ESO':{'MJD-OBS':'rjd','SNR':'snr','RA':'RA', 'DEC':'DEC'},
           'RASSINE':{'jdb':'rjd', 'berv':'berv', 'SNR_5500':'snr'},
           'YARARA':{'jdb':'rjd', 'berv':'berv', 'SNR_5500':'snr'},
           'TBD':{'KEYWORD BJD':'rjd', 'KEYWORD BERV':'berv', 'KEYWORD SNR':'snr', 'KEYWORD ALPHA':'RA', 'KEYWORD DELTA':'DEC'},
           }

    for file,source in zip(files,sources):

        if source=='GR8':
            ins = 'GR8'

        if (ins[0:5]=='ESPRE')&(source=='ESO'):
            ins = 'ESO'

        if ins=='RASSINE':
            header = pd.read_pickle(file)['parameters']
        elif source=='YARARA':
            header = {'rjd':0,'berv':0,'SNR_5500':9999}
            ins = 'YARARA'
        else:
            header = fits.open(file)[0].header
        infos = query_value(header,list(kws[ins].keys()))
        all_infos.append(infos)
    all_infos = np.array(all_infos)
    summary = pd.DataFrame(all_infos,columns=list(kws[ins].values()))
    if debug:
        snaky_help()
        print(summary, ins)

    if ins=='GR8':
        summary['rjd'] = np.nan
        summary['berv'] = np.nan
    if ins=='FEROS':
        summary['rjd'] = summary['rjd'].astype('float') + 2400000
    if ins=='UVES':
        summary['rjd'] = summary['rjd'].astype('float') + 2400000
    if ins=='ESO':
        summary['rjd'] = summary['rjd'].astype('float') + 2400000
        summary['berv'] = np.nan
    if ins=='PEPSI':
        for i in summary.index:
            RA = summary.loc[i,'RA'].replace(':','')
            summary.loc[i,'RA'] = RA
            DEC = summary.loc[i,'DEC'].replace(':','')
            summary.loc[i,'DEC'] = DEC
        summary['RA'] = np.round(np.array([ra_to_deg(ra2) for ra2 in np.array(summary['RA'])]),6)
        summary['DEC'] = np.round(np.array([dec_to_deg(dec2) for dec2 in np.array(summary['DEC'])]),6)
    if (ins=='SOPHI'):
        if (ra is None)|(dec is None):
            for i in summary.index:
                RA = summary.loc[i,'RA']
                length = len(str(RA).split('.')[0])
                RA = '0'*(6-length)+str(RA)
                summary.loc[i,'RA'] = RA
            summary['RA'] = np.round(np.array([ra_to_deg(ra2) for ra2 in np.array(summary['RA'])]),6)
            summary['DEC'] = np.round(np.array([dec_to_deg(dec2) for dec2 in np.array(summary['DEC'])]),6)
        else:
            summary['RA'] = ra
            summary['DEC'] = dec
    if (ins=='NEID'):
        for i in summary.index:
            RA = summary.loc[i,'RA'].replace(':','')
            summary.loc[i,'RA'] = RA
            DEC = summary.loc[i,'DEC'].replace(':','')
            summary.loc[i,'DEC'] = DEC
        summary['RA'] = np.round(np.array([ra_to_deg(ra2) for ra2 in np.array(summary['RA'])]),6)
        summary['DEC'] = np.round(np.array([dec_to_deg(dec2) for dec2 in np.array(summary['DEC'])]),6)
    if (ins=='HARPN'):
        summary['rjd'] = summary['rjd'].astype('float') + 2400000
        for i in summary.index:
            RA = summary.loc[i,'RA'].replace('h','').replace('m','')
            summary.loc[i,'RA'] = RA
            DEC = summary.loc[i,'DEC'].replace(':','')
            summary.loc[i,'DEC'] = DEC
        summary['RA'] = np.round(np.array([ra_to_deg(ra2) for ra2 in np.array(summary['RA'])]),6)
        summary['DEC'] = np.round(np.array([dec_to_deg(dec2) for dec2 in np.array(summary['DEC'])]),6)
    if (ins=='ESPRE')|(ins=='HARPS')|(ins=='harps'): 
        summary['RA'] = np.round(np.array([ra_to_deg(ra2) for ra2 in np.array(summary['RA'])]),6)
        summary['DEC'] = np.round(np.array([dec_to_deg(dec2) for dec2 in np.array(summary['DEC'])]),6)
    if ins=='CORAL':
        summary['RA'] = np.round(summary['RA'].astype('float'),6)
        summary['DEC'] = np.round(summary['DEC'].astype('float'),6)
    if ins=='RASSINE':
        missing_time = (summary['rjd']!=summary['rjd'])
        myv.vprint(' [INFO] Nb of missing time = %.0f'%(np.sum(missing_time)))
        if np.sum(missing_time)!=0: #search ut in filename
            if instrument[0:4]=='NEID':
                uttime = np.array([f[-17:-2] for f in files])
                uttime = np.array([f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:{s[11:13]}:{s[13:15]}.000" for s in uttime])
            else:
                uttime = np.array([f[-25:-2] for f in files])
            year = np.array([ut[0:4] for ut in uttime])
            year_num = np.array([float(y) if y.isdigit() else np.nan for y in year])
            valid = (year_num>1950)&(year_num<2050)
            missing_time = missing_time[valid]
            rjd = np.ones(len(summary))[missing_time]*np.nan
            if np.sum(valid)>0:
                rjd = myf.conv_time(uttime[valid])[0]
                summary.loc[missing_time,'rjd'] = rjd
            missing_time = (summary['rjd']!=summary['rjd'])
            myv.vprint(' [INFO] Nb of missing time after UT search = %.0f'%(np.sum(missing_time)))
        
        summary['rjd'] = summary['rjd'].astype('float') + 2400000
        summary['RA'] = ra
        summary['DEC'] = dec        

    if (instrument=='HARPS03')|(instrument=='HARPS15'):
        instrument = 'HARPS'

    if (instrument=='ESPRESSO18')|(instrument=='ESPRESSO19'):
        instrument = 'ESPRESSO'

    if (instrument=='CORALIE98')|(instrument=='CORALIE07')|(instrument=='CORALIE14'):
        instrument = 'CORALIE'

    if ra is not None:
        summary['RA'] = ra
        summary['DEC'] = dec

    missing_time = (summary['rjd']!=summary['rjd'])
    if np.sum(missing_time)!=0:
        summary.loc[missing_time, 'rjd'] = np.arange(np.sum(missing_time))

    if debug:
        print(summary)
        print(summary.loc[0])

    summary['berv'] = np.round(summary['berv'].astype('float'),6)
    summary['snr'] = np.round(summary['snr'].astype('float'),1)
    ra_deg = np.nanmedian(summary['RA'])
    dec_deg = np.nanmedian(summary['DEC'])
    if np.sum(~missing_time)!=0:
        obstime = Time(summary['rjd'].astype('float')[~missing_time], format='jd', scale='utc')
        obstime_utc = obstime.utc.isot
        berv = get_berv(ra_deg, dec_deg, obstime_utc, instrument).value
        summary.loc[~missing_time,'rjd'] = summary.loc[~missing_time,'rjd'].astype('float') - 2400000
        summary.loc[~missing_time,'berv_computed'] = np.round(berv,4)
    else:
        summary['berv_computed'] = np.nan

    return summary


def rassine_normalise(spec, min_radius=4.0, max_radius=76.0):
    spec.fit_rassine(min_radius, max_radius, 12.4, tag='%.0f'%(np.random.randint(1,10000)))
    return spec

def yarara_flux_density(dir_root,files,sub_dico='matching_diff',smooth=7):
    all_flux_density = []
    count = -1
    warning = 0
    plt.figure('flux_density',figsize=(7,7))

    grid = files[0]/100.
    for j in files[1]:
        count+=1
        #spec = import_spectrum(j,sub_dico=sub_dico)
        spec = myc.tableXY(grid,j.astype('float32')/10000.,0*grid)
        if smooth!=1:
            spec.smooth(box_pts=smooth,shape='savgol')
        mask = (spec.x<6250)&(spec.x>4000)
        flux_norm = spec.y[mask]
        flux_norm = flux_norm[flux_norm>0.01]
        used = np.round(len(flux_norm)*100/225000,1)
        if used<95:
            warning = 1
            plt.figure('warning')
            print(Fore.YELLOW+'\n [WARNING] Only %.1f%% of spectra used. Holes detected! Results may be inaccurate.'%(used)+Fore.RESET)
            plt.plot(spec.x,spec.y+count)
            plt.figure('flux_density')
        ha,hb = np.histogram(flux_norm,bins=100,density=True)
        hb = 0.5*(hb[1:]+hb[:-1])
        ha = np.nancumsum(ha)
        ha /= np.nanmax(ha)
        metric = hb[myf.find_nearest(ha,np.array([0.05,0.10,0.15,0.20,0.25]))[0]]
        plt.plot(hb,ha,color='C0',alpha=0.7)
        plt.scatter(metric,np.array([0.05,0.10,0.15,0.20,0.25]),marker='.',color='k',alpha=0.4)
        if used<50:
            metric = metric*np.nan
        all_flux_density.append(metric)

    all_flux_density = np.array(all_flux_density)
    all_flux_density = np.nanmedian(all_flux_density,axis=0)

    myv.vprint('\n [INFO] Flux density 5, 10, 15, 20, 25 : ',np.round(all_flux_density,3))

    xgb_file = MATERIAL_DIR+'/xgb_model_yarara_atmos_FluxD'+myv.SKLEARN_VERSION+'.p'
    xgb_obj = pickle.load(open(xgb_file,'rb'))
    model = xgb_obj['model']

    if np.sum(all_flux_density==all_flux_density)==len(all_flux_density):
        output = model.predict(all_flux_density[:,np.newaxis].T)
        Teff_rough_est = int(np.round(output[0,0],0)) # not better than +/- 300K
        FeH_rough_est = np.round(output[0,1],3) # not better than +/- 0.15 dex
    else:
        Teff_rough_est = 5778
        FeH_rough_est = 0.0

    plt.scatter(all_flux_density,np.array([0.05,0.10,0.15,0.20,0.25]),zorder=10,color='k',alpha=1.0,label='Teff=%.0f +/- 300 K \n FeH = %.2f +/- 0.15 dex'%(Teff_rough_est,FeH_rough_est))
    plt.legend(loc=2)
    plt.xlabel('Flux normalised')
    plt.ylabel('CDF')
    plt.grid()
    plt.xlim(0,1)
    plt.ylim(0,1)

    myv.vprint(' [INFO] Rough Teff estimation %.0f +/- 300 K'%(Teff_rough_est))
    myv.vprint(' [INFO] Rough FeH estimation %.2f +/- ?? dex'%(FeH_rough_est))

    if Teff_rough_est>7000:
        print(Fore.YELLOW+'\n [WARNING] Very hot star! Parameters unreliable'+Fore.RESET)

    plt.savefig(dir_root+'IMAGES/Teff_approximated'+myv.PRD_EXT+'.png')
    if warning:
        plt.figure('warning')
        plt.savefig(dir_root+'WARNING/WARNING_Flux_density'+myv.PRD_EXT+'.png')
        plt.close()

    del model
    del xgb_obj
    del flux_norm
    del spec
    del mask
    del grid
    del files

    return (Teff_rough_est,FeH_rough_est,np.round(all_flux_density,3),warning)

def yarara_rough_rv_sys(spec,teff=6000, verbose=False):
    
    if verbose:
        print(' [INFO] Rough RV_sys estimation...')

    wave = spec.x
    flux = spec.y

    if np.sum(flux!=0)!=0:
        wave_min = np.nanmin(wave[flux!=0])
        wave_max = np.nanmax(wave[flux!=0])
    else:
        wave_min = 10000
        wave_max = 0

    if (teff>6500):
        if verbose:
            print(' [INFO] Selected line set Teff>6500')        
        lines = np.array([myv.Heps[0],myv.Hd[0],myv.Hc[0],myv.Hb[0],myv.Ha[0]])
        box_pts = 50
    else:
        if (wave_max>5100):
            if verbose:
                print(' [INFO] Selected line set Teff<6500')
            lines = np.array([myv.NaDl[0],myv.NaDr[0],myv.Mg1b[0],myv.Mg1c[0],myv.Ha[0]])
            box_pts = 7
        else:
            if verbose:
                print(' [INFO] Selected line set CaII H&K')
            lines = np.array([myv.Ca2K[0],myv.Ca2H[0]])
            box_pts = 50

    lines = lines[lines<wave_max]
    lines = lines[lines>wave_min]

    if len(lines)==0:
        RV = np.nan
        RV_sys = np.nan
        print(Fore.YELLOW+' [WARNING] No lines available to compute the RV sys'+Fore.RESET)
    else:
        right,left = myf.doppler_r(np.array(lines),250*1000) # 200 km/s search

        RV = []
        for r,l,c in zip(right,left,lines):
            mask_wave = (wave>l)&(wave<r)
            if np.sum(mask_wave):
                flux2 = flux[mask_wave]
                rvs = []
                s = myc.tableXY(wave[mask_wave],flux2,0*flux2)
                s.smooth(box_pts=box_pts,shape='rectangular')
                s.find_min()
                maxi = np.max(s.y)
                mini = s.x_min[np.argmin(s.y_min)]
                rv = (mini-c)/c*myv.c_lum/1000
                if maxi>2:
                    rv = np.nan
                rvs.append(rv)
                RV.append(rvs)
        RV = np.array(RV)
        RV = np.nanmedian(RV,axis=1)
        RV_sys = np.round(np.nanmedian(RV),2)

    if verbose:
        print(' [INFO] Measured values :',np.round(RV,2))
        print(' [INFO] Rough RV_sys estimation = %.2f km/s'%(RV_sys))

    return RV_sys    

def yarara_check_rv_sys(spec, fwhm, rv_sys_approx, ccf_tag, dir_root=None):
    #UPDATE 12.12.2023 producing the plot even if condition satisfied

    mask_ccf = 'Magicat'

    myv.vprint(' [INFO] Selected CCF mask : %s'%(mask_ccf))
    mask = np.genfromtxt(MATERIAL_DIR+'/MASK_CCF/%s.txt'%(mask_ccf))
    mask = np.array([0.5*(mask[:,0]+mask[:,1]),mask[:,2]]).T

    rv_range = [15,fwhm][int(fwhm>15)]

    rv_sys_fit = rv_sys_approx*1000 # Update 28.08.24
    rv_sys_est1 = rv_sys_fit/1000

    spec.ccf(mask, weighted=True, rv_range=rv_range*1.5, rv_sys=rv_sys_fit, static=dir_root+'CCF_MASK/CCF_%s.fits'%(mask_ccf))

    rv_sys_fit += spec.ccf_params['cen'].value
    
    rv_sys_fit = np.round(rv_sys_fit/1000,2)

    fwhm = np.min([100,spec.ccf_params['wid'].value/1000*2.355])
    fwhm = np.round(fwhm,2)
    myv.vprint('\n [INFO] FWHM value fitted as %.2f kms'%(fwhm))

    rv_sys_est2 = rv_sys_fit

    mask_harps = 'G2'
    warning = 0
    if (abs(rv_sys_est1-rv_sys_est2)/abs(rv_sys_est1)*100)>20:
        myv.vprint('\n [WARNING] The two RV sys estimations (%.1f km/s, %.1f km/s) are very different!'%(rv_sys_est1,rv_sys_est2))
        if abs(rv_sys_est1)<300:
            myv.vprint(' [INFO] Second attempt to fit a CCF with standard HARPS DRS mask')
            mask = np.genfromtxt(MATERIAL_DIR+'/MASK_CCF/%s.txt'%(mask_harps))
            mask = np.array([0.5*(mask[:,0]+mask[:,1]),mask[:,2]]).T 
            spec.ccf(mask, weighted=True, rv_range=rv_range*1.5, rv_sys=rv_sys_est1*1000, static=dir_root+'CCF_MASK/CCF_'+mask_harps+'.fits')
            rv_sys_fit = rv_sys_est1*1000 + spec.ccf_params['cen'].value
            rv_sys_fit = np.round(rv_sys_fit/1000,2)
            rv_sys_est3 = rv_sys_fit
            fwhm = spec.ccf_params['wid'].value/1000*2.355
            myv.vprint('\n [INFO] %.1f km/s | %.1f km/s | %.1f km/s for FWHM = %.1f'%(rv_sys_est1,rv_sys_est2,rv_sys_est3,fwhm))
            if (abs(rv_sys_est1-rv_sys_est3)/fwhm*100)>20:
                if (abs(rv_sys_est2-rv_sys_est3)/fwhm*100)<20:
                    rv_sys_fit = rv_sys_est3
                else:
                    rv_sys_fit = rv_sys_est1
                    warning = 1

    if abs(rv_sys_fit)>300:
        print('\n [WARNING] RV sys (%.1f km/s) larger than 300 km/s! Set to default value 0.0 km/s'%(rv_sys_fit))
        rv_sys_fit = 0.0
        warning = 1

    if warning:
        fwhm = 15 # generic value 

    if warning:
        print('\n [WARNING] FWHM value fixed to %.2f kms'%(fwhm))
    else:
        fwhm = spec.ccf_params['wid'].value/1000*2.355

    fwhm = np.round(fwhm,2)
    ccf_beta = np.round(spec.params_beta,2)
    contrast_fit = 100*(abs(spec.ccf_params['amp'].value))
    rv_sys_fit = np.round(rv_sys_fit,2)

    y_max = 1.1
    y_min = 1-2*contrast_fit/100
    if y_min<0:
        y_min=0
    if contrast_fit<2:
        y_max = 1.01
    if (y_min==y_min)&(y_max==y_max):
        plt.ylim(y_min,y_max)

    myv.vprint('\n [INFO] RV_sys value fitted as %.2f kms'%(rv_sys_fit))
    
    if dir_root is not None:
        plt.savefig(dir_root+'IMAGES/RV_sys_fitting'+myv.PRD_EXT+'.png')
    
    SB1 = 0
    if spec.warning_multipeak==1:
        SB1 = 1
        if dir_root is not None:
            plt.savefig(dir_root+'WARNING/WARNING_RV_sys_fitting'+myv.PRD_EXT+'.png')

    ccf = pd.DataFrame(np.array([spec.ccf_profile.x/1000,spec.ccf_profile.y]).T,columns=['vrad','ccf'])
    contrast = np.round(contrast_fit/100,3)
    
    output = (fwhm,rv_sys_fit,contrast,ccf_beta,SB1,np.round(spec.ccf_Rcorr,2),ccf)
    if dir_root is not None:
        ccf.to_csv(dir_root+'STAR_INFO/CCF_RV_SYS.csv')
    return output

def yarara_check_sb(dir_root):

    if myv.VERBOSE:
        myf.print_box('\n---- RECIPE : CHECK SB ---- \n')

    ccf = pd.read_csv(dir_root+'STAR_INFO/CCF_RV_SYS.csv',index_col=0)
    ccf = myc.tableXY(np.array(ccf['vrad']),np.array(ccf['ccf']),0*np.array(ccf['ccf']))
    criterion = ccf.fit_multi_sb()
    plt.savefig(dir_root+'IMAGES/SB_check'+myv.PRD_EXT+'.png')
    if criterion:
        plt.savefig(dir_root+'WARNING/WARNING_SB'+myv.PRD_EXT+'.png')
    return criterion


def yarara_check_rv_sys_wrapper(dir_root, spec, rv_sys_approx, ccf_tag=0):
    
    if myv.VERBOSE:
        myf.print_box('\n---- RECIPE : RV_SYS EXTRACTION ---- \n')
    
    if ccf_tag!=0:
        os.system('rm -f '+dir_root+'CCF_MASK/*.fits')

    spec.clip(min=[4000,None])
    spec.y[spec.y>1.50] = 1.0
    save = []
    for fwhm in [6,10,20,40,60,100,200][::-1]:
        sinfo = yarara_check_rv_sys(spec, fwhm, rv_sys_approx, ccf_tag, dir_root=dir_root)
        if abs(sinfo[0])>500:
            save.append([fwhm,-999,-999,-999,-999,rv_sys_approx,0])
        else:
            save.append([fwhm,sinfo[0],sinfo[2],sinfo[1],sinfo[5],rv_sys_approx,sinfo[4]])
    save = np.array(save)
    plt.close('all')
    rvsys_backup = save[:,3].copy()
    myv.vprint(' [INFO] Table summary FWHM | RV_SYS \n')
    save[save[:,3]==save[:,5],3] = -999
    save[save[:,4]<0.70,3] = -998
        
    validated = (save[:,3]>-900)
    if (np.sum(validated)==0)&(np.sum(save[:,3]!=-999)!=0):
        index = np.arange(len(save))[save[:,3]!=-999]
        selected = index[np.argsort(save[index,4])[-1]]
        validated[selected] = True
        rvsys_backup[~validated] = -999
        save[:,3] = rvsys_backup
    
    if np.sum(save[:,3]!=-999)==0:
        validated = np.ones(len(validated)).astype('bool')
        save[save[:,4]>0.75,3] = save[save[:,4]>0.75,5]
    
    kept = save[validated]

    summary = pd.DataFrame(save,columns=['RVRANGE','FWHM','CT','RV','RCORR','RV_APPROX','SB1'])
    summary['!'] = ''
    if np.max(kept[:,2])>0.05: 
        fwhm1 = kept[np.argmin(abs(kept[:,3]-kept[:,5])),1]
        loc = np.argmin(abs(kept[:,0]-fwhm1))
    else: #if CT < 1% likely fast rotating stars and RV not reliable
        loc = np.argmax(kept[:,4])
    fwhm = kept[loc,1]
    rv_sys = kept[loc,3]
    loc = np.arange(len(summary))[validated][loc]
    summary.loc[loc,'!'] = '<--'
    if myv.VERBOSE:
        print(summary)
        print('\n [INFO] Best FWHM detected is %.2f km/s'%(fwhm))
        print('\n [INFO] Best RV_SYS detected is %.1f km/s \n'%(rv_sys))
    SB1 = int(np.sum(kept[:,-1])!=0)
    
    os.system('rm -f '+dir_root+'CCF_MASK/*.fits')
    sinfo = yarara_check_rv_sys(spec, fwhm, rv_sys, ccf_tag, dir_root=dir_root)

    if fwhm<50:
        pass#yarara_check_fwhm()

    return sinfo,SB1


def replace_none(y,yerr):
    if yerr is None:
        return np.nan, 1e6
    else:
        return y,yerr

def yarara_ccf(dir_root, files, rv_sys, fwhm, beta_gnd, mask, spectra=None, ccf_tag=0,
                mask_col='weight_rv', analytical_model='auto', sub_dico='matching_diff', rv_mode='RV',
                weighted=True, debug=False, normalisation='left', return_ccf=False, save=True,
                del_outside_max = False, ccf_oversampling=1, check_non_transform=True, continuum_method='flux',
                rv_range=None, rv_borders=None, bis_range=None, delta_window=5, rv_shift=None,
                wave_min=4000, wave_max=10000, squared=True):
    """ 
    Compute the CCF of a spectrum, reference to use always the same continuum (matching_anchors highest SNR). 
    Display_ccf to plot all the individual CCF. Plot to plot the FWHM, contrast and RV.
    
    Parameters
    ----------
    mask : The line mask used to cross correlate with the spectrum (mask should be located in MASK_CCF otherwise KITCAT dico)
    mask_col : Column of the KitCat column to use for the weight
    display_ccf : display all the ccf subproduct
    save : True/False to save the informations iun summary table
    normalisation : 'left' or 'slope'. if left normalise the CCF by the most left value, otherwise fit a line between the two highest point 
    del_outside maximam : True/False to delete the CCF outside the two bump in personal mask 
    """

    start = time.time()

    ins = dir_root.split('/')[-2]
    jdb = get_jdb(files[-1],dir_root)

    myv.vprint(' [INFO] RV sys : %.2f [km/s] '%(rv_sys))
    rv_sys = 1000*rv_sys

    myv.vprint('\n [INFO] FWHM: %.2f kms'%(fwhm))
    if rv_range is None:
        rv_range=int(3*fwhm)
        myv.vprint(' [INFO] RV range updated to : %.1f kms'%(rv_range))
    
    if rv_borders is None:
        rv_borders=int(2*fwhm)
        myv.vprint(' [INFO] RV borders updated to : %.1f kms'%(rv_borders))
    
    if bis_range is None:
        bis_range=np.round(0.33*fwhm,1)
        myv.vprint(' [INFO] BISSPAN borders updated to : %.1f kms'%(bis_range))

    if analytical_model=='auto':
        analytical_model = 'gaussian'
        if beta_gnd>2.5:
            analytical_model = 'GND%.1f'%(beta_gnd)
    myv.vprint(' [INFO] CCF analytical model :',analytical_model)
    
    if type(mask)==str:
        ccf_name = mask
        mask_name = mask
        mask_loc = MATERIAL_DIR+'/MASK_CCF/'+mask+'.txt'
        mask = np.genfromtxt(mask_loc)
        mask = np.array([0.5*(mask[:,0]+mask[:,1]),mask[:,2]]).T
        myv.vprint('\n [INFO] CCF mask selected : %s'%(mask_loc))
    elif type(mask)==pd.core.frame.DataFrame:
        mask = np.array([np.array(mask['freq_mask0']).astype('float'),np.array(mask[mask_col]).astype('float')]).T
        mask_name = 'ManualDF'

    shift_rv = np.zeros(len(files[-1]))
    if type(rv_shift)==np.ndarray:
        shift_rv = rv_shift
    
    mask[:,0] = myf.doppler_r(mask[:,0],rv_sys)[0]
            
    if spectra is None:
        grid, flux, flux_err = import_sts(files, rv_shift=shift_rv, err=False, sub_dico=sub_dico)
    else:
        grid, flux, flux_err = spectra

    flux_err = None

    if rv_mode=='EPRV':
        myv.vprint('\n [INFO] Reference color : ESPRESSO Procyon (Red weighting)')
        color = pd.read_csv(MATERIAL_DIR+'/EPRV_color.csv',index_col=0)
        color = myc.tableXY(color.wave.astype('float')/100,color.continuum.astype('float')/10000)
        color.interpolate(new_grid=grid, fill_value=0, method='linear')
        flux = flux * color.y
    else:
        myv.vprint('\n [INFO] Reference color : flat normalised continuum')
    
    mask_shifted = myf.doppler_r(mask[:,0],(rv_range+5)*1000)
    
    mask = mask[(myf.doppler_r(mask[:,0],30000)[0]<grid.max())&(myf.doppler_r(mask[:,0],30000)[1]>grid.min()),:] #supres line farther than 30kms
    mask = mask[mask[:,0]>wave_min,:] 
    mask = mask[mask[:,0]<wave_max,:] 
    
    mask_min = np.min(mask[:,0])
    mask_max = np.max(mask[:,0])

    myv.vprint('\n [INFO] Nb lines in the mask : %.0f'%(len(mask)))
    myv.vprint(' [INFO] Wave min : %.0f AA | Wave max : %.0f AA'%(mask_min,mask_max))

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
    #print(' [INFO] Percentage of the spectrum used : %.1f [%%] (%.0f)'%(100*sum(used_region)/len(grid),len(grid)))
    time.sleep(1)

    if not os.path.exists(dir_root+'CCF_MASK/CCF_'+mask_name.split('.')[0]+'.fits'):
        myv.vprint('\n [INFO] CCF mask reduced for the first time, wait for the static mask production... \n')
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
        hdul.writeto(dir_root+'CCF_MASK/CCF_'+mask_name.split('.')[0]+'.fits',overwrite=True)
        myv.vprint('\n [INFO] CCF mask saved under : %s'%(dir_root+'CCF_MASK/CCF_'+mask_name.split('.')[0]+'.fits'))

        del hdu
        del hdul
    else:
        myv.vprint('\n [INFO] CCF mask found : %s'%(dir_root+'CCF_MASK/CCF_'+mask_name.split('.')[0]+'.fits'))
        log_grid_mask, log_mask = fits.open(dir_root+'CCF_MASK/CCF_'+mask_name.split('.')[0]+'.fits')[0].data.T
    
    #log_mask = log_mask**(1.0+float(squared))
    #log_template = myc.tableXY(log_grid_mask,log_mask,0*log_mask)
    #log_template.interpolate(new_grid=log_grid,method='linear',replace=True,fill_value=0)
    #log_template = log_template.y
    log_template = myf.interpolate_rv_shift(log_grid_mask,log_mask**(1.0+float(squared)),xnew=log_grid,fill_value=0)

    amplitudes = [] ; amplitudes_std = []
    rvs = [] ; rvs_std = []
    fwhms = [] ; fwhms_std = []
    ew = [] ; ew_std = []
    centers = [] ; centers_std = []
    depths = [] ; depths_std = []
    bisspan = []  ; bisspan_std = []
    
    now = datetime.datetime.now()
    myv.vprint('\n Computing CCFs (Current time %.0fh%.0fm%.0fs) \n'%(now.hour, now.minute, now.second))
    
    chunks = np.array_split(np.arange(len(log_grid)), 5)

    if True:
        grid_log10 = np.log10(grid)
        for idx in chunks:
            idx2 = (grid_log10>log_grid[idx[0]])&(grid_log10<log_grid[idx[-1]])
            for j,i in enumerate(files[-1]):
                flux[j][idx] = myf.interpolate_rv_shift(grid_log10[idx2], flux[j][idx2], xnew=log_grid[idx], rv=0, fill_value=0, kind=interp_degree)
        del grid_log10
    else:
        print('ALGO1')
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
    
    myv.vprint('\n [INFO] Gravity center wavelength = %.0f AA \n'%(gravity_center_wave))
    #flux = flux[:,used_region]
    #log_grid = log_grid[used_region]
    #log_template = log_template[used_region]
    #if flux_err is not None:
    #    flux_err = flux_err[:,used_region]

    start3 = time.time()
    vrad, ccf_power, ccf_power_std = myf.ccf(log_grid, flux, log_template, 
                                                rv_range = rv_range, oversampling = ccf_oversampling, spec1_std = flux_err) #to compute on all the ccf simultaneously

    if fwhm>100:
        for n,c in enumerate(ccf_power.T):
            ccf_power[:,n] = myf.smooth(c,box_pts=30)

    mask_vmax = abs(vrad)>(200*1000)
    if np.sum(mask_vmax):
        for n,c in enumerate(ccf_power.T):
            poly_coeff = np.polyfit(vrad[mask_vmax],c[mask_vmax],3)
            model = np.polyval(poly_coeff,vrad)
            ccf_power[:,n] = 1 + c - model
            #plt.plot(vrad,model,color='orange')

    del log_grid
    del log_mask
    del log_template

    end = time.time()
    if myv.DEV:
        counter_dev+=1
        print(f"Line number: {inspect.currentframe().f_lineno}")
        print(Fore.YELLOW+f"Execution time {counter_dev}: {end - start:.3f} seconds"+Fore.RESET)

    del flux
    del flux_err

    now = datetime.datetime.now()
    dv = np.median(np.diff(vrad))
    myv.vprint('')
    myv.vprint('\n CCFs computed (Current time %.0fh%.0fm%.0fs)'%(now.hour, now.minute, now.second))
    myv.vprint('\n [INFO] CCF velocity step : %.0f m/s'%(dv))

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
    master_ccf.clip(min=[-rv_borders,None],max=[rv_borders,None])
    try:
        master_ccf.fit_GND(beta_fixed=0,Plot=False)
        beta0 = master_ccf.params['beta'].value
    except:
        beta0 = 2.0

    myv.vprint(' [INFO] Beta value of GND = %.2f'%(beta0))
    if (beta0>2.5)&(analytical_model=='gaussian'):
        myv.vprint(Fore.YELLOW+' \n [WARNING] Significant Kurtosis detected.'+Fore.RESET)
    
    dccf2 = (ccf_power-ccf_ref[:,np.newaxis])[top_ccf]/np.mean(ccf_power[continuum_ccf])*100
    dccf2 -= np.median(dccf2,axis=0)
    ccf_snr = 1/np.std(dccf2,axis=0)*100
    myv.vprint(' [INFO] SNR CCF continuum median : %.0f'%(np.median(ccf_snr)))

    noise_ccf = (np.sqrt(ccf_ref/np.max(ccf_ref))*ccf_ref[continuum_ccf])[:,np.newaxis]/ccf_snr #assume that the noise in the continuum is white (okay for matching_mad but wrong when tellurics are still there)
    sigma_rv = noise_ccf/(abs(np.gradient(ccf_ref))/np.gradient(vrad))[:,np.newaxis]
    w_rv = (1/sigma_rv)**2
    svrad_phot = 1/np.sqrt(np.sum(w_rv,axis=0))
    scaling = np.sqrt(820/np.mean(np.gradient(vrad))) #to penalize oversampling in vrad 
    svrad_phot*=scaling
    
    svrad_phot[svrad_phot==0] = 2*np.max(svrad_phot) #in case of null values
    
    myv.vprint(' [INFO] Photon noise RV median : %.2f m/s\n '%(np.median(svrad_phot)))        
    
    svrad_phot2 = {}
    svrad_phot2['rv'] = 10**(0.98*np.log10(svrad_phot)-3.08) # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2['contrast'] = 10**(0.98*np.log10(svrad_phot)-3.58) # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2['fwhm'] = 10**(0.98*np.log10(svrad_phot)-2.94) # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2['center'] = 10**(0.98*np.log10(svrad_phot)-2.83) # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2['depth'] = 10**(0.97*np.log10(svrad_phot)-3.62) # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2['ew'] = 10**(0.97*np.log10(svrad_phot)-3.47) # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2['vspan'] = 10**(0.98*np.log10(svrad_phot)-2.95) # from photon noise simulations Photon_noise_CCF.py
        
    myv.vprint(' [INFO] Photon noise RV from calibration : %.2f m/s '%(np.median(svrad_phot2['rv'])*1000))

    myv.vprint(' [INFO] Number of velocity bin = %.0f'%(len(vrad)))

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
        print(f"Line number: {inspect.currentframe().f_lineno}")
        print(Fore.YELLOW+f"Execution time {counter_dev}: {end - start:.3f} seconds"+Fore.RESET)

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
            if fwhm<100:
                center=ccf.x[ccf.y.argmin()]
            else:
                center = 0.0
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
                print(' \n [WARNING] Discrepancy detected between CCFs (%.4f/%.4f), value reset to non-transformed one'%(V1,V2))
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
            print(' [INFO] BISSPAN updated to +/- 0.5 km/s')
        if len(ccf.clipped.x)<5:
            ccf.clip(min=[-2,None],max=[2,None],replace=False) 
            print(' [INFO] BISSPAN updated to +/- 2 km/s')
        if len(ccf.clipped.x)<5:
            ccf.clip(min=[-5,None],max=[5,None],replace=False)    
            print(' [INFO] BISSPAN updated to +/- 5 km/s')

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
        print(f"Line number: {inspect.currentframe().f_lineno}")
        print(Fore.YELLOW+f"Execution time {counter_dev}: {end - start:.3f} seconds"+Fore.RESET)

    rvs_std = svrad_phot2['rv']
    fwhms = np.array(fwhms).astype('float')*2.355
    fwhms_std = np.array(fwhms_std).astype('float')*2.355
    
    warning_rv_borders = False
    if np.median(fwhms)>(rv_borders/1.5):
        print(Fore.YELLOW+' [WARNING] The CCF is larger than the RV borders for the fit'+Fore.RESET)
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

    ccf_infos = {
        'table':ccf_infos,
        'model_parametric':model_parametric,
        'rv_sys':rv_sys/1000,
        'rv_mode':rv_mode,
        'weighting':1.0+float(squared),
        'creation_date':datetime.datetime.now().isoformat()}
    
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
            myv.vprint(' [INFO] Reference value for %s is %.1f km/s'%(ins,ref))
            if abs(ref - fwhm_ins)>1:
                warning = 1
                print(Fore.YELLOW + '\n [WARNING] Instrumental resolution is not usual (%.1f km/s)'%(fwhm_ins)+Fore.RESET)
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

def import_summary(dir_root):
    material = pd.read_csv(dir_root+'WORKSPACE/Analyse_summary.csv',index_col=0)
    return material

def get_jdb(files,dir_root):
    try:
        summary = import_summary(dir_root)
        mask = myf.in1d(np.array(summary['filename']),files)
        jdb = np.array(summary.loc[mask,'jdb'])
        if np.sum(jdb!=jdb)!=0:
            jdb = np.arange(len(files))
    except:
        jdb = np.arange(len(files))
    return jdb

def import_material(dir_root):
    material = pd.read_pickle(dir_root+'WORKSPACE/Analyse_material.p')
    return material

def import_spectroscopy(dir_root):
    spectroscopy = pd.read_pickle(dir_root+'WORKSPACE/Analyse_spectroscopy.p')
    return spectroscopy

def import_master(dir_root):
    material = import_material(dir_root)
    master = myc.tableXY(material['wave'],material['reference_spectrum'],0*material['wave'])
    return master

def import_stellar_template(teff,feh=0.0,logg=4.5,model='ATLAS',rv_sys=0.0):
    if model=='ATLAS':
        table = np.load(MATERIAL_DIR+'/template_star_ATLAS_3800_7000.npy')
        table_columns = pd.read_csv(MATERIAL_DIR+'/template_star_ATLAS_3800_7000.csv',index_col=0).columns[1:]
        code = [[i,i.split('T')[1].split('_')[0],i.split('g')[1].split('_')[0],i.split('H')[-1].split('_')[0]] for i in table_columns]
        code = np.array(code)
        code[:,-1] = 0.0
        wave = np.unique(np.round(np.arange(3800,7000,0.01),2))
    else:
        table = np.load(MATERIAL_DIR+'/template_star_SNAKY_3900_6800.npy')
        table_columns = pd.read_csv(MATERIAL_DIR+'/template_star_SNAKY_3900_6800.csv',index_col=0).columns[1:]
        code = [[i,i.split('T')[1].split('_')[0],i.split('g')[-1].split('_')[0],i.split('H')[-1].split('_')[0]] for i in table_columns]
        code = np.array(code)
        code[:,2] = 4.5
        wave = np.unique(np.round(np.arange(3900,6800,0.01),2))
    
    value = myf.find_nearest(code[:,1].astype('float'),teff)[1]
    code = code[code[:,1].astype('float')==value]
    value = myf.find_nearest(code[:,2].astype('float'),logg)[1]
    code = code[code[:,2].astype('float')==value]
    value = myf.find_nearest(code[:,3].astype('float'),feh)[1]
    code = code[code[:,3].astype('float')==value]
    final_model = code[0][0]

    loc = np.where(table_columns==final_model)[0][0]
    template = (table[:,loc]/10000).astype('float')
    template = myc.tableXY(wave,template,0*wave)

    template.rv_shift(rv=rv_sys)

    return template

def import_dace_table(dir_root):
    dace = pd.read_csv(dir_root+'DACE_TABLE/Dace_extracted_table.csv',index_col=0)
    return dace

def import_star_info(dir_root):
    star = dir_root.split('/data')[0].split('/')[-1]
    sinfo = pd.read_pickle(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star))
    return sinfo

def import_ccf_profile(dir_root,mask_name):
    ccf_profile = pd.read_pickle(dir_root+'WORKSPACE/Analyse_ccf_saved.p')['CCF_%s'%(mask_name)]
    return ccf_profile

def import_ccf(dir_root,mask_name):

    ccf = pd.read_pickle(dir_root+'WORKSPACE/Analyse_ccf.p')['CCF_%s'%(mask_name)]
    ccf_infos = ccf['table']
    rvs = ccf_infos['rv']
    rvs_std = ccf_infos['rv_std']

    amplitudes = ccf_infos['contrast']
    amplitudes_std = ccf_infos['contrast_std']

    fwhms = ccf_infos['fwhm']
    fwhms_std = ccf_infos['fwhm_std']

    bisspan = ccf_infos['bisspan']
    bisspan_std = ccf_infos['bisspan_std']

    jdb = ccf_infos['jdb']

    ccf_rv = myc.tableXY(jdb,np.array(rvs)*1000,np.array(rvs_std)*1000)
    ccf_contrast = myc.tableXY(jdb,np.array(amplitudes)*100,np.array(amplitudes_std)*100)
    ccf_fwhm = myc.tableXY(jdb,fwhms,fwhms_std)
    ccf_vspan = myc.tableXY(jdb,np.array(bisspan)*1000,np.array(bisspan_std)*1000)

    output = {
        'table':ccf_infos,
        'filename':np.array(ccf_infos['filename']),
        'rv':ccf_rv,
        'contrast':ccf_contrast,
        'fwhm':ccf_fwhm,
        'vspan':ccf_vspan,
        }

    return output

def yarara_iron_lines(dir_root, master, fwhm, rv_sys=0.0):

    if myv.VERBOSE:
        myf.print_box('\n---- RECIPE : EW OF LINE SPECIES ----\n')

    myv.vprint('\n [INFO] Metallicity abundances recipe launched')
    myv.vprint(' [INFO] FWHM = %.2f km/s'%(fwhm))

    rv_range = 3*fwhm
    sigma_3wid = rv_range/2.3556
    grid = np.arange(0,sigma_3wid*1000,100) 
    grid = np.hstack([-grid[1:][::-1],grid])
    
    grid_out = np.arange(0,rv_range*1000,100) 
    grid_out = np.hstack([-grid_out[1:][::-1],grid_out])

    Contrast = {}
    EW = {}
    if (np.min(master.x)<6800)&(np.max(master.x)>5150):
        ew0 = 0
        w1 = myf.doppler_r(6341.16,rv_sys*1000)[0] ; w2 = myf.doppler_r(6346.39,rv_sys*1000)[0]
        w1 = myf.find_nearest(master.x,w1)[0][0] ; w2 = myf.find_nearest(master.x,w2)[0][0]
        ew0 += np.mean(1-master.y[w1:w2])*(w2-w1)*np.diff(master.x)[0]
        w1 = myf.doppler_r(6359.00,rv_sys*1000)[0] ; w2 = myf.doppler_r(6364.14,rv_sys*1000)[0]
        w1 = myf.find_nearest(master.x,w1)[0][0] ; w2 = myf.find_nearest(master.x,w2)[0][0]
        ew0 += np.mean(1-master.y[w1:w2])*(w2-w1)*np.diff(master.x)[0]
        ew0 *= 1000
        contrast0 = ew0/(1000*np.diff(master.x)[0]*2*(w2-w1))
        
        Contrast['Z']=np.round(contrast0,5)
        EW['Z'] = np.round(ew0,1)
        myv.vprint('\n [INFO] Contrast Z = %.3f | EW = %.1f mA'%(contrast0*100,ew0))
        
        mask1 = np.genfromtxt(MATERIAL_DIR+'/MASK_CCF/FeIU.txt')
        mask11 = np.array([0.5*(mask1[:,0]+mask1[:,1]),mask1[:,2]]).T
        master.ccf(mask11, weighted=False, rv_range=rv_range,rv_sys=rv_sys*1000,fit_gaussian=False, save_if_missing=False)

        master.ccf_profile.smooth(box_pts=10,replace=False,shape='rectangular')
        master.ccf_profile.y /= np.max(master.ccf_profile.smoothed.y)
        master.ccf_profile.interpolate(new_grid=grid_out,replace=False)
        quality_control = master.ccf_profile.y_interp
        master.ccf_profile.interpolate(new_grid=grid,replace=False)

        contrast1 = 1-np.mean(master.ccf_profile.y_interp)
        contrast1 = np.sum(1-master.ccf_profile.y_interp)*np.diff(grid)[0]/1000
        ew1 = contrast1/3e5*np.mean(mask1[:,0])*1000

        plt.fill_between(grid,master.ccf_profile.y_interp,1,color='g',alpha=0.2,label='%.2f'%(contrast1))
        plt.legend(loc=3)
        plt.savefig(dir_root+'IMAGES/Atmos_FeIU'+myv.PRD_EXT+'.png')
        
        myv.vprint('\n [INFO] Contrast FeIU = %.3f kms | EW = %.1f mA'%(contrast1,ew1))

        Contrast['FeIU']=np.round(contrast1,5)
        EW['FeIU'] = np.round(ew1,1)

        plt.figure('All_profiles',figsize=(12,12))
        plt.subplot(4,4,1)
        plt.scatter(master.ccf_profile.x/1000,master.ccf_profile.y,color='k',s=5)
        plt.fill_between(grid/1000,master.ccf_profile.y_interp,1,color='g',alpha=0.2,label='%.2f'%(contrast1))
        plt.axvline(x=0,color='k',ls=':',lw=1)
        plt.legend(loc=3)
        plt.title('FeIU')
        plt.ylim(0.0,1.1)
        plt.xlim(-rv_range,rv_range)
        count=1

        QC = [0,0]

        for species in ['FeIS','FeIIS','TiI','VI','MnI','NdII','TiII','CrI','NiI','CoI','CaI','SiI','ScII','CaH','LiI']:
            count+=1
            mask2 = np.genfromtxt(MATERIAL_DIR+'/MASK_CCF/%s.txt'%(species))
            mask22 = np.array([0.5*(mask2[:,0]+mask2[:,1]),mask2[:,2]]).T
            non_zero = np.sum([master.y[myf.find_nearest(master.x,w1)[0][0]] for w1 in myf.doppler_r(mask22[:,0],rv_sys*1000)[1]])
            master.ccf(mask22, weighted=False, rv_range=rv_range,rv_sys=rv_sys*1000,fit_gaussian=False, save_if_missing=False)

            master.ccf_profile.smooth(box_pts=10,replace=False,shape='rectangular')
            max_value = np.max(master.ccf_profile.smoothed.y)
            master.ccf_profile.interpolate(new_grid=grid_out,replace=False)
            quality_control = master.ccf_profile.y_interp/max_value
            if non_zero!=0:
                master.ccf_profile.y /= max_value
                master.ccf_profile.interpolate(new_grid=grid,replace=False)
            else:
                print(Fore.YELLOW+'\n [WARNING] No lines detected for %s!'%(species)+Fore.RESET)
                master.ccf_profile = myc.tableXY(grid,0*grid)
                master.ccf_profile.y_interp = np.nan*np.ones(len(master.ccf_profile.x))

            contrast2 = 1-np.mean(master.ccf_profile.y_interp)
            contrast2 = np.sum(1-master.ccf_profile.y_interp)*np.diff(grid)[0]/1000
            ew2 = contrast2/3e5*np.mean(mask2[:,0])*1000
            qc = (np.sum(1-quality_control)*np.diff(grid)[0]/1000)/3e5*np.mean(mask2[:,0])*1000
            QC[0] += ew2
            QC[1] += (qc-ew2)
            #print(QC)

            plt.fill_between(grid,master.ccf_profile.y_interp,1,color='g',alpha=0.2,label='%.2f'%(contrast2))
            plt.legend(loc=3)
            plt.savefig(dir_root+'IMAGES/Atmos_%s'%(species)+myv.PRD_EXT+'.png')

            myv.vprint('\n [INFO] Contrast %s = %.3f kms | EW = %.1f mA'%(species,contrast2,ew2))
            
            Contrast[species]=np.round(contrast2,5)
            EW[species] = np.round(ew2,1)

            plt.figure('All_profiles')
            plt.subplot(4,4,count)
            plt.scatter(master.ccf_profile.x/1000,master.ccf_profile.y,color='k',s=5)
            plt.fill_between(grid/1000,master.ccf_profile.y_interp,1,color='g',alpha=0.2,label='%.2f'%(contrast2))
            plt.axvline(x=0,color='k',ls=':',lw=1)
            plt.legend(loc=3)
            plt.title(species)
            plt.ylim(0.0,1.1)
            plt.xlim(-rv_range,rv_range)
        
        SNR = QC[0]/QC[1]
        FLAG = 0
        myv.vprint('\n [INFO] SNR signal power = %.2f (Tot = %.0f)'%(SNR,QC[0]))
        if (SNR<0.8):
            print(Fore.YELLOW+' [WARNING] Power SNR < 1 (no lines)'+Fore.RESET)
            FLAG = 1

        if (SNR<1.5)&(QC[0]>5000):
            print(Fore.YELLOW+' [WARNING] The power is not distributed on the line profiles.'+Fore.RESET)
            FLAG = 2
        
        plt.subplots_adjust(hspace=0.35,wspace=0.35,top=0.95,left=0.07,right=0.96,bottom=0.13)
        plt.savefig(dir_root+'IMAGES/Atmos_all'+myv.PRD_EXT+'.png')
    return Contrast, EW, FLAG

def yarara_atmos_xgb_spectroscopy(dir_root, star_info, resolution=110000, phot=False, flag=False):
    
    if myv.VERBOSE:
        myf.print_box('\n---- RECIPE : XGB ATMOSPHERIC PARAMETERS ----\n')

    sinfo = import_star_info(dir_root)

    if phot:
        lines = ['FeIU','FeIS','FeIIS','TiI','VI','MnI','NdII','TiII','CaH','Z']
        xgb_file = MATERIAL_DIR+'/xgb_model_yarara_atmos_phot'+myv.SKLEARN_VERSION+'.p'
    else:
        lines = ['Ha','NaD','MgI','Hb','FeIU','FeIS','FeIIS','TiI','VI','MnI','NdII','TiII','CaH','Z']
        xgb_file = MATERIAL_DIR+'/xgb_model_yarara_atmos'+myv.SKLEARN_VERSION+'.p'
    ew = np.array([star_info['Contrast'][kw] for kw in lines])
    rv_sys = star_info['Rv_sys']['SNAKY']

    if flag:
        ew *= np.nan

    myv.vprint(' [INFO] EW:',np.round(np.hstack(ew.T),2))

    R_ratio = resolution/110000 # CORALIE and ESPRESSO similar on HD10700
    factor = 1 # Update 31.10.24 seems no more useful now that EW is used

    warning = 0
    if np.sum(ew)==np.sum(ew):
        xgb_obj = pickle.load(open(xgb_file,'rb'))
        model = xgb_obj['model']
        means = xgb_obj['mean']
        stds = xgb_obj['std']

        norm_mean = np.array(means[[l+' EW' for l in lines]])
        norm_std = np.array(stds[[l+' EW' for l in lines]])
        ew = (ew-norm_mean)/norm_std

        output = model.predict(ew[:,np.newaxis].T)
        output = pd.DataFrame(output,columns=['teff','feh','logg'])

        norm_mean = np.array(means[['teff','feh','logg']])
        norm_std = np.array(stds[['teff','feh','logg']])

        output = output*norm_std+norm_mean
        teff,feh,logg = output.values[0]
        teff_rough = sinfo['Teff']['FluxD']

        if (sinfo['Teff']['FluxD']>6500): #outside calibration range
            if abs(teff-teff_rough)>600:
                print(Fore.YELLOW + '\n [WARNING] Too much difference (dT=%.0fK) in Teff (%.0fK) with FluxD (%.0fK). XGB skipped.'%(abs(teff-teff_rough),teff,teff_rough)+Fore.RESET)
            else:
                print(Fore.YELLOW + '\n [WARNING] The temperature is outside the calibration range. XGB skipped.'+Fore.RESET)
            teff = int(teff_rough)
            feh = 0
            logg = 4.0
        elif sinfo['Teff']['FluxD']<3500: #outside calibration range
            print(Fore.YELLOW + '\n [WARNING] The temperature is outside the calibration range. XGB skipped.'+Fore.RESET)
            teff = int(teff_rough)
            feh = 0
            logg = 5.0
        else:
            teff = int(teff)
            feh = np.round(feh,3)
            logg = np.round(logg,3)
    else:
        print(Fore.YELLOW+'\n [WARNING] NaN detected indicated missing lines. Parameters computation skipped.'+Fore.RESET)
        teff,feh,logg,M,R,BV,vmicro,vmacro = np.nan*np.ones(8)   
        teff = sinfo['Teff']['FluxD']
        logg = 4.5
        feh = 0.0
        warning = 1

    dteff = [70,300][warning]
    dlogg = [0.07,0.25][warning]
    dfeh = [0.07,0.5][warning]

    params = create_atmos_sample(dir_root, teff, dteff, logg, dlogg, feh, dfeh, rv_sys)
    return params

def create_atmos_sample(dir_root, teff, dteff, logg, dlogg, feh, dfeh, rv_sys):
    M, dM, R, dR, samples_ms, samples_rs = myf.find_stellar_mass_radius_MS(teff, logg, samples=99999, dTeff=dteff, dlogg=dlogg) #new function
    M = np.round(M,2)
    R = np.round(R,2)

    BV = -3.684*np.log10(teff) + 14.551 #http://www.isthe.com/chongo/tech/astro/HR-temp-mass-table-byhrclass.html
    BV = BV - 0.04 # solar correction ZP for Sun = 0.65
    BV_std = np.std(-3.684*np.log10(np.random.randn(99999)*75+teff) + 14.551)

    vmicro, vmacro = myf.find_turbulence(teff, logg)
    BV = np.round(BV,3)
    vmicro = np.round(vmicro,2)
    vmacro = np.round(vmacro,2)

    if myv.VERBOSE:
        print(' [INFO] Effective temperature = %.0f +/- %.0f K'%(teff,dteff))
        print(' [INFO] Metallicity [Fe/H] = %.2f +/- %.2f dex'%(feh,dfeh))
        print(' [INFO] Log(g) = %.2f +/- %.2f dex'%(logg,dlogg))
        print(' [INFO] Ms = %.2f +/- %.2f Msol'%(M,dM))
        print(' [INFO] Rs = %.2f +/- %.2f Rsol'%(R,dR))
        print(' [INFO] BV = %.2f +/- 0.02'%(BV))
        print(' [INFO] Vmic = %.1f km/s '%(vmicro))
        print(' [INFO] Vmac = %.1f km/s '%(vmacro))

    samples_teff = np.random.randn(99999)*70+teff
    samples_feh = np.random.randn(99999)*0.07+feh
    samples_logg = np.random.randn(99999)*0.07+logg

    samples = pd.DataFrame(np.array([samples_ms, samples_rs, samples_teff, samples_logg, samples_feh]).T,columns=['ms','rs','teff','logg','feh'])
    samples.to_csv(dir_root+'WORKSPACE/Analyse_samples.csv.gz',index=False)

    if logg<3.5:
        logg=3.5
    elif logg>5.0:
        logg=5.0

    current = plt.gca()
    fig = plt.gcf()
    name = fig.get_label()
    if name=='All_profiles':
        xlim = current.get_xlim()[1]
        fwhm = xlim/3
        plt.axes([0,0,1,0.1])
        plt.axis('off')
        plt.text(0.5,0.5,'FWHM = %.2f km/s | RV_sys = %.1f km/s\n'%(fwhm,rv_sys)+r'$T_{eff}$'+' = %.0f +/- %.0f K  |  logg = %.2f +/- %.2f dex  |  [Fe/H] = %.2f +/- %.2f dex\n Ms = %.2f +/- %.2f |  Rs = %.2f +/- %.2f'%(teff, dteff, logg, dlogg, feh, dfeh, M, dM, R, dR),ha='center',va='center',fontsize=15)
        plt.savefig(dir_root+'IMAGES/Atmos_all'+myv.PRD_EXT+'.png')

    return teff,feh,logg,M,R,BV,vmicro,vmacro


def yarara_vcat(dir_root, sub_dico='matching_diff', vsini=None, Prot=None, debug=False, std_bias_kms = 0.1, ref_value=None):

    if myv.VERBOSE:
        myf.print_box('\n---- RECIPE : VSINI EXTRACTION ----\n')

    sinfo = import_star_info(dir_root)
    ccf_values = import_ccf(dir_root,'G2')

    instrument = dir_root.split('/')[-2]
    ins = instrument.split('_')[0]

    teff = sinfo['Teff']['SNAKY']
    logg = sinfo['Log_g']['SNAKY']
    feh = sinfo['FeH']['SNAKY']
    try:
        ins_res = sinfo['FWHM']['O2']
    except:
        ins_res = np.nan
    
    samples_table = pd.read_csv(dir_root+'WORKSPACE/Analyse_samples.csv.gz')
    teff = samples_table['teff']
    logg = samples_table['logg']
    feh = samples_table['feh']

    myv.vprint(' [INFO] Teff = %.0f'%(np.median(teff)))
    myv.vprint(' [INFO] Logg = %.2f'%(np.median(logg)))
    myv.vprint(' [INFO] FeH = %.2f'%(np.median(feh)))

    vmacro = get_vmacro(teff,logg,feh,source='Cretignier+26') #Doyle, Bruntt, or Cretignier

    #ratio = garfield/kitty
    calib_teff = myc.tableXY([3500,3750,4000,4250,4500,4750,5000,5250,5500,5750,6000,6250,6500],[1.245,1.245,1.245,1.245,1.23,1.21,1.17,1.13,1.10,1.080,1.075,1.075,1.075]) ; calib_teff.null()
    calib_feh = myc.tableXY([-1.0,-0.75,-0.50,-0.25,0.00,0.25],[-0.035,-0.025, -0.014, -0.004, 0.000, 0.009]) ; calib_feh.null()

    calib_teff.interpolate(new_grid=teff,method='linear',replace=False)
    calib_feh.interpolate(new_grid=feh,method='linear',replace=False)
    ratio_kitty = calib_teff.y_interp+calib_feh.y_interp

    if ins=='FEROS':
        ratio_kitty = np.ones(len(samples_table))*1.3

    ratio = {'G2':np.ones(len(samples_table)),'Garfield':np.ones(len(samples_table)),'Kitty':ratio_kitty}
    #ratio = {'Garfield':1,'Kitty':1}
    myv.vprint(' [INFO] Ratio GARFIELD/KITTY from (Teff,FeH) calibration = %.2f'%(np.median(ratio['Kitty'])))


    if ins in myv.instrument_res_kms.keys():
        ref_resolution = myv.instrument_res_kms[ins]
    else:
        ref_resolution = ins_res
    diff = ref_resolution - ins_res

    if (ref_resolution!=ref_resolution)&(np.nanmedian(ccf_values['fwhm'].y)>30):
        ref_resolution = 5

    myv.vprint(' [INFO] Reference instrument resolution = %.2f km/s'%(ref_resolution))
    myv.vprint(' [INFO] Telluric measured one = %.2f km/s (Delta = %.2f)'%(ins_res,diff))
    
    #if instrument[0:6]=='SOPHIE':
    #    if abs(diff)>1:
    #        print(Fore.YELLOW+'\n [WARNING] Resolution is too different from reference value, O2 correction applied. \n'+Fore.RESET)
    #    else:
    #        ins_res = ref_resolution
    #else:
    #    ins_res = ref_resolution

    calib_product = 'Calib_HARPN_GKM_vsini_HD10700.p'
    myv.vprint(' [INFO] Calibration product used : %s'%(calib_product))
    calib = pd.read_pickle(MATERIAL_DIR+'/'+calib_product)

    calib_curve = {}
    for kw in ['GARFIELD','KITTY']:
        G = myc.tableXY(calib['%s_FWHM'%(kw)],calib['%s_VSINI'%(kw)],0*calib['%s_VSINI'%(kw)])
        if kw=='GARFIELD': # solar correction
            G.y = np.sqrt(G.y**2-1.20**2)
            #G.y = G.y-0.598/(G.x/6.085)**2
        elif kw=='KITTY':  # solar correction
            G.y = np.sqrt(G.y**2-1.20**2)
            #G.y = G.y-0.708/(G.x/5.895)**2
        calib_curve[kw] = G
    calib_curve['G2'] = calib_curve['GARFIELD']

    vmacro = get_vmacro(teff,logg,feh,source='Cretignier+26') #Doyle, Bruntt, or Cretignier
    vmacro_sun = get_vmacro(5775,4.44,0.00,source='Cretignier+26')
    vmacro_hd10700 = get_vmacro(5338,4.54,-0.47,source='Cretignier+26')
    dmacro = {kw:vmacro[kw]-vmacro_hd10700[kw] for kw in vmacro.keys()}
    for kw in vmacro.keys():
        myv.vprint(' [INFO] Cretignier+26 vmacro for %s (Teff , Logg, FeH) = %.1f kms (Sun = %.1f km/s, TauCeti = %.1f km/s)'%(kw,np.median(vmacro[kw]),vmacro_sun[kw],vmacro_hd10700[kw]))

    vsini_cdf = {}
    samples = []
    num = -1

    if np.nanmedian(ccf_values['fwhm'].y)>30:
        masks = ['G2','G2']
    else:
        masks = ['Garfield','Kitty']

    for mask in masks:
        num += 1
        ccf = import_ccf_profile(dir_root,mask)
        ccf_values = import_ccf(dir_root,mask)
        
        vrad = ccf[sub_dico]['ccf_vrad']
        vmax = np.max(vrad)
        try:
            saveG = 1-ccf[sub_dico]['ccf_shifted'].T
        except:
            saveG = 1-ccf[sub_dico]['ccf_flux'].T
        saveG = saveG - np.min(saveG,axis=1)[:,np.newaxis]
        saveG = saveG/np.max(saveG,axis=1)[:,np.newaxis]
        fwhmG_raw = []
        plt.figure()
        for s in saveG:
            c = myc.tableXY(vrad,s,0*s)
            c.clip(min=[-vmax*0.4,None],max=[vmax*0.4,None])
            c.interpolate(new_grid=1000,method='cubic')
            v2 = c.x[c.x>0][np.argmin(abs(c.y-0.5)[c.x>0])]
            v1 = c.x[c.x<0][np.argmin(abs(c.y-0.5)[c.x<0])]
            fwhm = (v2-v1)/1000
            fwhmG_raw.append(fwhm)
        fwhmG_raw = np.array(fwhmG_raw)  # TBD this could be removed
        fwhmG_raw = ccf_values['fwhm'].y #BECAUSE calibration were obtained from the FWHM yarara_ccf fit

        kw = mask.upper()

        if ins in myv.instrument_res_kms.keys():
            ins_calib = ins
        else:
            loc = myf.find_nearest(list(myv.instrument_res_kms.values()),ref_resolution)[0][0]
            ins_calib = list(myv.instrument_res_kms.keys())[loc]
            print(Fore.YELLOW+'\n [WARNING] %s is no part of the calibrated instruments'%(ins))
            print(' [WARNING] The list of existing instruments is:')
            for ins_item in myv.instrument_res_kms.keys():
                print(' ○ ',ins_item)
            print(' [WARNING] Closest instrument found based on resolution: %s'%(ins_calib))
            print("\n"+Fore.RESET)
        #print(np.round(np.median(fwhmG_raw),3),ins_res,ref_resolution)
        fwhmG = np.sqrt(fwhmG_raw**2 - ref_resolution**2) # correct the PSF deconvolve FWHM
        fwhmG = fwhmG * np.random.choice(ratio[mask],1000,replace=False)[:,np.newaxis] #transform Kitty in Garfield or keep Garfield
        fwhmG = np.sqrt(fwhmG**2 - np.random.choice(dmacro[mask],1000,replace=False)[:,np.newaxis]**2) #correct vmacro toward TauCeti value
        fwhmG = np.sqrt(fwhmG**2 + ref_resolution**2)
        #print(np.round(np.median(fwhmG),3),ins_res,ref_resolution)

        if ins_calib == 'NEID-HE':
            ins_calib = 'SOPHIE'
        
        if ins_calib=='PEPSI':
            ins_calib = 'ESPRESSO'

        if ins_calib=='HARPS':
            ins_calib = 'HARPS03'

        calib_ins = pd.read_csv(MATERIAL_DIR+'/Table_calib_vsini_%s.csv'%('GARFIELD'),index_col=0)
        calib_ins = myc.tableXY(calib_ins[ins_calib],calib_ins['HARPN'],0*calib_ins[ins_calib]) #reference HARPN
        calib_ins.order()
        fwhmG_HARPN = []
        for f in fwhmG:
            calib_ins.interpolate(new_grid=f,replace=False,method='linear')
            fwhmG_HARPN.append(calib_ins.y_interp)
        fwhmG_HARPN = np.array(fwhmG_HARPN)

        #print(np.round(np.median(fwhmG_HARPN),3))
        fwhmG_HARPN[fwhmG_HARPN!=fwhmG_HARPN] = np.min(calib_ins.y) # to avoid crash

        V = []
        G = calib_curve['GARFIELD']#[kw]

        for f in fwhmG_HARPN:
            G.interpolate(new_grid=f,method='linear',replace=False)
            V.append(G.y_interp)
        V = np.array(V)
        V[V!=V] = 0.0

        if debug:
            plt.figure()
            plt.errorbar(ccf_values['rv'].x,fwhmG_raw,yerr=0*fwhmG_raw,marker='.',capsize=0,ls='')
            plt.errorbar(ccf_values['rv'].x,np.nanmean(fwhmG,axis=0),yerr=np.std(fwhmG,axis=0),marker='.',capsize=0,ls='')
            plt.errorbar(ccf_values['rv'].x,np.nanmean(fwhmG_HARPN,axis=0),yerr=np.std(fwhmG_HARPN,axis=0),marker='.',capsize=0,ls='')
            plt.errorbar(ccf_values['rv'].x,np.nanmean(V,axis=0),yerr=np.std(V,axis=0),marker='.',capsize=0,ls='')

        plt.figure('vsin3')
        plt.subplot(3,1,num+1)
        plt.errorbar(ccf_values['rv'].x,np.mean(V,axis=0),yerr=np.std(V,axis=0),marker='.',capsize=0,color='C%.0f'%(num),ls='')
        plt.ylabel(r'$v$ $\sin$ $i$ [km/s]')
        plt.tick_params(labelbottom=False,direction='inout',top=True)
        ax = plt.gca()
        ax.twinx()
        plt.errorbar(ccf_values['rv'].x,fwhmG_raw,yerr=0*fwhmG_raw,marker='.',capsize=0,color='C%.0f'%(num),ls='')
        plt.ylabel('CCF FWHM\n%s [km/s]'%(ins_calib))
        plt.subplot(3,1,3)
        plt.errorbar(ccf_values['rv'].x,np.mean(V,axis=0),yerr=np.std(V,axis=0),marker='.',capsize=0,color='C%.0f'%(num),ls='')

        std_bias = std_bias_kms # 100m/s bias uncertainty is a good guess
        std_accuracy = np.nanmedian(np.nanstd(V,axis=0)) 
        std_precision = myf.mad(np.nanmedian(fwhmG_HARPN,axis=0))
        std_tot = np.sqrt(std_bias**2+std_accuracy**2+std_precision**2)
        myv.vprint('\n [INFO] Mask = %s'%(kw))
        myv.vprint(' [INFO] Bias uncertainties = %.0f m/s'%(std_bias*1000))
        myv.vprint(' [INFO] Accuracy uncertainties = %.0f m/s'%(std_accuracy*1000))
        myv.vprint(' [INFO] Precision uncertainties = %.0f m/s'%(std_precision*1000))
        myv.vprint(' [INFO] Total uncertainties = %.0f m/s'%(std_tot*1000))

        plt.figure('vsini2')
        flat_vsini = np.ravel(fwhmG_HARPN)
        flat_vsini[flat_vsini<np.min(G.x)] = np.min(G.x)
        flat_vsini = flat_vsini + np.random.randn(len(flat_vsini))*std_tot
        G.interpolate(new_grid=flat_vsini,method='linear',replace=False)
        V = G.y_interp

        sample = V
        sample = sample[sample>=0]
        samples.append(sample)
        plt.hist(sample,bins=100,density=True,histtype='step')
        plt.hist(sample,bins=100,density=True,alpha=0.4,color='C%.0f'%(num),label=mask+' : \nv sin i = %.2f +/- %.2f km/s'%(np.mean(sample),np.std(sample)))
        plt.xlabel(r'v $\sin$ i [km/s]')
        plt.figure('dust')
        infos = plt.hist(sample,bins=100,density=True,histtype='step',cumulative=True)
        plt.close('dust')
        infos = myc.tableXY(0.5*(infos[1][1:]+infos[1][:-1]),infos[0],0*infos[0])
        vsini_cdf[kw] = infos
    plt.legend()     
    samples = np.hstack(samples)
    plt.hist(samples,bins=100,density=True,histtype='step',color='k',lw=2)
    plt.title('v broad = %.2f km/s \n v sin i = %.2f +/- %.2f km/s'%(np.mean(vmacro['Garfield']),np.mean(samples),np.std(samples)))
    myv.vprint('\n [INFO] v sin i = %.2f +/- %.2f km/s'%(np.mean(samples),np.std(samples)))
    if ref_value is not None:
        plt.axvline(x=ref_value, color='k', ls='-.', lw=1, alpha=0.7)
    plt.savefig(dir_root+'IMAGES/Vsini_CCF_hist'+myv.PRD_EXT+'.png')

    plt.figure('vsin3')
    plt.subplot(3,1,3)
    plt.tick_params(top=True,direction='inout')
    plt.axhline(y=np.mean(samples),color='k',ls=':')
    plt.axhline(y=np.mean(samples)+np.std(samples),color='k',ls='-.')
    plt.axhline(y=np.mean(samples)-np.std(samples),color='k',ls='-.')
    plt.xlabel('Jdb - 2,400,000 [days]')
    plt.ylabel(r'$v$ $\sin$ $i$ [km/s]')
    plt.subplots_adjust(hspace=0,top=0.96,right=0.83)

    plt.savefig(dir_root+'IMAGES/Vsini_CCF_sts'+myv.PRD_EXT+'.png')

    samples = np.random.choice(samples,99999)
    samples_table = pd.read_csv(dir_root+'WORKSPACE/Analyse_samples.csv.gz')
    samples_table['vsini'] = samples
    samples_table.to_csv(dir_root+'WORKSPACE/Analyse_samples.csv.gz',index=False)

    plt.figure('dust')
    infos = plt.hist(samples,bins=100,density=True,histtype='step',cumulative=True)
    plt.close('dust')
    infos = myc.tableXY(0.5*(infos[1][1:]+infos[1][:-1]),infos[0],0*infos[0])
    vsini_cdf['ALL'] = infos

    return samples

def yarara_vsini(dir_root, Prot=None, Rs=None):

    sinfo = import_star_info(dir_root)
    if (Prot is None):
        try:
            Prot = sinfo['Prot']['FINCH']
            if Prot!=0:
                myv.vprint(' [INFO] Stellar Prot measured by FINCH found! Prot = %.1f days'%(Prot))
            else:
                Prot = sinfo['Prot']['YARARA']
        except:
            pass
    else:
        myv.vprint(' [INFO] Stellar Prot provided by user! Prot = %.1f days'%(Prot))
    
    if (Prot!=Prot)|(Prot==0): #np.nan or null
        Prot = None

    vsun = 1.87 ; psun = 27.5

    vmax_veq = 2*pd.read_pickle(glob.glob(dir_root+'STAR_INFO/Stell*')[0])['FWHM']['G2']
    samples_table = pd.read_csv(dir_root+'WORKSPACE/Analyse_samples.csv.gz')

    sample_Rs = np.array(samples_table['rs'])
    sample_vsini = np.array(samples_table['vsini'])

    if Rs is not None:
        sample_Rs = np.random.randn(99999)*0.01+Rs #1% radius uncertainty

    sample_prot90 = psun*sample_Rs/(sample_vsini/vsun) 

    p90m = [
        np.nanpercentile(sample_prot90,50),
        np.nanpercentile(sample_prot90,84)-np.nanpercentile(sample_prot90,50),
        np.nanpercentile(sample_prot90,50)-np.nanpercentile(sample_prot90,16)]

    myv.vprint(' [INFO] ')
    myv.vprint(' [INFO] Prot (if i=90) estimated = %.2f [%.2f - %.2f] days '%(p90m[0],p90m[0]-p90m[1],p90m[0]+p90m[2]))

    rm = [
        np.nanpercentile(sample_Rs,50),
        np.nanpercentile(sample_Rs,84)-np.nanpercentile(sample_Rs,50),
        np.nanpercentile(sample_Rs,50)-np.nanpercentile(sample_Rs,16)]

    vm = [
        np.nanpercentile(sample_vsini,50),
        np.nanpercentile(sample_vsini,84)-np.nanpercentile(sample_vsini,50),
        np.nanpercentile(sample_vsini,50)-np.nanpercentile(sample_vsini,16)]

    myv.vprint(' [INFO] vsini = %.2f [%.2f - %.2f] km/s'%(vm[0],vm[0]-vm[1],vm[0]+vm[2]))

    plt.figure('inclination',figsize=(18,5))
    plt.subplot(1,4,1)
    plt.title(r'Rs = %.2f$^{+%.2f}_{-%.2f}$ Rs'%(rm[0],rm[1],rm[2]))
    pby,pbx = np.histogram(sample_Rs,bins=np.arange(0,2,0.025),density=True)
    pbx = 0.5*(pbx[1:]+pbx[0:-1])
    plt.fill_between(pbx,pby,alpha=0.2,color='C0')
    plt.plot(pbx,pby,color='C0')
    plt.xlim(0,2)
    plt.ylim(0,None)
    plt.xlabel(r'Radius [Rs]')

    plt.subplot(1,4,2)
    plt.title(r'v $\sin$ i = %.2f$^{+%.2f}_{-%.2f}$ km/s'%(vm[0],vm[1],vm[2]))
    pby,pbx = np.histogram(sample_vsini,bins=np.arange(0,vmax_veq,vmax_veq/200),density=True)
    pbx = 0.5*(pbx[1:]+pbx[0:-1])
    plt.fill_between(pbx,pby,alpha=0.2,color='C0')
    plt.plot(pbx,pby,color='C0')
    plt.axvline(x=vm[0],ls=':',color='C0')
    if vm[0]>4:
        plt.xlim(vm[0]*0.5,vm[0]*1.5)
    else:
        plt.xlim(0,7)
    plt.ylim(0,None)
    plt.tick_params(top=True,bottom=True,direction='inout',which='both')
    plt.xlabel(r'v $\sin$ i [km/s]')

    sample_sini = np.sqrt(1-np.random.rand(99999)**2) # np.sin of np.arccos
    sample_prot = np.ravel(sample_prot90*sample_sini)

    plt.subplot(2,4,3)
    plt.title(r'$P_{90} = %.1f^{+%.1f}_{-%.1f}$ days'%(p90m[0],p90m[1],p90m[2]))

    pby,pbx = np.histogram(sample_prot90,bins=np.linspace(0,100,100),density=True)
    pbx = 0.5*(pbx[1:]+pbx[0:-1])
    plt.fill_between(pbx,pby,alpha=0.2,color='C1')
    plt.plot(pbx,pby,color='C1')
    plt.axvline(x=p90m[0],ls=':',color='k')

    plt.xlim(0,100)
    plt.ylim(0,None)
    #plt.xscale('log')
    plt.tick_params(top=True,bottom=True,direction='inout')

    plt.subplot(2,4,7)
    pby,pbx = np.histogram(sample_prot,bins=np.linspace(0,100,100),density=True)
    pbx = 0.5*(pbx[1:]+pbx[0:-1])
    plt.fill_between(pbx,pby,alpha=0.2,color='C0')
    plt.plot(pbx,pby,color='C0')

    if Prot is not None:
        sample_prot_known = np.random.randn(99999)*(0.10*Prot)+Prot #10% prot uncertainty
        pby,pbx = np.histogram(sample_prot_known,bins=np.linspace(0,100,100),density=True)
        pbx = 0.5*(pbx[1:]+pbx[0:-1])
        plt.fill_between(pbx,pby,alpha=0.2,color='k')
        plt.plot(pbx,pby,color='k')
        plt.axvline(x=Prot,color='k',ls=':',label=r'$P_{rot}$=%.1f days'%(Prot))
        plt.legend()

    plt.xlim(0,100)
    plt.ylim(0,None)
    plt.xlabel(r'$P_{rot}$ [days]')
    plt.tick_params(top=True,bottom=True,direction='inout')

    if Prot is not None:
        plt.subplot(1,4,2)
        sample_veq = vsun*sample_Rs/(sample_prot_known/psun)
        pby,pbx = np.histogram(sample_veq,bins=np.arange(0,vmax_veq,vmax_veq/200),density=True)
        pbx = 0.5*(pbx[1:]+pbx[0:-1])
        plt.fill_between(pbx,pby,alpha=0.2,color='k')
        plt.plot(pbx,pby,color='k')
        plt.axvline(np.median(sample_veq),color='k',label='v=%.1f km/s'%(np.median(sample_veq)),ls=':')
        plt.legend()

    sample_sininc = np.ravel((sample_vsini/vsun)*(sample_prot/psun)/sample_Rs)
    
    plt.subplot(1,4,4)
    
    iby,ibx = np.histogram(sample_sininc,bins=np.arange(0,1,0.01),density=True)
    ibx = 0.5*(ibx[1:]+ibx[0:-1])
    iby = iby/np.sum(iby)

    #pdf isotropic analytic
    p = ibx/np.sqrt(1-ibx**2) ; p /= np.sum(p)

    plt.fill_between(ibx,iby,alpha=0.2,color='C0')
    plt.plot(ibx,p,color='C0',label='Isotropic = [33-60-81]')
    plt.legend(loc=2)

    if Prot is not None:
        result = myf.posterior_sin_i_from_samples(sample_veq, #Masuda+20 2001.04973
                                 sample_vsini,
                                 Ndraw=200000,
                                 Npost=399999,
                                 vsini_sigma_override=None,
                                 rng_seed=0,
                                 plot=False)
        sample_sininc = result['sin_i_post']
        iby,ibx = np.histogram(sample_sininc,bins=np.arange(0,1,0.01),density=True)
        ibx = 0.5*(ibx[1:]+ibx[0:-1])
        iby = iby/np.sum(iby)
        plt.fill_between(ibx,iby,alpha=0.2,color='g')
        plt.plot(ibx,iby,color='g',label='Posterior (isotropic)')
        plt.legend()

        sample_sininc2 = np.ravel((sample_vsini/vsun)*(sample_prot_known/psun)/sample_Rs)
        f_bad = 100*np.sum(sample_sininc>1)/len(sample_sininc)
        print(' [INFO] Bad fraction not in [0,1] = %.1f %%'%(f_bad))

        iby,ibx = np.histogram(sample_sininc2,bins=np.arange(0,1,0.01),density=True)
        ibx = 0.5*(ibx[1:]+ibx[0:-1])
        iby = iby/np.sum(iby)
        plt.fill_between(ibx,iby,alpha=0.2,color='k')#,label='Measured = [%.0f-%.0f]'%(Ii,Is))
        plt.plot(ibx,iby,color='k')

    I = np.arcsin(np.nanpercentile(sample_sininc,50))*180/np.pi
    Ii =  np.arcsin(np.nanpercentile(sample_sininc,16))*180/np.pi
    Is = np.arcsin(np.nanpercentile(sample_sininc,84))*180/np.pi

    I = [I,90][int(I!=I)]
    Ii = [Ii,90][int(Ii!=Ii)]
    Is = [Is,90][int(Is!=Is)]
    #plt.legend(loc=2)

    myv.vprint('\n [INFO] Isotropic distribution = 60 [33 - 81] degree')
    myv.vprint(' [INFO] Inclination estimated = %.0f [%.0f - %.0f] degree'%(I,Ii,Is))
    plt.title(r'$i = %.0f^{+%.0f}_{-%.0f}$ [°]  |  $i =$ [%.0f - %.0f - %.0f]'%(I,Is-I,I-Ii,Ii,I,Is))

    plt.axvline(x=np.sin(I*np.pi/180),color='k',ls=':')
    plt.tick_params(top=True,bottom=True,direction='inout',which='both')
    plt.xlim(0,1)
    plt.ylim(0,None)
    plt.xlabel(r'$\sin i$ []')
    plt.subplots_adjust(left=0.05,right=0.97)
    plt.savefig(dir_root+'IMAGES/Vsini_inclination'+myv.PRD_EXT+'.png')

    samples_table = pd.read_csv(dir_root+'WORKSPACE/Analyse_samples.csv.gz')
    if Prot is not None:
        samples_table['prot'] = sample_prot_known
    else:
        samples_table['prot'] = sample_prot
    samples_table['sini'] = np.random.choice(sample_sininc,99999)
    samples_table.to_csv(dir_root+'WORKSPACE/Analyse_samples.csv.gz',index=False)

    if Prot is not None:
        plt.figure(figsize=(8,8))
        plt.axes([0,0,1,1])
        iby,ibx = np.histogram(np.arcsin(samples_table['sini'])*180/np.pi,bins=np.arange(0,180,1),density=True)
        ibx = 0.5*(ibx[1:]+ibx[0:-1])
        iby = iby/np.sum(iby)
        iby = myf.smooth(iby,box_pts=5,shape='savgol')
        iby[iby<0] = 0

        theta = ibx*np.pi/180 ; r = 1.0 + 10 * iby
        X = r * np.cos(theta) ; Y = r * np.sin(theta)
        plt.plot(X, Y, '-k')

        sinc = np.random.choice(samples_table['sini'],10000)
        r = np.random.randn(10000)*0.03+1
        plt.scatter(np.sqrt(1-sinc**2)*r,sinc*r,color='k',marker='.',alpha=0.01,s=20)
        for perc,lw in zip([16,50,84],[1,2,1]):
            x = np.sqrt(1-np.nanpercentile(sinc,perc)**2)
            y = np.nanpercentile(sinc,perc)
            plt.plot([0,x],[0,y],color='k',lw=lw,ls=['-','-.'][int(lw==1)])
            plt.text(x*1.1,y*1.1,'%.0f°'%(np.arcsin(y)*180/np.pi))
        plt.plot(np.cos(np.linspace(0,2*np.pi,100)),np.sin(np.linspace(0,2*np.pi,100)),color='k')
        plt.plot([0,0],[-1,1],color='k',alpha=0.3,lw=1,ls='--')
        plt.axis('off')
        plt.xlim(-1.1,2.9)
        plt.ylim(-2.0,2.0)
        plt.plot([2.7,2.8,2.7],[0.1,0.0,-0.1],color='k')
        plt.scatter([2.72],[0.015],color='white',ec='k')
        eye = np.linspace(7*np.pi/8,9*np.pi/8,100)
        plt.plot(np.cos(eye)*0.1+2.8,np.sin(eye)*0.1,color='k')
        plt.plot([0,2.6],[0,0],ls=':',lw=1,color='k',alpha=0.4)
        plt.savefig(dir_root+'IMAGES/Stellar_inclination'+myv.PRD_EXT+'.png')


def yarara_activity_index(files, rv_sys, shift_rv, fwhm=6.0, material=None, sub_dico='matching_diff'):

    if myv.VERBOSE:
        myf.print_box('\n---- RECIPE : ACTIVITY PROXIES EXTRACTION ----\n')

    #[center, half-window, hole_size, half-window-continuum,database_kw, subplot]
    Ca2H =  myv.Ca2H+[None]
    Ca2K =  myv.Ca2K+[None]
    Ca1 =   myv.Ca1+[2]
    Mg1a =  myv.Mg1a+[6]
    Mg1b =  myv.Mg1b+[7]
    Mg1c =  myv.Mg1c+[8]
    NaDl =  myv.NaDl+[3]
    NaDr =  myv.NaDr+[4]
    NaDC = myv.NaDC+[None] 
    Ha = myv.Ha+[9]
    Hb = myv.Hb+[10]
    Hc = myv.Hc+[11]
    Hd = myv.Hd+[12]
    Heps = myv.Heps+[None] 
    He1D3 = myv.He1D3+[5] 
    
    all_proxies = [Ca2H, Ca2K, Ca1, Mg1a, Mg1b, Mg1c, NaDl, NaDr, NaDC, Ha, Hb, Hc, Hd, Heps, He1D3]
    
    grid, flux, err_flux = import_sts(files, rv_shift=shift_rv, err=False, sub_dico=sub_dico, scale=False)
    grid = np.round(grid/100,2)
    dgrid = np.mean(np.diff(grid))

    all_prox_names = np.array(all_proxies)[:,4]
    proxy_found = ((np.array(all_proxies)[:,0]-np.nanmin(grid))>0)&((np.nanmax(grid))>0)
    all_proxies = list(np.array(all_proxies)[proxy_found])

    flux_ref = myf.master_spectrum(grid,flux)
    #ratio = flux/(flux_ref+1e-6)

    #flux_err = myf.mad(flux - flux_ref,axis=1)
    #err_flux = np.ones(len(grid))*flux_err[:,np.newaxis]

    #if material is not None:
    #    correction_factor = np.array(material['correction_factor']).astype('float32')
    #else:
    #    correction_factor = np.ones(len(grid)).astype('float32')

    correction_factor = np.ones(len(grid)).astype('float32')

    def find_proxy(vec):
        center = myf.doppler_r(vec[0],rv_sys*1000)[0]
        left = myf.doppler_r(vec[0]-vec[1],rv_sys*1000)[0]
        right = myf.doppler_r(vec[0]+vec[1],rv_sys*1000)[0]
        
        center_idx_proxy = myf.find_nearest(grid,center)[0][0]
        left_idx_proxy = myf.find_nearest(grid,left)[0][0]
        right_idx_proxy = myf.find_nearest(grid,right)[0][0]

        left = myf.doppler_r(vec[0]-vec[3],rv_sys*1000)[0]
        right = myf.doppler_r(vec[0]+vec[3],rv_sys*1000)[0]

        left_idx_cont = myf.find_nearest(grid,left)[0][0]
        right_idx_cont = myf.find_nearest(grid,right)[0][0]            
        
        return int(center_idx_proxy), int(left_idx_proxy), int(right_idx_proxy), int(left_idx_cont), int(right_idx_cont)
    
    def extract_proxy(vec):
        c, l, r, l_cont, r_cont = find_proxy(vec)            

        lines = (flux[:,l:r].astype('float32')*correction_factor[l:r])/10000.
        line_ref = (flux_ref[l:r].astype('float32')*correction_factor[l:r])/10000.
        ratio_left = (flux[:,l_cont:l].astype('float32')/(flux_ref[l_cont:l].astype('float32')+1))
        ratio_right = (flux[:,r:r_cont].astype('float32')/(flux_ref[r:r_cont].astype('float32')+1))   
        
        continuum=1
        if r!=l:
            r+=1
        if l_cont!=l:
            r_cont+=1
            continuum = np.hstack([ratio_left,ratio_right]) 
            continuum = np.nanmedian(continuum,axis=1)
            continuum[np.isnan(continuum)] = 1
            continuum[continuum==0] = 1

        proxy = np.sum(lines,axis=1)
        err_flux = myf.mad(lines-line_ref,axis=1)
        err_flux = np.ones(r-l)*err_flux[:,np.newaxis]
        proxy_std = np.sum((err_flux)**2,axis=1)
        proxy_std = np.sqrt(proxy_std)
        norm_proxy = (r - l)
        
        proxy/=continuum
        proxy_std/=continuum
        
        if norm_proxy:
            proxy /= norm_proxy
            proxy_std /= norm_proxy      
            return proxy, proxy_std, c, l, r
        else:
            return 0*proxy, 0*proxy_std, c, l, r

    save = {'null':0}
    mask_activity = np.zeros(len(grid))
    for p in all_proxies:
        proxy, proxy_std, c, l, r= extract_proxy(p)
        fwhm_line = r - l
        fwhm_wave = 1.5*fwhm*grid[c]/3e5
        fwhm_sampling = int(fwhm_wave/dgrid) 
        if fwhm_line<fwhm_sampling:
            vel_extension = int(0.5*(fwhm_sampling-fwhm_line))
            r = r+vel_extension
            l = l-vel_extension
        mask_activity[l:r] = 1    
        save[p[4]] = proxy
        save[p[4]+'_std'] = proxy_std
    del save['null']
    
    for n in all_prox_names:
        if n not in save.keys():
            save[n] = np.zeros(len(files[-1]))
            save[n+'_std'] = np.zeros(len(files[-1]))
        
    #del ratio
    
    def non_neg(prox,prox_std):
        mask = (prox<=0)
        prox[mask] = np.median(prox[~mask])
        prox_std[mask] = np.median(prox[~mask])*0.99
        return prox, prox_std
    
    for kw in save.keys():
        if kw[-3:]!='std':
            save[kw],save[kw+'_std'] = non_neg(save[kw],save[kw+'_std'])

    save['CaII'] = 0.5*(save['CaIIK']+save['CaIIH'])  
    save['CaII_std'] = 0.5*np.sqrt((save['CaIIK_std'])**2+(save['CaIIH_std'])**2)

    save['NaD'] = 0.5*(save['NaD1']+save['NaD2'])  
    save['NaD_std'] = 0.5*np.sqrt((save['NaD1_std'])**2+(save['NaD2_std'])**2)

    save['MgI'] = 0.5*(save['MgIa']+save['MgIb']+save['MgIc'])  
    save['MgI_std'] = 0.5*np.sqrt((save['MgIa_std'])**2+(save['MgIb_std'])**2+(save['MgIc_std'])**2)
    
    conv_slope = 1
    conv_offset = 0

    save['RHK'] = np.nan
    save['RHK_std'] = np.nan
    
    tab = pd.DataFrame(save)
    tab['filename'] = files[-1]

    #teff from Ha and NaD EW
    C = np.nanpercentile(save['NaDC'],50)
    X = np.nanpercentile(save['Ha'],50)
    Y = np.nanpercentile(save['NaD'],50)
    Z = np.nanpercentile(save['MgI'],50)
    H = np.nanpercentile(save['Hb'],50)
    CT = {'NaDC':C,'Ha':X,'NaD':Y,'MgI':Z,'Hb':H}

    return tab, CT, mask_activity

def yarara_compute_snr(dir_root,sub_dico):
    material = import_material(dir_root)
    summary = import_summary(dir_root)
    master = material['reference_spectrum']*material['correction_factor']
    grid = material['wave']
    star_info = import_star_info(dir_root)

    rv_sys = star_info['Rv_sys']['SNAKY']
    teff = star_info['Teff']['SNAKY']
    logg = star_info['Log_g']['SNAKY']
    feh = star_info['FeH']['SNAKY']

    template_empi = import_stellar_template(teff,logg=logg,feh=feh,model='SNAKY',rv_sys=rv_sys)
    template_empi.interpolate(grid,method='linear',fill_value=np.nan)

    grid, flux, err_flux = import_sts(summary['filename'], err=False, sub_dico=sub_dico)    

    ratio = master/template_empi.y-1
    mask = template_empi.y!=0
    for j in range(4):
        ratio = ratio - myf.smooth(ratio,100)
        sigma = myf.mad(ratio[mask])
        mask = (ratio<2*sigma)&(ratio>(-2*sigma))&(template_empi.y!=0)
    snr = 1/sigma
    print(' [INFO] Master spectrum SNR = %.0f'%(snr))

    snrs = []
    for f in flux:
        ratio = f/template_empi.y-1
        mask = template_empi.y!=0
        for j in range(4):
            ratio = ratio - myf.smooth(ratio,100)
            sigma = myf.mad(ratio[mask])
            mask = (ratio<2*sigma)&(ratio>(-2*sigma))&(template_empi.y!=0)
        snrs.append(int(np.round(1/sigma,0)))
    return np.array(snrs)

def yarara_atmos_fast_rotator(dir_root, rv_sys, vsini, model='SNAKY'):

    if myv.VERBOSE:
        myf.print_box('\n---- RECIPE : ATMOS PARAMETERS FAST-ROTATORS ----\n')

    master = import_master(dir_root)
    star_info = import_star_info(dir_root)

    master.clip(min=[5700,None],max=[6490,None])
    master.y[master.y<=0] = np.nan
    grid = master.x

    #all_template = []
    parameters = []
    teff_grid = np.arange(3500,6500,250)
    v_grid = np.linspace(0.5*vsini,vsini+(0.5*vsini),15)
    metric = np.zeros((len(teff_grid), len(v_grid)))

    for i,teff in enumerate(teff_grid):
        template = import_stellar_template(teff,feh=0.0,logg=4.3,model=model,rv_sys=rv_sys)
        template.interpolate(new_grid=grid,replace=True,method='cubic',interpolate_x=False)
        template.y[template.y>1] = 1
        template.y[template.y<=0] = np.nan
        for j,v in enumerate(v_grid):
            template.rotation_broadening(veq=v,replace=False)
            #all_template.append(template.degraded.y.copy())
            ratio = abs(template.degraded.y / master.y - 1)
            metric[i, j] = np.nansum(ratio)
            #plt.plot(grid,all_template[-1]/master.y+n)
    
    #for n in range(7):
    #    plt.plot(master.x,1+master.x*0,color='k',zorder=100)

    # bicubic interpolation of the metric   
    f = RectBivariateSpline(teff_grid, v_grid, metric, kx=3, ky=3)

    # initial guess = grid minimum
    imin, jmin = np.unravel_index(np.argmin(metric), metric.shape)
    teff_best = np.round(teff_grid[imin],0)
    v_best = np.round(v_grid[jmin],2)

    x0 = np.array([teff_best,v_best])

    # objective function
    fun = lambda p: f(p[0], p[1])[0, 0]

    res = minimize(
        fun,
        x0=x0,
        bounds=[
            (teff_grid.min(), teff_grid.max()),
            (v_grid.min(), v_grid.max())
        ],
    )

    teff_opt, vsini_opt = res.x
    teff_best = np.round(teff_opt,0)
    v_best = np.round(vsini_opt,2)
    
    dteff = 125
    dv_grid = np.diff(v_grid)[0]*0.5

    chi2 = metric

    L = np.exp(-(chi2 - np.nanmin(chi2))/2)
    L /= np.nansum(L)
    P_teff = np.nansum(L, axis=1)
    P_vsini = np.nansum(L, axis=0)

    myv.vprint(' [INFO] Best params (model = %s) Teff = %.0f +/- %.0f K  | vsini = %.1f +/- %.1f km/s'%(model, teff_best, dteff, v_best, dv_grid))

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(np.log10(metric), origin="lower", aspect="auto", cmap="jet")

    ax.set_xticks(np.arange(len(v_grid)))
    ax.set_xticklabels(np.round(v_grid).astype(int))

    ax.set_yticks(np.arange(len(teff_grid)))
    ax.set_yticklabels(teff_grid)

    ax.set_xlabel(r"$v\sin i$ (km/s)")
    ax.set_ylabel(r"$T_{\rm eff}$ (K)")

    fig.colorbar(im, ax=ax, label="Metric")

    ax.scatter(jmin, imin, color="white", marker="x")

    ax.set_title(
        "model = %s\nTeff = %.0f +/- %.0f K  | vsini = %.1f +/- %.1f km/s"
        % (model, teff_best, dteff, v_best, dv_grid)
    )

    # Create a second axes on top of the image
    ax2 = ax.twinx()

    ax2.plot(
        np.arange(len(v_grid)),
        np.log10(metric)[imin, :],
        color="white", 
        lw=2,
    )
    plt.tick_params(labelright=False,labelleft=False)
    plt.savefig(dir_root + f"IMAGES/Atmos_fast_rotators_{model}.png")

    params = create_atmos_sample(dir_root, teff_best, dteff, 4.3, 0.2, 0.0, 0.5, rv_sys)
    params = list(params)+[v_best]

    samples_teff = np.random.randn(99999)*dteff + teff_best
    samples_vsini = np.random.randn(99999)*dv_grid + v_best
    samples_table = pd.read_csv(dir_root+'WORKSPACE/Analyse_samples.csv.gz')
    samples_table['vsini'] = samples_vsini
    samples_table['teff'] = samples_teff
    samples_table.to_csv(dir_root+'WORKSPACE/Analyse_samples.csv.gz',index=False)

    return params

def yarara_correct_continuum_absorption(dir_root, rv_sys, feh, model, vsini=0.0):
    
    if myv.VERBOSE:
        myf.print_box('\n---- RECIPE : CORRECT ABSORPTION CONTINUUM ----\n')

    ins = dir_root.split('/')[-2].split('_')[0]
    
    master = import_master(dir_root)
    star_info = import_star_info(dir_root)

    reject_zones = [[5875,5910]]
    force_zones = [[3916.5,3918.5],[3923,3926],[3927.5,3929.5],[3931,3933],[3932.1,3932.6],[3935.4,3935.8],[3936,3937],[3937.5,3939],[3940,3943],
                    [3958,3960],[3962.5,3964.5],[3965.5,3967],[3966.9, 3967.4],[3969.9, 3970.4],[3971.5,3972],[3972.5,3975],[3980,3980.5],[3982,3984],
                    ] 

    rassine_zones = np.array(myv.rassine_continuum)

    reject_zones = [np.round(myf.doppler_r(np.array(i),rv_sys*1000)[0],1) for i in reject_zones]
    force_zones = [np.round(myf.doppler_r(np.array(i),rv_sys*1000)[0],1) for i in force_zones]
    rassine_zones = np.round(myf.doppler_r(rassine_zones,rv_sys*1000)[0],2)

    grid = master.x
    master.y[-100:] = 1 #issues on the border right

    parameter = '_'.join(model.split('_')[1:])
    model = model.split('_')[0]
        
    if vsini<20:
        vsini = 0.0

    myv.vprint(' Model selected : %s (%s) - Vsini = %.1f km/s'%(model,parameter,vsini))

    teff = float(parameter.split('T')[-1].split('_')[0])
    logg = float(parameter.split('g')[-1].split('_')[0])    
    template = import_stellar_template(teff,feh=0.0,logg=logg,model='ATLAS',rv_sys=rv_sys)

    template.interpolate(new_grid=grid,replace=True,method='cubic',interpolate_x=False)
    template.y[template.y>1] = 1
    template.y[template.y<0] = 0

    if vsini<20:
        #compute resolution
        s1 = master.copy()
        s2 = template.copy()
        s1.clip(min=[6000,None],max=[6400,None])
        s2.clip(min=[6000,None],max=[6400,None])
        res = []
        resolution_grid = np.arange(60000,140000,1000)
        for reso in resolution_grid:
            chi2 = np.sum(abs(s1.y - myf.instrBroadGaussFast(s2.x,s2.y,reso,maxsig=5.0)))
            res.append(chi2)
        res = np.array(res)
        resolution = resolution_grid[np.argmin(res)]
        myv.vprint('\n [INFO] Resolution found R=%.0f \n'%(resolution))

        del s1
        del s2

        template_flux = myf.instrBroadGaussFast(template.x,template.y,resolution,maxsig=5.0)
    else:
        template.rotation_broadening(veq=vsini,replace=True)
        template_flux = template.y.copy()

    smooth = pd.DataFrame(template.y).rolling(100,min_periods=1,center=True).quantile(0.9)
    smooth= np.array(smooth).T[0]
    template.y[template.y<np.array(smooth)]=0
    template.find_max(vicinity=10)
    
    mask = np.zeros(len(template.x_max)).astype('bool')
    for zone in reject_zones:
        mask = mask|((template.x_max>zone[0])&(template.x_max<zone[1]))
        template.x_max = template.x_max[~mask]
        template.y_max = template.y_max[~mask]
        template.index_max = template.index_max[~mask]

    #replace by fix wavelength     
    anchor_idx = np.unique(myf.find_nearest(template.x,rassine_zones)[0])
    match = myf.match_nearest(anchor_idx, template.index_max)
    
    template.x_max = template.x_max[match[:,1]]
    template.y_max = template.y_max[match[:,1]]
    
    del match
    del anchor_idx

    master.y[master.y>10] = 0 #absurd values

    for zone in force_zones:
        mask = ((template.x<zone[0])|(template.x>zone[1]))
        index_max = np.argmax(template_flux*master.y-mask.astype('float')*99)
        wave_max = template.x[index_max]
        index_max2 = myf.find_nearest(grid,wave_max)[0][0]
        template.x_max = np.hstack([template.x_max,wave_max])
        new_y = template_flux[index_max]/master.y[index_max2]
        template.y_max = np.hstack([template.y_max,new_y])
    ordering = np.argsort(template.x_max)
    template.y_max = template.y_max[ordering]
    template.x_max = template.x_max[ordering]

    parameter = parameter.split('_')
    
    plt.figure(figsize=(15,8))
    plt.subplot(2,1,1)
    plt.title('Before correction',fontsize=16)
    plt.xlabel(r'Wavelength $\lambda$ [$\AA$]',fontsize=16)
    plt.ylabel(r'Flux normalised',fontsize=16)
    plt.plot(master.x, master.y, color='k',label='RASSINE')
    plt.plot(template.x, template_flux, color='r',label='Template (%s, Teff = %s, log(g) = %s, vsini = %.1f)'%(model,parameter[0][1:],parameter[1][1:],vsini))       
    plt.legend(loc=4,prop={'size': 14})
    plt.scatter(template.x_max, template.y_max,color='orange',zorder=10,s=20)
    ax = plt.gca()
    local = myc.tableXY(template.x_max, template.y_max)
    local.x = np.hstack([grid[0],local.x])
    local.y = np.hstack([local.y[0],local.y])
    local.xerr = np.hstack([local.xerr[0],local.xerr])
    local.yerr = np.hstack([local.yerr[0],local.yerr])
    local.interpolate(new_grid=grid,method='linear',interpolate_x=False)
    local.replace_nan(value=np.nanmedian(local.y))
    local.smooth(box_pts=1000,shape='savgol',replace=True)
    correction = local.y_smoothed        

    del local

    if ins[0:4]=='NEID': # because S1D from E2DS without flat field
        template_empi = import_stellar_template(teff,logg=logg,feh=feh,model='SNAKY',rv_sys=rv_sys)
        template_empi.interpolate(master.x,method='linear',fill_value=np.nan)

        s1 = master.y*correction
        s2 = template_empi.y        

        ratio = s2/s1
        ratio[ratio==0] = 1
        extra_correction = myf.smooth(ratio,100)
        extra_correction[extra_correction!=extra_correction] = 1

        extra_correction2 = np.ones(len(correction))
        for c in myf.doppler_r(np.array([myv.Ca2K[0],myv.Ca2H[0]]),rv_sys*1000)[0]:
            mask_line = abs(master.x-c)<2.0
            mask_chrom = abs(master.x-c)<0.3
            mask_phot = mask_line&(~mask_chrom)
            calib = myc.tableXY((master.x-c)[mask_phot],(s1*extra_correction)[mask_phot]/s2[mask_phot])
            calib.fit_poly(d=9,Draw=False)
            model = np.polyval(calib.poly_coefficient,(master.x-c)[mask_line])
            extra_correction2[mask_line] = 1/model
        correction = correction*extra_correction*extra_correction2
        del extra_correction2
        del extra_correction

    correction[correction==np.inf] = 1
    correction[correction!=correction] = 1

    plt.plot(grid,correction,color='orange')
    plt.subplot(2,1,2,sharex=ax,sharey=ax)
    plt.title('After correction',fontsize=16)
    plt.xlabel(r'Wavelength $\lambda$ [$\AA$]',fontsize=16)
    plt.ylabel(r'Flux normalised',fontsize=16)
    plt.plot(master.x, master.y*correction,color='k')
    plt.plot(grid, template_flux,color='r') 
    ymin = ax.get_ylim()[0]
    ymax = ax.get_ylim()[1]
    if ymin<-1:
        plt.ylim(-1,None)
    if ymax>1.5:
        plt.ylim(None,1.5)
    plt.subplots_adjust(left=0.07,right=0.97,top=0.93,bottom=0.1,hspace=0.4)
    for loc,line in zip([0.07,0.19],np.array([myv.Ca2H[0],myv.Ca2K[0]])):
        plt.axes([loc,0.45,0.1,0.1])
        i0 = myf.find_nearest(grid,line)[0][0]
        plt.plot(master.x[i0-500:i0+500], (master.y*correction)[i0-500:i0+500],color='k')
        plt.plot(grid[i0-500:i0+500], template_flux[i0-500:i0+500],color='r') 
        plt.xlim(line-5,line+5)
        myf.only_axis()
    plt.savefig(dir_root+'IMAGES/Correction_absolute_continuum'+myv.PRD_EXT+'.png')
    
    del master
    del grid

    return template_flux, correction

def yarara_measure_berv(dir_root,files,sub_dico='matchinf_diff'):
    myv.vprint('\n [INFO] Automatic BERV measurement... Wait... \n')
    grid, flux, err_flux = import_sts(files, err=False, sub_dico=sub_dico)
    mask = (grid>=6250)&(grid<=6350)  
    grid = grid[mask]
    flux = flux[:,mask]
    sinfo = import_star_info(dir_root)
    teff = sinfo['Teff']['SNAKY']
    logg = sinfo['Log_g']['SNAKY']
    feh = sinfo['FeH']['SNAKY']
    rv_sys = sinfo['Rv_sys']['SNAKY']
    template = import_stellar_template(teff,logg=logg,feh=feh,model='SNAKY',rv_sys=rv_sys)
    template.interpolate(new_grid=grid,method='linear',fill_value=1)  
    
    model = pd.read_csv(MATERIAL_DIR+'/template_oxygen_6250_6350.csv')
    model = myc.tableXY(model['wave'],model['telluric'])
    model.interpolate(new_grid=grid,method='linear')

    ratio = 1-flux/template.y
    ratio[ratio<0.001] = 0.001
    model = 1-model.y
    models = np.array([np.roll(model,i) for i in np.arange(-1000,1001,1)])
    ccfs = np.array([np.nansum(r*models,axis=1) for r in ratio])
    dw = (np.nanargmax(ccfs,axis=1)-1000)*np.nanmean(np.diff(grid))
    berv_computed = dw*3e5/6300

    mask_fail = abs(berv_computed)>31
    berv_computed[mask_fail] = 0

    myv.vprint('\n [INFO] BERV values derived:',berv_computed)

    return berv_computed
    

def yarara_instrumental_resolution(dir_root, files, shift_rv, berv, sub_dico='matching_diff'):

    if myv.VERBOSE:
        myf.print_box('\n---- RECIPE : EXTRACTION INSTRUMENTAL RESOLUTION ----\n')
    
    grid, flux, err_flux = import_sts(files, rv_shift=shift_rv, err=False, sub_dico=sub_dico) 
    missing_values = (berv!=berv)
    if (sub_dico=='matching_diff')&(sum(missing_values)!=0):
        berv_computed = yarara_measure_berv(dir_root,(grid*100, flux[missing_values]*10000, files[-1][missing_values]),sub_dico='matching_diff')
        berv[missing_values] = berv_computed

    berv_mad = myf.mad(berv)

    if berv_mad>3:
        myv.vprint('\n [INFO] BERV MAD = %.1f km/s'%(berv_mad))
        flux_ref = myf.master_spectrum(grid,flux)

        berv_range = np.append(np.arange(0,30,1),50)
        berv_range = np.hstack([-berv_range[::-1],berv_range[1:]])
        chunck = np.linspace(0,len(grid),50).astype('int')
        for c1,c2 in zip(chunck[:-1],chunck[1:]):
            master = []
            for b1,b2 in zip(berv_range[:-1],berv_range[1:]):
                mask = (berv>=b1)&(berv<b2)
                if sum(mask):
                    master.append(np.nanmedian(flux[mask,c1:c2],axis=0))
                else:
                    master.append(np.nan*flux_ref[c1:c2])
            master = np.array(master)
            flux_ref[c1:c2] = np.nanmedian(master,axis=0)
    else:
        print(Fore.YELLOW+'\n [WARNING] BERV SPAN too small (%.1f), use of a reference spectrum'%(berv_mad)+Fore.RESET)
        
        sinfo = import_star_info(dir_root)
        teff = sinfo['Teff']['SNAKY']
        logg = sinfo['Log_g']['SNAKY']
        feh = sinfo['FeH']['SNAKY']
        rv_sys = sinfo['Rv_sys']['SNAKY']

        template = import_stellar_template(teff,logg=logg,feh=feh,model='SNAKY',rv_sys=rv_sys)
        template.interpolate(new_grid=grid,method='linear',fill_value=1)        
        flux_ref = template.y

    flux_ref[flux_ref<=0] = 1
    flux_ref[flux_ref>1] = 1

    flux /= (flux_ref+1e-8)

    if True:
        for i in np.arange(len(files[-1])):
            flux[i] = myf.interpolate_rv_shift(grid,flux[i],rv=berv[i]*1000,kind='linear',fill_value=1)
    else:
        print('ALGO1')
        for i in tqdm(np.arange(len(files[-1]))):
            f = myc.tableXY(myf.doppler_r(grid,berv[i]*1000)[1],flux[i],0*grid)
            f.interpolate(new_grid=grid,method='linear',fill_value=1)
            flux[i] = f.y

    ccf_output = yarara_ccf(dir_root, files, 0, 6, 2.0, 'mask_telluric_o2', spectra=(grid,flux,err_flux), debug=False, wave_max=6800)
    fwhm_ins = ccf_output['fwhm'].y
    FWHM_ins = np.nanmedian(fwhm_ins)
    myv.vprint('\n [INFO] Instrumental resolution measured by O2 lines = %.1f km/s \n'%(FWHM_ins))
    calib = myc.tableXY([1, 2, 3, 4, 5, 6, 7, 8, 9],[299792,149896,99931,74948,59958,49965,42828,37474,33310])
    calib.interpolate(new_grid=np.array([FWHM_ins]),method='cubic',fill_value=np.nan)
    myv.vprint(' [INFO] Estimate intrumental resolution = %.0f'%(np.round(calib.y[0],-3)))

    fwhm_ins[fwhm_ins<1] = np.nan
    fwhm_ins[fwhm_ins>10] = np.nan

    return fwhm_ins, berv

mhk_c1 = -4.04840205e+01        #calibration with RHK DRS
mhk_c2 = 3927259.0994665725
def mhk_rhk(mhk):
    mhk = np.array(mhk)
    mhk[mhk<-40] = -40
    rhk = np.array(np.log10((mhk-mhk_c1)/mhk_c2))    
    return rhk

def rhk_mhk(rhk):
    rhk = np.array(rhk)
    mhk = mhk_c1 + mhk_c2 * 10**rhk
    return mhk

def yarara_activity_mhk(dir_root, files, rv_sys, shift_rv, teff, material, proxy, vsini=2.0, sub_dico='matching_diff'):
    
    if myv.VERBOSE:
        myf.print_box('\n---- RECIPE : NEW MHK EXTRACTION ----\n')

    jdb = get_jdb(files[-1],dir_root)
    photosphere = pd.read_pickle(MATERIAL_DIR+'/Photospheric_profiles_V.p')
    chromosphere = myf.touch_pickle(MATERIAL_DIR+'/Chromospheric_profiles_V.p')
    
    grid, flux, err_flux = import_sts(files, rv_shift=shift_rv, err=False, sub_dico=sub_dico)

    liste_proxy = [myv.Ca2K,myv.Ca2H]

    if vsini<5:
        vsini = 0.0
    else:
        vsini = np.sqrt(vsini**2-2**2)

    myv.vprint(' [INFO] Broadening Vsini = %.1f km/s x 75%%'%(vsini))
    vsini *= 0.75

    save = {}
    for n,l in enumerate(liste_proxy):
        unit_filling = {'CaIIK':0.14,'CaIIH':0.14*0.90}[l[-1]]
        deg_poly = {'CaIIK':0,'CaIIH':0,}[l[-1]]
        name = {'CaIIK':'CaII','CaIIH':'CaII'}[l[-1]]
        nb_line = {'CaII':2}[name]

        myv.vprint('\n [INFO] Analysis of %s(%s)'%(name,l[-1]))

        fig = plt.figure(name,figsize=(14,8))
        gs = fig.add_gridspec(3, nb_line)
        ax = []
        for i in range(3): 
            ax.append(fig.add_subplot(gs[i,n]))

        temp_correction = myc.tableXY(np.array(material['wave']),np.array(material['correction_factor'])*0+1) # Update now in create_sts and load_data
        temp_correction.clip(min=[l[0]-50,None],max=[l[0]+50,None])

        w1 = myf.doppler_r(l[0]-5*l[1],rv_sys*1000)[0]
        w2 = myf.doppler_r(l[0]+5*l[1],rv_sys*1000)[0]
        center = myf.doppler_r(l[0],rv_sys*1000)[0]
        
        i1 = myf.find_nearest(grid,w1)[0][0]
        i2 = myf.find_nearest(grid,w2)[0][0]
        line = flux[:,i1:i2]
        line_std = line*0 # flux_err[:,i1:i2]
        line_wave = grid[i1:i2]

        scale_temp = (teff/5775)**4 # not the local continuum but total photospheric energy
        scale_temp = myf.black_body_ratio(5775,teff,l[0]) #planck function ratio
        conversion_filling = 1/(unit_filling)*scale_temp
        wave_vel = 3e5*(line_wave-center)/center
        
        mask_activity = 1-material['activity_proxies']
        mask_activity = mask_activity[(material['wave']>=np.min(line_wave))&(material['wave']<=np.max(line_wave))]
        mask_activity = mask_activity.astype('bool')

        loc = myf.find_nearest(photosphere[l[-1]]['teff'],teff)[0][0]
        db = myc.tableXY(photosphere[l[-1]]['wave'],photosphere[l[-1]]['model'][loc])
        db.rotation_broadening(veq=vsini,replace=True)

        if False:
            models = []
            rv_grid = np.linspace(-4,4,20)
            for r in rv_grid:
                db.rv_shift(rv_sys+r,fill_value=np.nan,x_grid=line_wave,replace=False)
                models.append(db.shifted.copy().y)
            ratio = np.array(models)/np.median(line,axis=0)
            residuals = myf.mad(ratio,axis=1)
            rv_model = rv_sys + rv_grid[np.argmin(residuals)]*0.5
            print(' [INFO] Model RV shifted by %.2f km/s'%(rv_grid[np.argmin(residuals)]))
        else:
            rv_model = rv_sys

        db.rv_shift(rv_model,fill_value=np.nan,x_grid=line_wave,replace=False)

        quiet = db.shifted.copy()
        quiet.interpolated = quiet.copy()
        quiet.y_interp = quiet.y
        db = db.shifted
        db.y[db.x<(np.mean(db.x)-1.00)] = np.nan
        db.y[db.x>(np.mean(db.x)+1.00)] = np.nan

        db_E1 = chromosphere[l[-1]]
        loc = myf.find_nearest(db_E1['teff'],teff)[0][0]
        E1 = myc.tableXY(db_E1['vel'].copy(),db_E1['model'][loc].copy())
        E1.interpolate(new_grid=wave_vel,replace=True,method='linear')
        E1.y[np.abs(E1.x)>35] = 0
        E1.null()

        E1.x = E1.x/3e5*center+center
        E1.rotation_broadening(veq=vsini,replace=True)

        base_profile = np.array(E1.y)[:,np.newaxis]
        fmodel = 'amp'
        
        mask_z = abs(line-np.mean(line,axis=0))/myf.mad(line,axis=0)
        med_line = np.median(line,axis=0)
        if (np.shape(line)[0]>10):
            line[mask_z>10] = (med_line*np.ones(np.shape(line)[0])[:,np.newaxis])[mask_z>10] #

        temp_correction.interpolate(new_grid=line_wave,replace=False)
        line = line*temp_correction.y_interp
        line_std = line_std*temp_correction.y_interp

        i0 = myf.find_nearest(line_wave,myf.doppler_r(l[0]-l[1],rv_sys*1000)[0])[0][0]
        i1 = myf.find_nearest(line_wave,myf.doppler_r(l[0]+l[1],rv_sys*1000)[0])[0][0]

        if np.shape(line)[0]>10:
            ref = np.nanpercentile(line,50,axis=0)
            ref_std = np.nanstd(line,axis=0)
        else:
            ref = np.nanmean(line,axis=0)
            ref_std = 0*ref+0.01

        plt.axes(ax[0])
        mat = myc.table(line.copy())
        plt.title(l[-1],fontsize=16)

        quiet.interpolate(new_grid=line_wave,replace=False)
        calib = myc.tableXY(quiet.y_interp[mask_activity],ref[mask_activity])
        index_vec = np.arange(len(ref))[mask_activity]
        index_vec = index_vec[(~np.isnan(calib.x))&(~np.isnan(calib.y))]
        if len(index_vec)<5:
            return np.nan, np.nan, np.nan

        calib.yerr = calib.yerr*0+ref_std[mask_activity]
        calib.supress_nan()
        if len(calib.y)>3:
            for j in range(3):
                calib.fit_line(recenter=False)
                mask = myf.rm_outliers(calib.y-calib.x*calib.lin_slope_w,m=2,kind='mad')[0]
                if int(np.sum(mask)*100/len(mask))>60:
                    calib.masked(mask)
                    index_vec = index_vec[mask]

            index_vec = myf.in1d(np.arange(len(ref)),index_vec)
            if calib.lin_slope_w<0.2:
                print(Fore.YELLOW+' [WARNING] Flux scaling failed.'+Fore.RESET)
            else:
                mat.table-=calib.lin_intercept_w
                mat.table/=calib.lin_slope_w
                line_std/=calib.lin_slope_w
        mat.plot(x=wave_vel,cmap='seismic',color=proxy,new=False,alpha=0.07,fontsize=14)
        v1 = 3e5*(2.25)/center

        axlim = plt.gca()
        ymax = axlim.get_ylim()[1]
        plt.ylim(0,np.max([ymax,0.224]))
        plt.xlim(-v1,v1)
        plt.ylabel(r'$I(\lambda)$ []',fontsize=14)
        plt.axvline(x=0,ls='-',color='k',lw=1,alpha=0.1)

        def kms_to_wave(x):
            return x/3e5*center+center

        def wave_to_kms(x):
            return 3e5*(x-center)/center

        wcore_left = db.x[np.where(db.y==db.y)[0][0]]
        wcore_right = db.x[np.where(db.y==db.y)[0][-1]]

        wcore_left_v = 3e5*(wcore_left-center)/center
        wcore_right_v = 3e5*(wcore_right-center)/center

        error = 1#np.median(quiet_obs[mask_error]/db.y[mask_error])
        #print(' [INFO] Error in continuum detected at %.2f'%(error))
        db.y[~np.isnan(db.y)] = db.y[~np.isnan(db.y)]*error
        if len(files[-1])>5:
            db.y[np.isnan(db.y)] = np.median(mat.table,axis=0)[np.isnan(db.y)]
        else:
            db.y[np.isnan(db.y)] = quiet.y_interp[np.isnan(db.y)]
            db.y[np.isnan(db.y)] = np.median(mat.table,axis=0)[np.isnan(db.y)]
        
        plt.plot(3e5*(db.x-center)/center,db.y,color='C2',ls='-',lw=2,label=r'$I_{Q}$($\lambda$,%.0fK,%.0fkms)'%(teff,vsini))
        plt.legend(loc=1)
        plt.ylabel(r'$I(\lambda)$ []',fontsize=14)
        axlim = plt.gca()
        ymax = axlim.get_ylim()[1]
        plt.xlim(-v1,v1)
        plt.axvline(x=wcore_left_v,ls=':',color='k')
        plt.axvline(x=wcore_right_v,ls=':',color='k')
        plt.axhline(y=0,ls='-',color='C2',lw=2,zorder=10000)

        ax1 = plt.gca()
        ax2 = ax1.secondary_xaxis('top', functions=(kms_to_wave, wave_to_kms))
        ax2.set_xlabel(r"Wavelength [$\AA$]")

        plt.axes(ax[2])
        mat2 = myc.table(mat.table-db.y)
        wings_uncertainties = (line_wave<wcore_left)|(line_wave>wcore_right)
        offset = np.nanmedian(mat2.table[:,wings_uncertainties],axis=1)
        offset_std = myf.mad(mat2.table[:,wings_uncertainties],axis=1)
        mat2.table = mat2.table - offset[:,np.newaxis]

        if len(files[-1])>5:
            uncertainties = mat2.table-np.median(mat2.table,axis=0)
        else:
            uncertainties = mat2.table-quiet.y_interp*0

        if not np.sum(wings_uncertainties):
            wings_uncertainties = np.ones(len(line_wave)).astype('bool')
        index_extracted_std = myf.mad(uncertainties[:,wings_uncertainties],axis=1)*100
        med_err = np.nanmedian(index_extracted_std)

        flux_core = np.mean(db.y[int(len(db.y)/2)-10:int(len(db.y)/2)+10])
        flux_uncertainties = np.mean(db.y[wings_uncertainties])
        snr_core = flux_core/index_extracted_std*100
        snr_q1 = np.nanpercentile(snr_core,25)
        snr_q2 = np.nanpercentile(snr_core,50)
        snr_q3 = np.nanpercentile(snr_core,75)

        #print(med_err,flux_core,flux_uncertainties,snr_q2)

        noise_level = np.nanpercentile(mat2.table[:,wings_uncertainties],75,axis=1)*100

        med_precision = np.nanmedian(index_extracted_std)*conversion_filling/np.sqrt(np.sum(~wings_uncertainties))
        med_accuracy = np.nanmedian(index_extracted_std)*conversion_filling/np.sqrt(np.sum(wings_uncertainties))

        std_res = 0
        std_res2 = 0

        index_extracted_std[index_extracted_std==0] = 2*np.median(index_extracted_std[index_extracted_std!=0])
        
        std = index_extracted_std*np.ones(len(line_wave))[:,np.newaxis]/100
        weights = 1/std**2
        N_bootstrap=30

        s = base_profile[:,0]

        offset_std = []
        for w in weights.T:
            S_ss = np.sum(w * s * s,axis=0)
            S_s  = np.sum(w * s)
            S_1  = np.sum(w)
            stdA_linear = np.sqrt(1.0 / S_ss + (S_s**2 / S_ss**2) * (1/w[0]))
            offset_std.append(stdA_linear*100)
        offset_std = np.array(offset_std)

        mat3 = myc.table(mat2.table[:,~wings_uncertainties])
        mat3.fit_unique_base(base_profile.T[:,~wings_uncertainties],weights.T[:,~wings_uncertainties],perm=N_bootstrap) #do not increase because of RAM overflow
        res1 = mat3.vec_residues*100
        res2 = res1 - np.nanmedian(res1,axis=0)
        std_res = myf.mad(np.ravel(res1))
        std_res2 = myf.mad(np.ravel(res2))

        plt.figure('model%s'%(l[-1]),figsize=(18,5))
        plt.subplot(1,4,2)
        vmax = np.nanpercentile(mat3.vec_fitted,99)*100
        vmin = np.nanpercentile(mat3.vec_fitted,1)*100
        plt.imshow(mat3.vec_fitted*100,aspect='auto',vmin=vmin,vmax=vmax)
        plt.colorbar(pad=0)
        plt.subplot(1,4,1)
        plt.imshow(mat3.table*100,aspect='auto',vmin=vmin,vmax=vmax)
        plt.colorbar(pad=0)
        plt.subplot(1,4,3)
        plt.title('STD = %.2f %% '%(std_res))
        plt.imshow(res1,aspect='auto',vmin=-0.3,vmax=0.3)
        plt.colorbar(pad=0)
        plt.subplot(1,4,4)
        plt.title('STD = %.2f %% '%(std_res2))
        plt.imshow(res2,aspect='auto',vmin=-0.3,vmax=0.3)
        plt.colorbar(pad=0)
        plt.subplots_adjust(left=0.07,right=0.97)
        plt.savefig(dir_root+'IMAGES/Activity_profiles_%s_model_%s'%(l[-1],fmodel)+myv.PRD_EXT+'.png')
        plt.figure(name,figsize=(14,8))

        index_extracted = mat3.coeff_fitted[:,0]*100
        index_extracted_std = (mat3.coeff_fitted_std[:,0]*100 + offset_std)
        med_precision = np.nanmedian(index_extracted_std)*conversion_filling
        med_accuracy = np.nanmedian(index_extracted_std)*conversion_filling

        myv.vprint(' [INFO] Med flux uncertainties = %.2f'%(med_err)+'%')
        myv.vprint(' [INFO] Med filling uncertainties (precision) = %.2f'%(med_precision)+'%')
        myv.vprint(' [INFO] Med filling uncertainties (accuracy) = %.2f'%(med_accuracy)+'%')
        
        index_extracted_std = index_extracted_std*conversion_filling
        index_extracted = index_extracted*conversion_filling

        q1 = np.nanpercentile(index_extracted,10)
        q2 = np.nanmean(index_extracted)
        q3 = np.nanpercentile(index_extracted,90)
        q_std = med_accuracy

        plt.axhline(y=q2/100,color='k',alpha=0.1,lw=1)
        mat2.table = mat2.table*100*conversion_filling
        mat2.plot(x=wave_vel,cmap='seismic',color=proxy,new=False,alpha=0.07,fontsize=14)
        plt.axvline(x=wcore_left_v,ls=':',color='k')
        plt.axvline(x=wcore_right_v,ls=':',color='k')
        mean_activity = myc.tableXY(line_wave,np.nanmean(mat2.table,axis=0),0*line_wave)
        plt.plot(mean_activity.x,mean_activity.y,color='k',ls='-')
        plt.plot(wave_vel,base_profile[:,0]*np.nanpercentile(index_extracted,90),color='k',ls='-.')
        plt.plot(wave_vel,base_profile[:,0]*np.nanpercentile(index_extracted,10),color='k',ls='-.')

        #minimum = -5*med_err*conversion_filling
        minimum = np.min(np.percentile(mat2.table,5,axis=0))
        maximum = np.max(np.percentile(mat2.table,95,axis=0))
        ylim1 = minimum-(maximum-minimum)*0.1
        ylim2 = maximum+(maximum-minimum)*0.1
        plt.xlim(-v1,v1)
        plt.ylim(ylim1,ylim2)
        plt.xlabel(r'Wavelength $\lambda$ [km/s]',fontsize=14)
        plt.ylabel(r'$\tilde{\delta S} (\lambda) $ [%]',fontsize=14)
        plt.axhline(y=0,ls='-',color='C2',lw=2,zorder=10000)
        plt.axvline(x=center,ls='-',color='k',lw=1,alpha=0.1)
        plt.title(r'M-index = %.2f$\pm$%.2f (min - max = %.2f - %.2f)'%(q2,q_std,q1,q3))

        plt.axes(ax[1])
        mat2.table = mat2.table/conversion_filling
        mat2.plot(x=wave_vel,cmap='seismic',color=proxy,new=False,alpha=0.07,fontsize=14) #line_wave -> wave_vel
        plt.axvline(x=wcore_left_v,ls=':',color='k')
        plt.axvline(x=wcore_right_v,ls=':',color='k')
        plt.ylabel(r'$\delta S (\lambda) $ [%]',fontsize=14)
        plt.xlim(-v1,v1)
        plt.ylim(ylim1/conversion_filling,ylim2/conversion_filling)
        plt.errorbar(wcore_left_v-20,[0],yerr=[med_err*2],fmt='.',color='k',label=r'$S/N^{\gamma}_{core}$ = %.0f $\pm$ %.0f'%(snr_q2,0.5*(snr_q3-snr_q1)),zorder=100)
        plt.legend(loc=1)

        save[l[-1]] = {'flux':mat.table, 'flux_std':line_std, 'wave':line_wave, 'model':quiet.y_interp, #'flux_old':line,
        'index':index_extracted, 'index_std':index_extracted_std, 'std':std_res, 'snr_core':np.round(snr_q2,2),
        'index_p10':q1, 'index_p50':q2, 'index_p90':q3, 'fmodel':fmodel}
        
        plt.subplots_adjust(left=0.07,right=0.98,top=0.90,bottom=0.08,hspace=0.35)
        plt.savefig(dir_root+'IMAGES/Activity_profiles_%s_%s'%(name,fmodel)+myv.PRD_EXT+'.png')

    #CaII

    index1 =  save['CaIIK']['index']
    index1_std = save['CaIIK']['index_std']

    index2 =  save['CaIIH']['index']
    index2_std = save['CaIIH']['index_std']

    index = (index1*0.5+index2*0.5)/(0.5+0.5)
    index_std = np.sqrt(1/(1/index1_std**2+1/index2_std**2))

    myv.vprint('\n [INFO] M-index = %.2f +/- %.2f'%(np.median(index),np.median(index_std))+'%')

    save['CaII'] = {}
    save['CaII']['index'] = index
    save['CaII']['index_std'] = index_std
    save['CaII']['snr_core'] = 0.5*(save['CaIIK']['snr_core']+save['CaIIH']['snr_core'])*np.sqrt(2)

    mhk = [np.nanpercentile(save['CaII']['index'],i) for i in [16,50,86]]
    myv.vprint('\n [INFO] M-index = %.2f +/- %.2f [%.2f -> %.2f] \n'%(mhk[1],np.median(index_std),mhk[0],mhk[2]))

    dico = {'filename':files[-1]}
    dico['MHK'] = index
    dico['MHK_std'] = index_std

    MHK_RHK = mhk_rhk(dico['MHK'])
    MHK_RHK_std = abs(dico['MHK_std']/(np.log(10)*(mhk_c1-dico['MHK'])))
    dico['RHK'] = MHK_RHK
    dico['RHK_std'] = MHK_RHK_std

    dico = pd.DataFrame(dico)

    MHK_mean = np.nansum(index/index_std**2)/np.nansum(1/index_std**2)
    MHK_mean_std = np.sqrt(1/np.nansum(1/index_std**2))

    samples = MHK_mean_std*np.random.randn(99999) + MHK_mean
    samples_rhk = mhk_rhk(samples)

    RHK_mean = np.nanmean(samples_rhk)
    RHK_mean_std = np.nanstd(samples_rhk)

    plt.figure('MHK')
    plt.errorbar(jdb,index,yerr=index_std,fmt='ko')
    plt.title('Teff = %.0f K\n'%(teff)+r'<MHK> = %.1f $\pm$ %.1f %% | <RHK> = %.2f $\pm$ %.2f '%(MHK_mean, MHK_mean_std, RHK_mean, RHK_mean_std))
    plt.axhline(y=MHK_mean,color='r')
    plt.axhspan(ymin=MHK_mean-MHK_mean_std,ymax=MHK_mean+MHK_mean_std,color='r',alpha=0.4)
    plt.axhline(y=0,color='k',ls=':')
    plt.ylabel('M-index [%]')
    plt.xlabel('Time index')
    ax = plt.gca()
    y_ticks = ax.get_yticks()[1:-1]
    ylim = ax.get_ylim()
    ax.twinx()
    plt.ylim(ylim)
    plt.yticks(y_ticks,np.round(mhk_rhk(y_ticks),2))
    plt.ylabel(r'$\log$ $R_{HK}$ [dex]')
    plt.subplots_adjust(right=0.85) 
    plt.savefig(dir_root+'IMAGES/MHK_RHK'+myv.PRD_EXT+'.png')

    nb = int(99999/len(jdb))
    samples_mhk = []
    for i,j in zip(np.array(dico['MHK']),np.array(dico['MHK_std'])):
        samples_mhk.append(np.random.randn(nb)*j+i)
    samples_mhk = np.ravel(samples_mhk)
    samples_mhk = samples_mhk[samples_mhk>-40]
    #samples_mhk = samples_mhk[samples_mhk<300]

    #assuming no magnetic cycle and a single mu for all the Xi
    #Xi = np.array(dico['MHK'])
    #sigma = np.array(dico['MHK_std'])
    #samples_gpt = np.random.randn(99999,len(Xi))*sigma+Xi
    #samples_mhk2 = np.sum(samples_gpt / sigma**2, axis=1) / np.sum(1 / sigma**2)

    plt.figure('MHK_samples',figsize=(10,6))
    plt.subplot(2,1,1) ; plt.title('MHK = %.1f +/- %.1f %%'%(np.nanmedian(samples_mhk),myf.mad(samples_mhk)))
    pby,pbx = np.histogram(samples_mhk,bins=np.arange(-41,200,1),density=True)
    pbx = 0.5*(pbx[1:]+pbx[0:-1])
    plt.fill_between(pbx,pby,alpha=0.2,color='C0')
    plt.plot(pbx,pby,color='k')

    plt.xlabel('MHK [%]',fontsize=13)
    plt.ylim(0,None)
    plt.xlim(-40,200)
    plt.axvline(x=0,ls=':',color='k')
    plt.axvline(x=50,ls='-.',color='k',label='active stars',lw=1)

    samples_rhk = []
    for i,j in zip(np.array(dico['RHK']),np.array(dico['RHK_std'])):
        samples_rhk.append(np.random.randn(nb)*j+i)
    samples_rhk = np.ravel(samples_rhk)
    samples_rhk = samples_rhk[samples_rhk>-6]
    #samples_rhk = samples_rhk[samples_rhk<-4]

    plt.subplot(2,1,2) ; plt.title('RHK = %.2f +/- %.2f dex'%(np.nanmedian(samples_rhk),myf.mad(samples_rhk)))
    pby,pbx = np.histogram(samples_rhk,bins=np.arange(-6.0,-4.0,0.02),density=True)
    pbx = 0.5*(pbx[1:]+pbx[0:-1])
    plt.fill_between(pbx,pby,alpha=0.2,color='C1')
    plt.plot(pbx,pby,color='k')
    plt.xlabel(r'$\log$ $R_{HK}$ [dex]',fontsize=13)
    plt.axvline(x=-5.0,ls=':',color='k')
    plt.axvline(x=-4.65,ls='-.',color='k',label='active stars',lw=1)
    plt.ylim(0,None)
    plt.xlim(-6,-4)
    plt.subplots_adjust(hspace=0.35)

    plt.savefig(dir_root+'IMAGES/MHK_samples'+myv.PRD_EXT+'.png')
    
    if len(samples_mhk)&len(samples_rhk):
        samples_table = pd.read_csv(dir_root+'WORKSPACE/Analyse_samples.csv.gz')
        samples_table['mhk'] = np.random.choice(samples_mhk,99999)
        samples_table['rhk'] = np.random.choice(samples_rhk,99999)
        samples_table.to_csv(dir_root+'WORKSPACE/Analyse_samples.csv.gz',index=False)

    return dico, RHK_mean, MHK_mean

def create_finch_db(dir_root,sub_dico='matching_diff'):

    files = [dir_root+'WORKSPACE/Analyse_summary.csv']

    infos = []
    for f2 in tqdm(files):
        summary = pd.read_csv(f2,index_col=0)
        if 'MHK' in summary.keys():
            f = glob.glob(f2.replace('WORKSPACE','STAR_INFO/Stellar_info*&&').split('&&')[0])[0]
            info = pd.read_pickle(f)
            star = f.split('/data')[0].split('/')[-1]
            instrument = f.split('/STAR_INFO')[0].split('/')[-1]
            spectro = instrument.split('_')[0]
            drs = instrument.split('_')[1]
            pipeline = 'SNAKY'
            processing = {'matching_diff':'YV0','matching_instrument':'YVA','matching_mad':'YV1'}[sub_dico] 
            code = star+'_'+spectro+'_'+drs+'_'+pipeline
            
            teff = np.round(myf.get_info_lvl2(info,'Teff',pipeline.upper()),0)
            logg = np.round(myf.get_info_lvl2(info,'Log_g',pipeline.upper()),2)
            feh = np.round(myf.get_info_lvl2(info,'FeH',pipeline.upper()),2)
            rv_sys = np.round(myf.get_info_lvl2(info,'Rv_sys',pipeline.upper()),2)

            summary['star'] = star
            summary['ins'] = instrument
            summary['source'] = pipeline.upper()
            summary['yvx'] = processing
            summary['finch_offset'] = 0.0
            summary['smw'] = 0.0
            summary['teff'] = teff
            summary['logg'] = logg
            summary['feh'] = feh
            summary['rv_sys'] = rv_sys
            summary['flag'] = summary['flag2']
            summary = summary.rename(columns={'RHK':'rhk','RHK_std':'rhk_std','MHK':'mhk','MHK_std':'mhk_std'})
            summary['mhk_cleaned'] = 0.0 ; summary['mhk_cleaned_std'] = 0.0
            summary = summary[['star','jdb','mhk','mhk_std','mhk_cleaned','mhk_cleaned_std','rhk','rhk_std','ins','source','yvx','flag','finch_offset','smw','teff','logg','feh','rv_sys']]
            summary['mhk'] = np.round(summary['mhk'],3)
            summary['mhk_std'] = np.round(summary['mhk_std'],3)
            summary['rhk'] = np.round(summary['rhk'],4)
            summary['rhk_std'] = np.round(summary['rhk_std'],4)
            summary.to_csv(f2.replace('summary.csv','Finch_table.csv'))
