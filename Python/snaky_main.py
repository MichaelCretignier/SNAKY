import getopt
import datetime
import snaky_variables as myv
import snaky_functions as myf
import snaky_classes as myc
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

import my_classes as myyc

try:
    import Finch as Finch
    installation = 'complete'
    print('[IMPORT] FINCH module is found')
except:
    installation = 'partial'
    print(Fore.YELLOW+'[IMPORT] FINCH module is missing'+Fore.RESET)

"""

SNAKY — Spectroscopic Novel Analysis Kit of Yarara

"""

__version__ = '0.5.0'

print(Fore.GREEN+"""\n[INFO SNAKY]
[INFO USER] SNAKY version = """+__version__ +""" 
[INFO USER] READ ME CAREFULLY 
[INFO USER] Vsini is still in validation and can't be used except for solar analogs ([5600 - 5800])
[INFO USER] Continuum normalisation made by RASSINE explained in Cretignier et al. 2020b
[INFO USER] Atmospheric parameters were explained in Cretignier et al. 2024a
[INFO USER] The MHK activity index was explained in Cretignier et al. 2024b
[INFO USER] An issue or an upgrade? Contact me at:  michael.cretignier@physics.ox.ac.uk
      """+Fore.RESET)

cwd = os.getcwd()
root = '/'.join(cwd.split('/')[:-1])

#INTERMEDIATE FUNCTIONS

def get_berv(ra_deg, dec_deg, obstime_utc, instrument):
    ins = instrument.split('_')[0][0:6].split('-')[0]
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

def create_snaky_dir(star,ins):
    
    if not os.path.exists(root+'/Snaky/'+star+'/data/s1d/ALLINS_MERGED'):
        os.makedirs(root+'/Snaky/'+star+'/data/s1d/ALLINS_MERGED', exist_ok=True)

    if os.path.exists(root+'/Snaky/'+star+'/data/s1d/'+ins+'/WORKSPACE'):
        print(' [INFO] SNAKY directory found!\n')
        return star, ins
    else:
        if len(star.split(' '))>1:
            print(Fore.YELLOW + '\n [WARNING] The specified star (%s) contains spaces'%(star)+Fore.RESET)
            star = star.replace(' ','')
            print(Fore.YELLOW + ' [WARNING] Spaces have been removed, new starname: %s'%(star)+Fore.RESET)

        if len(ins.split('_'))!=2:
            print(Fore.YELLOW + '\n [WARNING] The specified instrument (%s) is wrong'%(ins)+Fore.RESET)
            print(Fore.YELLOW + ' [WARNING] The format should follow: SPECTRO_DRS (ESPRES'
            '_3.3.1, HARPS_3.5)'+Fore.RESET)
        if len(ins.split('_'))==1:
            ins = ins+'_1.0'
            print(Fore.YELLOW + ' [WARNING] The instrument DRS version was set to 1.0 (%s)'%(ins)+Fore.RESET)

        directories = ['RAW','IMAGES','WORKSPACE','EXPORT','CCF_MASK','DACE_TABLE','DETECTION_LIMIT','FILM','KEPLERIAN','KITCAT','MASTER','PCA','REDUCTION_INFO','TEMP','STAR_INFO','WARNING']
        for d in directories:
            base = root+'/Snaky/'+star+'/data/s1d/'+ins+'/'+d
            os.makedirs(base, exist_ok=True)
        print(Fore.GREEN +'\n [SNAKY ADVICE] Put the %s spectra you want to process in:'%(ins)+Fore.RESET)
        print(Fore.GREEN +' [SNAKY ADVICE] '+root+'/Snaky/'+star+'/data/s1d/'+ins+'/RAW'+Fore.RESET)
        return star, ins

def clean_light_dir(dir_root):
    os.system('rm '+dir_root+'CCF_MASK/CCF*.fits')
    material = import_material(dir_root)
    del material['activity_proxies']
    del material['stellar_template']
    pickle.dump(material,open(dir_root+'WORKSPACE/Analyse_material.p','wb'))

def fix_dir_root(instrument='SOPHIE-HE_0.5'):
    a = glob.glob(root+'/Snaky/*/data/s1d/*/WORKSPACE/Analyse_summary.csv')
    for file in a:
        star = file.split('/data')[0].split('/')[-1]
        ins = file.split('/WORKSPACE')[0].split('/')[-1]
        summary = pd.read_csv(file,index_col=0)
        dace_file = file.replace('WORKSPACE/Analyse_summary.csv','DACE_TABLE/Dace_extracted_table.csv')
        dace = pd.read_csv(dace_file,index_col=0)
        mask = (np.array(summary['ins'])==instrument)

        if (sum(mask)!=0)&(ins!=instrument):
            print(Fore.YELLOW+' [WARNING] Some spectra with the wrong instrument found for %s'%(star)+Fore.RESET)
            create_snaky_dir(star,instrument)
            
            all_files = []
            for f in np.array(summary.loc[mask,'filename']):
                b = pd.read_pickle(f)
                arc = b['parameters']['arcfiles']
                for ar in arc:
                    os.system('mv '+ar+' '+ar.replace('/'+ins+'/','/'+instrument+'/'))
                    all_files.append(ar)
                arc2 = [n.replace('/'+ins+'/','/'+instrument+'/') for n in arc]
                b['parameters']['arcfiles'] = arc2
                pickle.dump(b,open(f.replace('/'+ins+'/','/'+instrument+'/'),'wb'))

            mask_dace = np.in1d(dace['fileroot'],np.array(all_files))

            new_summary = summary.loc[mask].reset_index(drop=True)
            new_summary.to_csv(file.replace('/'+ins+'/','/'+instrument+'/'))
            new_dace = dace.loc[mask_dace].reset_index(drop=True)
            new_dace['ins'] = instrument
            new_dace['fileroot'] = np.array([n.replace('/'+ins+'/','/'+instrument+'/') for n in new_dace['fileroot']])
            new_dace.to_csv(dace_file.replace('/'+ins+'/','/'+instrument+'/'))

            old_summary = summary.loc[~mask].reset_index(drop=True)
            old_summary.to_csv(file)
            old_dace = dace.loc[~mask_dace].reset_index(drop=True)
            old_dace.to_csv(dace_file)

            rassine_files = glob.glob(file.replace('Analyse_summary.csv','RASSINE*.p'))
            for k in np.setdiff1d(rassine_files,np.array(old_summary['filename'])):
                os.system('rm '+k)
            
            if np.sum(~mask_dace)==0:
                print(star)
                os.system('rm -rf '+file.split('/WORKSPACE')[0])


def create_finch_db(dir_root=None):

    if dir_root is None:
        files = np.sort(glob.glob(root+'/Snaky/*/data/s1d/*/WORKSPACE/Analyse_summary.csv'))
    else:
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
            pipeline = f.split('/'+star)[0].split('/')[-1]
            processing = 'YV0' #TBD add the info if matching_diff or matching_mad
            code = star+'_'+spectro+'_'+drs+'_'+pipeline
            
            teff = np.round(myf.get_info_lvl2(info,'Teff',pipeline.upper()),0)
            logg = np.round(myf.get_info_lvl2(info,'Log_g',pipeline.upper()),2)
            feh = np.round(myf.get_info_lvl2(info,'FeH',pipeline.upper()),2)
            rv_sys = np.round(myf.get_info_lvl2(info,'Rv_sys',pipeline.upper()),2)

            summary['star'] = star
            summary['ins'] = instrument
            summary['source'] = pipeline.upper()
            summary['finch_offset'] = 0.0
            summary['smw'] = 0.0
            summary['teff'] = teff
            summary['logg'] = logg
            summary['feh'] = feh
            summary['rv_sys'] = rv_sys
            summary['flag'] = summary['flag2']
            summary = summary.rename(columns={'RHK':'rhk','RHK_std':'rhk_std','MHK':'mhk','MHK_std':'mhk_std'})
            summary['mhk_cleaned'] = 0.0 ; summary['mhk_cleaned_std'] = 0.0
            summary = summary[['star','jdb','mhk','mhk_std','mhk_cleaned','mhk_cleaned_std','rhk','rhk_std','ins','source','flag','finch_offset','smw','teff','logg','feh','rv_sys']]
            summary['mhk'] = np.round(summary['mhk'],3)
            summary['mhk_std'] = np.round(summary['mhk_std'],3)
            summary['rhk'] = np.round(summary['rhk'],4)
            summary['rhk_std'] = np.round(summary['rhk_std'],4)
            summary.to_csv(f2.replace('summary.csv','Finch_table.csv'))

def create_snaky_db(filename='All_stars', stars=['*'], branch='Snaky'):
    os.makedirs(root+'/Snaky/database', exist_ok=True)
    
    old_table = [] ; old_ccf = [] ; old_spec = [] ; old_finch = []
    if os.path.exists(root+'/Snaky/database/'+filename+'_summary_infos.csv'):
        old_table = pd.read_csv(root+'/Snaky/database/'+filename+'_summary_infos.csv',index_col=0)
        old_ccf = np.load(root+'/Snaky/database/'+filename+'_ccf.npy')
        old_spec = np.load(root+'/Snaky/database/'+filename+'_spec.npy')
        #old_finch = pd.read_csv(root+'/Snaky/database/'+filename+'_Finch_table.csv',index_col=0)
    
    files = []
    if stars is None:
        #files = np.sort(glob.glob(root+'/'+branch+'/*/data/s1d/*/WORKSPACE/Analyse_spectroscopy.p'))
        files = np.sort(glob.glob(root+'/'+branch+'/*/data/s1d/*/STAR_INFO/Stellar_info*.p'))
    else:
        for s in stars:
            #files.append(glob.glob(root+'/'+branch+'/'+s+'/data/s1d/*/WORKSPACE/Analyse_spectroscopy.p'))
            files.append(glob.glob(root+'/'+branch+'/'+s+'/data/s1d/*/STAR_INFO/Stellar_info*.p'))
        files = np.sort(np.hstack(files))
    files = np.sort(np.unique(files))

    qc = np.array([len(f.split('/STAR_INFO')[0].split('/')[-1].split('_')) for f in files])
    files = files[qc==2]

    infos = []
    for f in tqdm(files):
        info = pd.read_pickle(f)
        star = f.split('/data')[0].split('/')[-1]
        instrument = f.split('/STAR_INFO')[0].split('/')[-1]
        spectro = instrument.split('_')[0]
        drs = instrument.split('_')[1]
        pipeline = f.split('/'+star)[0].split('/')[-1]
        processing = 'YV0' #TBD add the info if matching_diff or matching_mad
        code = star+'_'+spectro+'_'+drs+'_'+pipeline
        
        teff = np.round(myf.get_info_lvl2(info,'Teff',pipeline.upper()),0)
        logg = np.round(myf.get_info_lvl2(info,'Log_g',pipeline.upper()),2)
        feh = np.round(myf.get_info_lvl2(info,'FeH',pipeline.upper()),2)
        rhk = np.round(myf.get_info_lvl2(info,'RHK',pipeline.upper()),2)
        mhk = np.round(myf.get_info_lvl2(info,'MHK',pipeline.upper()),1)
        rv_sys = np.round(myf.get_info_lvl2(info,'Rv_sys',pipeline.upper()),2)

        fwhm_ins = np.round(myf.get_info_lvl2(info,'FWHM','O2'),2)
        fwhm_ccf1 = np.round(myf.get_info_lvl2(info,'FWHM','G2'),2)
        fwhm_ccf2 = np.round(myf.get_info_lvl2(info,'FWHM','GARFIELD'),2)
        fwhm_ccf3 = np.round(myf.get_info_lvl2(info,'FWHM','KITTY'),2)
        vsini = np.round(myf.get_info_lvl2(info,'Vsini',pipeline.upper()),2)

        infos.append([code,star,spectro,drs,pipeline,processing,teff,logg,feh,mhk,rhk,fwhm_ins,fwhm_ccf1,fwhm_ccf2,fwhm_ccf3,vsini])

    infos = pd.DataFrame(infos,columns=['code','star','ins','drs','pipeline','yvx','teff','logg','feh','mhk','rhk','fwhm_ins','fwhm_g2','fwhm_garfield','fwhm_kitty','vsini'])

    spectro_exist = []
    for f in tqdm(files):
        f2 = f.replace('STAR_INFO','WORKSPACE/Analyse_spectro*&&').split('&&')[0]
        f2 = glob.glob(f2)
        spectro_exist.append(0)

        if len(f2):
            spec = pd.read_pickle(f2[0])
            if len(spec):
                if 'flux_corrected' in spec.keys():
                    spectro_exist[-1] = 1

    spectro_exist = np.array(spectro_exist)
    infos['IF_SPEC'] = spectro_exist

    ccf_exist = []
    for f in tqdm(files):
        f2 = f.replace('STAR_INFO','WORKSPACE/Analyse_ccf_saved*&&').split('&&')[0]
        ccf = glob.glob(f2)
        ccf_exist.append(0)
        if len(ccf):
            ccf = pd.read_pickle(ccf[0])
            if 'CCF_G2' in ccf.keys():
                ccf_exist[-1] = 1
    ccf_exist = np.array(ccf_exist)
    infos['IF_CCF'] = ccf_exist

    vgrid = np.arange(0,150000,538)
    vgrid = np.hstack([-vgrid[::-1],vgrid[1:]])
    all_ccf = []
    for n in tqdm(np.arange(len(files))):
        f = files[n]
        f2 = f.replace('STAR_INFO','WORKSPACE/Analyse_ccf_saved*&&').split('&&')[0]
        flag = infos.loc[n,'IF_CCF']
        if flag==0:
            all_ccf.append(np.zeros(len(vgrid)))
        else:
            ccf = pd.read_pickle(glob.glob(f2)[0])['CCF_G2']
            try:
                master_ccf = np.nanmean(ccf['ccf_shifted'],axis=1)
                vrad = ccf['ccf_vrad']
            except: #yarara different file structure
                master_ccf = np.nanmean(ccf['matching_diff']['ccf_flux'],axis=1)
                master_ccf /= np.nanmax(master_ccf)
                vrad = ccf['matching_diff']['ccf_vrad']
            ccf = myc.tableXY(vrad,master_ccf,0*master_ccf)
            ccf.y[ccf.y>1.1] = np.nan
            ccf.y[ccf.y<0.0] = np.nan
            ccf.supress_nan()
            ccf.interpolate(new_grid=vgrid,fill_value=1)
            all_ccf.append(ccf.y)
    all_ccf = np.array(all_ccf)

    wgrid = np.round(np.arange(3900,6800,0.01),2)
    all_spec = []
    for n in tqdm(np.arange(len(files))):
        f = files[n]
        f2 = f.replace('STAR_INFO','WORKSPACE/Analyse_spectro*&&').split('&&')[0]
        flag = infos.loc[n,'IF_SPEC']
        if flag==0:
            all_spec.append(np.zeros(len(wgrid)))
        else:
            spec = pd.read_pickle(glob.glob(f2)[0])
            flux = spec['flux_corrected']
            wave = spec['wave']
            spec = myc.tableXY(wave,flux,wave)
            spec.interpolate(new_grid=wgrid,fill_value=np.nan,method='linear')
            all_spec.append(spec.y)
    all_spec = (np.array(all_spec)*1e4).astype('int16')
    
    print('\n')
    if len(old_table):
        mask = ~np.in1d(np.array(old_table['code']),np.array(infos['code']))
        mask2 = ~np.in1d(np.array(infos['code']),np.array(old_table['code']))
        print(' [INFO] %.0f datasets modified'%(np.sum(~mask)))
        print(' [INFO] %.0f new datasets added'%(np.sum(mask2)))
        if sum(mask):
            infos = pd.concat([old_table.loc[mask],infos],axis=0)
            all_spec = np.vstack([old_spec[mask],all_spec])
            all_ccf = np.vstack([old_ccf[mask],all_ccf])
    else:
        print(' [INFO] %.0f new datasets added'%(len(infos)))

    print(' [INFO] Nb unique stars = %.0f'%(len(np.unique(infos['star']))))

    infos = infos.sort_values(by='code')
    all_spec = all_spec[infos.index]
    all_ccf = all_ccf[infos.index]
    infos = infos.reset_index(drop=True)

    print(infos)

    infos.to_csv(root+'/Snaky/database/'+filename+'_summary_infos.csv')
    np.save(root+'/Snaky/database/'+filename+'_ccf.npy',all_ccf)
    np.save(root+'/Snaky/database/'+filename+'_spec.npy',all_spec)


def plot_starinfo(branch='Snaky', ins='*', xvar='Teff_SNAKY', yvar='FWHM_G2', cvar=None, svar=None):
    all_files = np.sort(glob.glob(root+'/'+branch+'/*/data/s1d/'+ins+'/STAR_INFO/Stellar_info*.p'))
    output = []
    for f in all_files:
        tab = pd.read_pickle(f)
        try:
            x = tab[xvar.split('_')[0]][xvar.split('_')[1]]
            y = tab[yvar.split('_')[0]][yvar.split('_')[1]]
        except:
            x = np.nan ; y = np.nan
        output.append([tab['Name'],x,y])
    output = pd.DataFrame(output,columns=['starname','x','y'])
    plt.scatter(output['x'],output['y'])
    
def plot_fwhm(dir_root, ccf_mask='mask_telluric_o2', xvar='jdb', alpha=0.4, color='k', branch='Snaky'):
    all_files = glob.glob(dir_root+'WORKSPACE/Analyse_ccf.p')
    var = []
    for f in all_files:
        print(f)
        f2 = f.replace('/WORKSPACE/Analyse_ccf.p','/STAR_INFO/Ste*')
        if xvar!='jdb':
            v = pd.read_pickle(glob.glob(f2)[0])[xvar][branch.upper()]
        else:
            v = 0
        var.append(v)

    for v,f in zip(var,all_files):
        tab = pd.read_pickle(f)
        try:
            ccf = tab['CCF_'+ccf_mask]['table']
            if xvar=='jdb':
                x = ccf['jdb']
            else:
                x = np.ones(len(ccf['fwhm']))*v
            plt.scatter(x, ccf['fwhm'], color=color, alpha=alpha)
            plt.ylabel('FWHM [km/s]')
        except:
            pass

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

def plot_mhk(dir_root, hide_outliers=True, daily_binned=True, debug=False):
    directory = '/'.join(dir_root.split('/')[0:-2])
    ins = dir_root.split('/')[-2]
    summaries = glob.glob(directory+'/*/WORKSPACE/Analyse_summary.csv')
    plt.figure(figsize=(15,6))
    count=-1
    for s in summaries:
        count+=1
        instrument = s.split('/WORKSPACE')[0].split('/')[-1]
        if debug:
            star_info = glob.glob(s.split('/WORKSPACE')[0]+'/STAR_INFO/Stellar*')[0]
            teff = np.round(pd.read_pickle(star_info)['Teff']['SNAKY'],-1)
            instrument = instrument+'(%.0f)'%(teff)
        tab = pd.read_csv(s,index_col=0)
        if ('jdb' in tab.keys())&('MHK' in tab.keys()):
            proxy = myc.tableXY(np.array(tab['jdb']),np.array(tab['MHK']),np.array(tab['MHK_std']))
            if daily_binned:
                proxy.night_stack(replace=True)
            if hide_outliers:
                valid = proxy.yerr<20
            else:
                valid = proxy.yerr>0
            plt.errorbar(proxy.x[valid],proxy.y[valid],yerr=proxy.yerr[valid],label=instrument,marker=['o','s','^'][int(count//10)],ls='',capsize=0,color='C%.0f'%(count),mec='k')
            plt.scatter(proxy.x[~valid],proxy.y[~valid],marker='x',color='C%.0f'%(count))

    plt.legend()
    plt.ylabel('M-index [%]',fontsize=14)
    plt.xlabel('Jdb - 2,400,000 [days]',fontsize=14)

    solar_cycle = pd.read_csv(root+'/Python/Material_snaky/Solar_Mg2.csv',index_col=0)
    sun_mag = myc.tableXY(solar_cycle['jdb'],solar_cycle['plage_fill'],0*solar_cycle['jdb'])
    sun_mag.smooth(box_pts=100,shape='savgol')
    plt.plot(sun_mag.x,sun_mag.y,color='gold',lw=1,alpha=0.7)
    plt.fill_between(sun_mag.x,0,sun_mag.y,color='gold',alpha=0.25)

    ax = plt.gca()
    x_ticks = ax.get_xticks()[1:-1]
    xlim = ax.get_xlim()
    y_ticks = ax.get_yticks()[1:-1]
    ylim = ax.get_ylim()
    ax.twiny()
    plt.xlim(xlim)
    plt.xticks(x_ticks,np.round(2000+(x_ticks-51544.5)/365,1))
    plt.xlabel('Date [year]',fontsize=14)
    ax.twinx()
    plt.ylim(ylim)
    plt.yticks(y_ticks,np.round(mhk_rhk(y_ticks),2))
    plt.ylabel(r'$\log$ $R_{HK}$ [dex]',fontsize=14)
    plt.savefig(dir_root+'IMAGES/MHK.png')
    plt.savefig(dir_root.replace(ins,'ALLINS_MERGED')+'MHK.png')


def yarara_finch(dir_root, branch='Snaky', proxy_name='MHK',ext='',trend_degree=0, harm=0, offset_instrument='no!', automatic_fit=False, x_unit='years',predict='today', print_reference=True, rm_source=['DACE']):

    myf.print_box('\n---- RECIPE : FINCH MAGNETIC CYCLE PERIOD ----\n')

    starname = dir_root.split('Snaky/')[-1].split('/')[0]
    ins = dir_root.split('/')[-2]
    star_info = import_star_info(dir_root)

    x=[] ; y=[] ; yerr=[] ; instrument = [] ; reference = [] ; flag = []

    files = glob.glob(dir_root.replace(ins,'*').replace('Snaky',branch)+'WORKSPACE/Analyse_Finch_table.csv')
    for file in files:
        table = pd.read_csv(file,index_col=0)
        yerr.append(np.array(table[proxy_name.lower()+'_std']))
        y.append(np.array(table[proxy_name.lower()]))
        x.append(np.array(table['jdb']))
        instrument.append(np.array(table['ins']))   
        reference.append(np.array(table['source']))   
        flag.append(np.array(table['flag']))   

    folder = dir_root.split('/Snaky')[0]
    files = glob.glob(folder+'/Python/Material_snaky/Activity_MHK_*.csv')
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
        rm_source = rm_source)
    
    if db_finch is not None:
        if proxy_name.split('_')[0]=='MHK':
            db_finch.convert_smw_mhk(int(star_info['Teff']['SNAKY']))
            
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

    vec = Finch.tableXY(proxy.x, proxy.y, proxy.yerr, proxy_name = proxy_name) 
    vec.set_instrument(instru)
    vec.set_reference(reference)
    vec.set_ins_uncertainties(null_yerr=True)
    vec.set_flag(flag)

    vec.set_star(
        starname = starname, 
        teff = star_info['Teff']['SNAKY'],
        logg = star_info['Log_g']['SNAKY'],
        feh = star_info['FeH']['SNAKY'],
        )
    if not print_reference:
        vec.print_reference = False

    vec.mask_flag[vec.yerr>20] = True

    #self.debug = vec,trend_degree,harm,automatic_fit,automatic_fit,offset_instrument,predict,x_unit

    if ((np.max(vec.x)-np.min(vec.x))/365.25)>4: #at least 4 years baseline to fit

        vec.fit_period_cycle(
            trend_degree = trend_degree, 
            harm=harm,
            automatic_fit = automatic_fit, 
            data_driven_std = True, 
            offset_instrument = offset_instrument, 
            predict = 'today',
            x_unit = x_unit)
        
        plt.savefig(dir_root.replace(ins,'ALLINS_MERGED')+'Finch_magnetic_cycle'+ext+'.png')
        if not vec.out_convergence_flag:
            vec.out_pmag = 0.00

        for i in np.unique(vec.out_model_offset.instrument):
            value_offset = np.median(vec.out_model_offset.y[vec.out_model_offset.instrument==i])
            vec.y[vec.instrument==i] -= value_offset
            vec.bin.y[vec.bin.instrument==i] -= value_offset
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

    solar_cycle = pd.read_csv(root+'/Python/Material_snaky/Solar_Mg2.csv',index_col=0)
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

    plt.savefig(dir_root.replace(ins,'ALLINS_MERGED')+'Finch_magnetic_cycle_GP'+ext+'.png')

    fig_gp.set_figwidth(10)
    plt.title('Teff = %.0f K   |    Logg = %.2f dex   |    Fe/H = %.2f dex   |    Pmag = %.2f years   |    < M > = %.1f %% (A = %.1f %%)    '%(vec.star_teff, vec.star_logg, vec.star_feh, vec.out_gp_pmag, vec.out_gp_meanmag, vec.out_gp_ampmag))
    plt.xlim(1965,2040)
    plt.ylim(-15,90)
    plt.legend()
    plt.axhline(y=0,color='k',ls='-',alpha=0.7,lw=1)
    plt.savefig(dir_root.replace(ins,'ALLINS_MERGED')+'Finch_magnetic_cycle_GP_fixed_axis'+ext+'.png')

    if not vec.out_convergence_flag:
        vec.bin.fit_line()
        trend = myc.tableXY(myf.conv_time(vec.bin.x)[1]-2000,vec.bin.y,vec.bin.yerr)
        trend.fit_line(recenter=False)
        model = (vec.out_gp_model[0]-2000)*trend.lin_slope_w + trend.lin_intercept_w
        model[vec.out_gp_model[0]<=(np.min(trend.x)+2000)] = vec.out_gp_model[1][vec.out_gp_model[0]<=(np.min(trend.x)+2000)]
        model[vec.out_gp_model[0]>=(np.max(trend.x)+2000)] = vec.out_gp_model[1][vec.out_gp_model[0]>=(np.max(trend.x)+2000)]
        vec.out_gp_model[1] = model
    
    exportation3 = myc.tableXY(myf.conv_time(vec.out_gp_model[0])[0],vec.out_gp_model[1],vec.out_gp_model[2])
    exportation3.export(dir_root.replace(ins,'ALLINS_MERGED')+'Finch_%s_GP_model.csv'%(proxy_name),format='csv',columns=['jdb','proxy','proxy_std','qc'])

    output = [
        FINCH_Pmag,
        FINCH_Pmag_GP,
        FINCH_Mmag_GP,
        FINCH_Kmag_GP]+vec.out_gp_predict
    
    return output


def import_spectrum(file,sub_dico='matching_diff'):
    file = pd.read_pickle(file)
    spec = myc.tableXY(file['wave'],file['flux']/file[sub_dico]['continuum_linear'],0*file['wave'])
    return spec

def master_spectrum(files, rv_shift, rv_sys, plot=False, sub_dico='matching_diff'):
    wave_grid, sts, sts_err = import_sts(files, sub_dico=sub_dico)
    rv_syst = rv_sys*1000
    shift_ms = rv_shift + rv_syst
        
    for m,rv in enumerate(shift_ms):
        spec = myc.tableXY(myf.doppler_r(wave_grid,rv)[1], sts[m],0*wave_grid)
        spec.interpolate(new_grid=wave_grid,fill_value=0,method='linear')
        sts[m] = spec.y
    master = np.nanmedian(sts,axis=0)
    master = myc.tableXY(wave_grid,master,0*wave_grid)

    if plot:
        plt.figure('master')
        master.plot()
    
    return master

def import_sts(files, rv_shift=None, err=False, sub_dico='matching_diff'):
    "rv_shift in m/s"
    wmin = []
    wmax = []
    for f in files:
        spec = import_spectrum(f, sub_dico=sub_dico)
        wmin.append(np.nanmin(spec.x))
        wmax.append(np.nanmax(spec.x))
    wmin = np.round(np.nanmax(wmin)+10,2)
    wmax = np.round(np.nanmin(wmax)-10,2)

    wave_grid = np.round(np.arange(wmin,wmax,0.01),4)
    if rv_shift is None:
        rv_shift = np.zeros(len(files))

    sts = []
    sts_err = []
    for f, rv in zip(files,rv_shift):
        spec = import_spectrum(f, sub_dico=sub_dico)
        if rv!=0:
            spec.x = myf.doppler_r(spec.x,rv)[1]
        spec.interpolate(new_grid=wave_grid,method='linear',replace=True)
        sts.append(spec.y)
        if err:
            sts_err.append(spec.yerr)
    sts = np.array(sts)
    sts_err = np.array(sts_err)
    if len(sts_err)==0:
        sts_err = None

    return wave_grid, sts, sts_err    

def read_neid(file,force=False,debug=False):
    outname = file.replace('RAW','WORKSPACE').replace('.fits','.p').replace('WORKSPACE/','WORKSPACE/RASSINE_Stacked_spectrum_B0.00_')
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

def read_espresso(file,force=False,debug=False):
    outname = file.replace('RAW','WORKSPACE').replace('.fits','.p').replace('WORKSPACE/','WORKSPACE/RASSINE_Stacked_spectrum_B0.00_')
    if (not os.path.exists(outname))|(force):
        t = fits.open(file)
        wave = t[1].data['wavelength_air']
        flux = t[1].data['flux']
        flux_std = t[1].data['error']
        wave_grid = np.arange(np.min(wave),np.max(wave),0.01)
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
    

def read_sophie(file,force=False,debug=False):
    outname = file.replace('RAW','WORKSPACE').replace('.fits','.p').replace('WORKSPACE/','WORKSPACE/RASSINE_Stacked_spectrum_B0.00_')
    if (not os.path.exists(outname))|(force):
        t = fits.open(file)
        cdelt1 = t[0].header['CDELT1']
        cval1 = t[0].header['CRVAL1']
        flux = t[0].data
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
    
def query_value(header,kws):
    output = []
    for kw in kws:
        try:
            output.append(header[kw])
        except:
            output.append(np.nan)
    return output

def ra_to_deg(ra):
    if ra==ra:
        ra = float(ra)
        h  = int(ra // 10000)
        m  = int((ra % 10000) // 100)
        s  = ra % 100
        return (h + m/60 + s/3600) * 15
    else:
        return np.nan

def dec_to_deg(dec):
    if dec==dec:
        dec = float(dec)
        sign = -1 if dec < 0 else 1
        dec = abs(dec)
        d  = int(dec // 10000)
        m  = int((dec % 10000) // 100)
        s  = dec % 100
        return sign * (d + m/60 + s/3600)
    else:
        return np.nan

def get_vmacro(teff,logg, source='Cretignier+26'):
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
    elif source=='Cretignier+26':
        calib_g2 = myc.tableXY(np.array(myv.calib_vmacro)[:,0],np.array(myv.calib_vmacro)[:,1],np.array(myv.calib_vmacro)[:,0]*0)
        calib_garfield = myc.tableXY(np.array(myv.calib_vmacro)[:,0],np.array(myv.calib_vmacro)[:,2],np.array(myv.calib_vmacro)[:,0]*0)
        calib_kitty = myc.tableXY(np.array(myv.calib_vmacro)[:,0],np.array(myv.calib_vmacro)[:,3],np.array(myv.calib_vmacro)[:,0]*0)
        calib_g2.interpolate(new_grid=np.array([teff]),method='linear')
        calib_garfield.interpolate(new_grid=np.array([teff]),method='linear')
        calib_kitty.interpolate(new_grid=np.array([teff]),method='linear')
        value = (np.round(calib_g2.y[0],4),np.round(calib_garfield.y[0],4),np.round(calib_kitty.y[0],4))
    value = {'G2':value[0],'Garfield':value[1],'Kitty':value[2]}
    return value

def extract_header(files,instru,debug=False,ra=None,dec=None):
    instrument = instru.split('_')[0]
    ins = instrument[0:5]
    if (ins=='HARPS')&(instru.split('_')[-1]=='3.5'):
        ins = 'harps'
    all_infos = []
    kws = {'SOPHI':{'HIERARCH OHP DRS BJD':'rjd', 'HIERARCH OHP DRS BERV':'berv', 'HIERARCH OHP DRS CAL EXT SN30':'snr', 'HIERARCH OHP TARG ALPHA':'RA', 'HIERARCH OHP TARG DELTA':'DEC'},
           'NEID':{'OBSJD':'rjd', 'SSBRV100':'berv', 'EXTSNR':'snr', 'QRA':'RA', 'QDEC':'DEC'},
           'harps':{'ESO DRS BJD':'rjd', 'ESO DRS BERV':'berv', 'ESO DRS SPE EXT SN50':'snr', 'ESO TEL TARG ALPHA':'RA', 'ESO TEL TARG DELTA':'DEC'}, #old DRS (3.5)
           'HARPS':{'ESO QC BJD':'rjd', 'ESO QC BERV':'berv', 'ESO QC ORDER50 SNR':'snr', 'ESO TEL TARG ALPHA':'RA', 'ESO TEL TARG DELTA':'DEC'}, #new DRS (3.3.6)
           'HARPN':{'MJD-OBS':'rjd', 'TNG QC BERV':'berv', 'TNG QC ORDER50 SNR':'snr', 'TNG TEL TARG ALPHA':'RA', 'TNG TEL TARG DELTA':'DEC'}, #new DRS (3.0.1)
           'CORAL':{'ESO DRS BJD':'rjd', 'ESO DRS BERV':'berv', 'ESO DRS SPE EXT SN50':'snr', 'ESO TEL TARG ALPHA':'RA', 'ESO TEL TARG DELTA':'DEC'},
           'ESPRE':{'HIERARCH ESO QC BJD':'rjd', 'HIERARCH ESO QC BERV':'berv', 'HIERARCH ESO QC ORDER100 SNR':'snr', 'HIERARCH ESO TEL1 TARG ALPHA':'RA', 'HIERARCH ESO TEL1 TARG DELTA':'DEC'},
           'TBD':{'KEYWORD BJD':'rjd', 'KEYWORD BERV':'berv', 'KEYWORD SNR':'snr', 'KEYWORD ALPHA':'RA', 'KEYWORD DELTA':'DEC'},
           }
    for file in tqdm(files):
        header = fits.open(file)[0].header
        infos = query_value(header,list(kws[ins].keys()))
        all_infos.append(infos)
    all_infos = np.array(all_infos)
    summary = pd.DataFrame(all_infos,columns=list(kws[ins].values()))
    if debug:
        snaky_help()
        print(summary, ins)

    if ins=='SOPHI': 
        for i in summary.index:
            RA = summary.loc[i,'RA']
            length = len(str(RA).split('.')[0])
            RA = '0'*(6-length)+str(RA)
            summary.loc[i,'RA'] = RA
        summary['RA'] = np.round(np.array([ra_to_deg(ra) for ra in np.array(summary['RA'])]),6)
        summary['DEC'] = np.round(np.array([dec_to_deg(dec) for dec in np.array(summary['DEC'])]),6)
    if (ins=='NEID'):
        for i in summary.index:
            RA = summary.loc[i,'RA'].replace(':','')
            summary.loc[i,'RA'] = RA
            DEC = summary.loc[i,'DEC'].replace(':','')
            summary.loc[i,'DEC'] = DEC
        summary['RA'] = np.round(np.array([ra_to_deg(ra) for ra in np.array(summary['RA'])]),6)
        summary['DEC'] = np.round(np.array([dec_to_deg(dec) for dec in np.array(summary['DEC'])]),6)
    if (ins=='HARPN'):
        summary['rjd'] = summary['rjd'].astype('float') + 2400000
        for i in summary.index:
            RA = summary.loc[i,'RA'].replace('h','').replace('m','')
            summary.loc[i,'RA'] = RA
            DEC = summary.loc[i,'DEC'].replace(':','')
            summary.loc[i,'DEC'] = DEC
        summary['RA'] = np.round(np.array([ra_to_deg(ra) for ra in np.array(summary['RA'])]),6)
        summary['DEC'] = np.round(np.array([dec_to_deg(dec) for dec in np.array(summary['DEC'])]),6)
    if (ins=='ESPRE')|(ins=='HARPS')|(ins=='harps'): 
        summary['RA'] = np.round(np.array([ra_to_deg(ra) for ra in np.array(summary['RA'])]),6)
        summary['DEC'] = np.round(np.array([dec_to_deg(dec) for dec in np.array(summary['DEC'])]),6)
    if ins=='CORAL':
        summary['RA'] = np.round(summary['RA'].astype('float'),6)
        summary['DEC'] = np.round(summary['DEC'].astype('float'),6)

    if (instrument=='HARPS03')|(instrument=='HARPS15'):
        instrument = 'HARPS'

    if ra is not None:
        summary['RA'] = ra
        summary['DEC'] = dec

    summary['berv'] = np.round(summary['berv'].astype('float'),6)
    summary['snr'] = np.round(summary['snr'].astype('float'),1)
    ra_deg = np.nanmedian(summary['RA'])
    dec_deg = np.nanmedian(summary['DEC'])
    obstime = Time(summary['rjd'].astype('float'), format='jd', scale='utc')
    obstime_utc = obstime.utc.isot
    berv = get_berv(ra_deg, dec_deg, obstime_utc, instrument).value
    summary['rjd'] = summary['rjd'].astype('float') - 2400000
    summary['berv_computed'] = np.round(berv,4)
    return summary

def rassine_normalise(spec, min_radius=4.0, max_radius=76.0):
    spec.fit_rassine(min_radius, max_radius, 12.4, tag='%.0f'%(np.random.randint(1,10000)))
    return spec

def yarara_flux_density(files,sub_dico='matching_diff'):
    all_flux_density = []
    for j in tqdm(files):
        spec = import_spectrum(j,sub_dico=sub_dico)
        mask = (spec.x<6250)&(spec.x>4000)
        flux_norm = spec.y[mask]
        ha,hb = np.histogram(flux_norm,bins=100,density=True)
        hb = 0.5*(hb[1:]+hb[:-1])
        ha = np.nancumsum(ha)
        ha /= np.nanmax(ha)
        metric = hb[myf.find_nearest(ha,np.array([0.05,0.10,0.15,0.20,0.25]))[0]]
        all_flux_density.append(metric)

    all_flux_density = np.array(all_flux_density)
    all_flux_density = np.median(all_flux_density,axis=0)

    print('\n [INFO] Flux density 5, 10, 15, 20, 25 : ',np.round(all_flux_density,2))

    xgb_file = '/Python/Material_snaky/xgb_model_yarara_atmos_FluxD.p'
    xgb_obj = pickle.load(open(root+xgb_file,'rb'))
    model = xgb_obj['model']

    output = model.predict(all_flux_density[:,np.newaxis].T)
    Teff_rough_est = int(np.round(output[0,0],0)) # not better than +/- 300K
    FeH_rough_est = np.round(output[0,1],3) # not better than +/- 0.15 dex

    print(' [INFO] Rough Teff estimation %.0f +/- 300 K'%(Teff_rough_est))
    print(' [INFO] Rough FeH estimation %.2f +/- ?? dex'%(FeH_rough_est))

    return (Teff_rough_est,FeH_rough_est)

def yarara_rough_rv_sys(spec,teff=6000, verbose=False):
    
    if verbose:
        print(' [INFO] Rough RV_sys estimation...')

    wave = spec.x
    flux = spec.y
    if teff>6500:
        if verbose:
            print(' [INFO] Selected line set Teff>6500')        
        lines = [myv.Heps[0],myv.Hd[0],myv.Hb[0],myv.Hc[0],myv.Ha[0]]
        box_pts = 50
    else:
        if verbose:
            print(' [INFO] Selected line set Teff<6500')
        lines = [myv.NaDl[0],myv.NaDr[0],myv.Mg1b[0],myv.Mg1c[0],myv.Ha[0]]
        box_pts = 7
    right,left = myf.doppler_r(lines,250*1000) # 200 km/s search

    RV = []
    for r,l,c in zip(right,left,lines):
        mask_wave = (wave>l)&(wave<r)
        if np.sum(mask_wave):
            flux2 = flux[mask_wave]
            rvs = []
            s = myc.tableXY(wave[mask_wave],flux2,0*flux2)
            s.smooth(box_pts=box_pts,shape='rectangular')
            s.find_min()
            mini = s.x_min[np.argmin(s.y_min)]
            rv = (mini-c)/c*myv.c_lum/1000
            rvs.append(rv)
            RV.append(rvs)
    RV = np.array(RV)
    RV = np.nanmedian(RV,axis=1)
    RV_sys = np.round(np.median(RV),2)

    if verbose:
        print(' [INFO] Measured values :',np.round(RV,2))
        print(' [INFO] Rough RV_sys estimation = %.2f km/s'%(RV_sys))

    return RV_sys    

def yarara_check_rv_sys(spec, fwhm, rv_sys_approx, dir_root=None):
    #UPDATE 12.12.2023 producing the plot even if condition satisfied

    print(' [INFO] Selected CCF mask : MagiCat')
    mask = np.genfromtxt(root+'/Python/Material_snaky/MASK_CCF/Magicat.txt')
    mask = np.array([0.5*(mask[:,0]+mask[:,1]),mask[:,2]]).T
    mask_harps = 'G2'

    rv_range = [15,fwhm][int(fwhm>15)]

    rv_sys_fit = rv_sys_approx*1000 # Update 28.08.24
    rv_sys_est1 = rv_sys_fit/1000

    spec.ccf(mask, weighted=True, rv_range=rv_range*1.5,rv_sys=rv_sys_fit)
    
    rv_sys_fit += spec.ccf_params['cen'].value
    
    rv_sys_fit = np.round(rv_sys_fit/1000,2)

    fwhm = np.min([100,spec.ccf_params['wid'].value/1000*2.355])
    fwhm = np.round(fwhm,2)
    print('\n [INFO] FWHM value fitted as %.2f kms'%(fwhm))

    rv_sys_est2 = rv_sys_fit

    warning = 0
    if (abs(rv_sys_est1-rv_sys_est2)/abs(rv_sys_est1)*100)>20:
        print('\n [WARNING] The two RV sys estimations (%.1f km/s, %.1f km/s) are very different!'%(rv_sys_est1,rv_sys_est2))
        if abs(rv_sys_est1)<300:
            print(' [INFO] Second attempt to fit a CCF with standard HARPS DRS mask')
            mask = np.genfromtxt(root+'/Python/Material_snaky/MASK_CCF/%s.txt'%(mask_harps))
            mask = np.array([0.5*(mask[:,0]+mask[:,1]),mask[:,2]]).T 
            spec.ccf(mask, weighted=True, rv_range=rv_range*1.5,rv_sys=rv_sys_est1*1000)
            rv_sys_fit = rv_sys_est1*1000 + spec.ccf_params['cen'].value
            rv_sys_fit = np.round(rv_sys_fit/1000,2)
            rv_sys_est3 = rv_sys_fit
            fwhm = spec.ccf_params['wid'].value/1000*2.355
            print('\n [INFO] %.1f km/s | %.1f km/s | %.1f km/s for FWHM = %.1f'%(rv_sys_est1,rv_sys_est2,rv_sys_est3,fwhm))
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

    y_min = 1-2*contrast_fit/100
    if y_min<0:
        y_min=0
    plt.ylim(y_min,1.1)

    print('\n [INFO] RV_sys value fitted as %.2f kms'%(rv_sys_fit))
    
    if dir_root is not None:
        plt.savefig(dir_root+'IMAGES/RV_sys_fitting.pdf')
    
    SB2 = 0
    if spec.warning_multipeak==1:
        SB2 = 1
        if dir_root is not None:
            plt.savefig(dir_root+'WARNING/WARNING_RV_sys_fitting.pdf')

    ccf = pd.DataFrame(np.array([spec.ccf_profile.x/1000,spec.ccf_profile.y]).T,columns=['vrad','ccf'])
    contrast = np.round(contrast_fit/100,3)
    
    output = (fwhm,rv_sys_fit,contrast,ccf_beta,SB2,ccf)
    if dir_root is not None:
        ccf.to_csv(dir_root+'STAR_INFO/CCF_RV_SYS.csv')
    return output

def yarara_check_rv_sys_wrapper(dir_root,spec,rv_sys_approx):
    
    myf.print_box('\n---- RECIPE : RV_SYS EXTRACTION ---- \n')

    spec.clip(min=[4000,None])

    save = []
    for fwhm in [6,10,20,50,100,200][::-1]:
        sinfo = yarara_check_rv_sys(spec, fwhm, rv_sys_approx, dir_root=dir_root)
        if sinfo[0]>500:
            save.append([fwhm,-999,-999,-999])
        else:
            save.append([fwhm,sinfo[0],sinfo[1],rv_sys_approx])
    save = np.array(save)
    plt.close('all')

    print(' [INFO] Table summary FWHM | RV_SYS \n')
    save[save[:,2]==save[:,3],2] = -999
    print(save)
    fwhm1 = save[np.argmin(abs(save[:,2]-save[:,3])),1]
    fwhm = save[np.argmin(abs(save[:,0]-fwhm1)),1]
    rv_sys = save[np.argmin(abs(save[:,0]-fwhm1)),2]
    print('\n [INFO] Best FWHM detected is %.2f km/s \n'%(fwhm))
    sinfo = yarara_check_rv_sys(spec, fwhm, rv_sys, dir_root=dir_root)
    
    if fwhm<50:
        pass#yarara_check_fwhm()

    return sinfo


def replace_none(y,yerr):
    if yerr is None:
        return np.nan, 1e6
    else:
        return y,yerr

def yarara_ccf(dir_root, files, rv_sys, fwhm, beta_gnd, mask, spectra=None, 
                mask_col='weight_rv', analytical_model='auto', sub_dico='matching_diff',
                weighted=True, debug=False, normalisation='left', return_ccf=False,
                del_outside_max = False, ccf_oversampling=1, check_non_transform=True, continuum_method='flux',
                rv_range=None, rv_borders=None, bis_range=None, delta_window=5, rv_shift=None,
                wave_min=4000, wave_max=10000, hole_left=0, hole_right=0, squared=True):
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

    ins = dir_root.split('/')[-2]
    jdb = get_jdb(files,dir_root)

    print(' [INFO] RV sys : %.2f [km/s] '%(rv_sys))
    rv_sys = 1000*rv_sys

    print('\n [INFO] FWHM: %.2f kms'%(fwhm))
    if rv_range is None:
        rv_range=int(3*fwhm)
        print(' [INFO] RV range updated to : %.1f kms'%(rv_range))
    
    if rv_borders is None:
        rv_borders=int(2*fwhm)
        print(' [INFO] RV borders updated to : %.1f kms'%(rv_borders))
    
    if bis_range is None:
        bis_range=np.round(0.33*fwhm,1)
        print(' [INFO] BISSPAN borders updated to : %.1f kms'%(bis_range))

    if analytical_model=='auto':
        analytical_model = 'gaussian'
        if beta_gnd>2.5:
            analytical_model = 'GND%.1f'%(beta_gnd)
    print(' [INFO] CCF analytical model :',analytical_model)
    
    if type(mask)==str:
        ccf_name = mask
        mask_name = mask
        mask_loc = root+'/Python/Material_snaky/MASK_CCF/'+mask+'.txt'
        mask = np.genfromtxt(mask_loc)
        mask = np.array([0.5*(mask[:,0]+mask[:,1]),mask[:,2]]).T
        print('\n [INFO] CCF mask selected : %s'%(mask_loc))
    elif type(mask)==pd.core.frame.DataFrame:
        mask = np.array([np.array(mask['freq_mask0']).astype('float'),np.array(mask[mask_col]).astype('float')]).T
        mask_name = 'ManualDF'

    shift_rv = np.zeros(len(files))
    if type(rv_shift)==np.ndarray:
        shift_rv = rv_shift
    
    mask[:,0] = myf.doppler_r(mask[:,0],rv_sys)[0]
            
    if spectra is None:
        grid, flux, flux_err = import_sts(files, rv_shift=shift_rv, err=False, sub_dico=sub_dico)
    else:
        grid, flux, flux_err = spectra

    print('\n [INFO] Reference color : flat normalised continuum')
    
    mask_shifted = myf.doppler_r(mask[:,0],(rv_range+5)*1000)
    
    mask = mask[(myf.doppler_r(mask[:,0],30000)[0]<grid.max())&(myf.doppler_r(mask[:,0],30000)[1]>grid.min()),:] #supres line farther than 30kms
    mask = mask[mask[:,0]>wave_min,:] 
    mask = mask[mask[:,0]<wave_max,:] 
    
    mask_min = np.min(mask[:,0])
    mask_max = np.max(mask[:,0])

    print('\n [INFO] Nb lines in the mask : %.0f'%(len(mask)))
    print(' [INFO] Wave min : %.0f AA | Wave max : %.0f AA'%(mask_min,mask_max))

    #supress useless part of the spectra to speed up the CCF
    grid_min = int(myf.find_nearest(grid,myf.doppler_r(mask_min,-100000)[0])[0])
    grid_max = int(myf.find_nearest(grid,myf.doppler_r(mask_max,100000)[0])[0])
    grid = grid[grid_min:grid_max]
    flux = flux[:,grid_min:grid_max]
    if flux_err is not None:
        flux_err = flux_err[:,grid_min:grid_max]

    log_grid = np.linspace(np.log10(grid).min(),np.log10(grid).max(),len(grid))
    dgrid = log_grid[1] - log_grid[0]
    #dv = (10**(dgrid)-1)*299.792e6    

    #computation of region free of spectral line to increase code speed
    used_region = ((10**log_grid)>=mask_shifted[1][:,np.newaxis])&((10**log_grid)<=mask_shifted[0][:,np.newaxis])
    used_region = (np.sum(used_region,axis=0)!=0).astype('bool')
    print(' [INFO] Percentage of the spectrum used : %.1f [%%] (%.0f)'%(100*sum(used_region)/len(grid),len(grid)))
    time.sleep(1)

    if not os.path.exists(dir_root+'CCF_MASK/CCF_'+mask_name.split('.')[0]+'.fits'):
        print('\n [INFO] CCF mask reduced for the first time, wait for the static mask producing... \n')
        time.sleep(1)
        mask_wave = np.log10(mask[:,0])
        mask_contrast = mask[:,1]*weighted + (1-weighted)
                    
        mask_hole = (mask[:,0]>myf.doppler_r(hole_left,-30000)[0])&(mask[:,0]<myf.doppler_r(hole_right,30000)[0])
        mask_contrast[mask_hole] = 0
        
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
        hdul.writeto(dir_root+'CCF_MASK/CCF_'+mask_name.split('.')[0]+'.fits')
        print('\n [INFO] CCF mask saved under : %s'%(dir_root+'CCF_MASK/CCF_'+mask_name.split('.')[0]+'.fits'))
    else:
        print('\n [INFO] CCF mask found : %s'%(dir_root+'CCF_MASK/CCF_'+mask_name.split('.')[0]+'.fits'))
        log_grid_mask, log_mask = fits.open(dir_root+'CCF_MASK/CCF_'+mask_name.split('.')[0]+'.fits')[0].data.T
    
    log_mask = log_mask**(1.0+float(squared))
    log_template = myc.tableXY(log_grid_mask,log_mask,0*log_mask)
    log_template.interpolate(new_grid=log_grid,method='linear',replace=True)
    log_template = log_template.y

    amplitudes = [] ; amplitudes_std = []
    rvs = [] ; rvs_std = []
    fwhms = [] ; fwhms_std = []
    ew = [] ; ew_std = []
    centers = [] ; centers_std = []
    depths = [] ; depths_std = []
    bisspan = []  ; bisspan_std = []
    
    now = datetime.datetime.now()
    print('\n Computing CCFs (Current time %.0fh%.0fm%.0fs) \n'%(now.hour, now.minute, now.second))
    
    for j,i in enumerate(files):   
        if flux_err is None:
            f_err = 0*flux[j]
        else:
            f_err = flux_err[j]
        temp = myc.tableXY(np.log10(grid), flux[j], f_err)
        temp.interpolate(new_grid=log_grid,method='cubic')
        flux[j] = temp.y
        if flux_err is not None:
            flux_err[j] = temp.yerr

    gravity_center_wave = np.sum(10**log_grid[used_region]*log_template[used_region])/np.sum(log_template[used_region])
    
    print('\n [INFO] Gravity center wavelength = %.0f AA \n'%(gravity_center_wave))
    flux = flux[:,used_region]
    log_grid = log_grid[used_region]
    log_template = log_template[used_region]
    if flux_err is not None:
        flux_err = flux_err[:,used_region]

    vrad, ccf_power, ccf_power_std = myf.ccf(log_grid, flux, log_template, 
                                                rv_range = rv_range, oversampling = ccf_oversampling, spec1_std = flux_err) #to compute on all the ccf simultaneously
    
    del flux
    del flux_err

    now = datetime.datetime.now()
    dv = np.median(np.diff(vrad))
    print('')
    print('\n CCFs computed (Current time %.0fh%.0fm%.0fs)'%(now.hour, now.minute, now.second))
    print('\n [INFO] CCF velocity step : %.0f m/s'%(dv))

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

    print(' [INFO] Beta value of GND = %.2f'%(beta0))
    if (beta0>2.5)&(analytical_model=='gaussian'):
        print(' \n [WARNING] Significant Kurtosis detected.')

    dccf2 = (ccf_power-ccf_ref[:,np.newaxis])[top_ccf]/np.mean(ccf_power[continuum_ccf])*100
    dccf2 -= np.median(dccf2,axis=0)
    ccf_snr = 1/np.std(dccf2,axis=0)*100
    print(' [INFO] SNR CCF continuum median : %.0f'%(np.median(ccf_snr)))

    noise_ccf = (np.sqrt(ccf_ref/np.max(ccf_ref))*ccf_ref[continuum_ccf])[:,np.newaxis]/ccf_snr #assume that the noise in the continuum is white (okay for matching_mad but wrong when tellurics are still there)
    sigma_rv = noise_ccf/(abs(np.gradient(ccf_ref))/np.gradient(vrad))[:,np.newaxis]
    w_rv = (1/sigma_rv)**2
    svrad_phot = 1/np.sqrt(np.sum(w_rv,axis=0))
    scaling = np.sqrt(820/np.mean(np.gradient(vrad))) #to penalize oversampling in vrad 
    svrad_phot*=scaling
    
    svrad_phot[svrad_phot==0] = 2*np.max(svrad_phot) #in case of null values
    
    print(' [INFO] Photon noise RV median : %.2f m/s\n '%(np.median(svrad_phot)))        
    
    svrad_phot2 = {}
    svrad_phot2['rv'] = 10**(0.98*np.log10(svrad_phot)-3.08) # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2['contrast'] = 10**(0.98*np.log10(svrad_phot)-3.58) # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2['fwhm'] = 10**(0.98*np.log10(svrad_phot)-2.94) # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2['center'] = 10**(0.98*np.log10(svrad_phot)-2.83) # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2['depth'] = 10**(0.97*np.log10(svrad_phot)-3.62) # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2['ew'] = 10**(0.97*np.log10(svrad_phot)-3.47) # from photon noise simulations Photon_noise_CCF.py
    svrad_phot2['vspan'] = 10**(0.98*np.log10(svrad_phot)-2.95) # from photon noise simulations Photon_noise_CCF.py
        
    print(' [INFO] Photon noise RV from calibration : %.2f m/s '%(np.median(svrad_phot2['rv'])*1000))

    print(' [INFO] Number of velocity bin = %.0f'%(len(vrad)))

    if np.sum(noise_ccf!=0)>0:
        noise_ccf[noise_ccf==0] = np.mean(noise_ccf[noise_ccf!=0])
    else:
        noise_ccf *= 0
        noise_ccf += 0.01
    ccf_power_std = noise_ccf
    factor = 1/(np.percentile(noise_ccf,75,axis=0))**2
    ccf_power = ccf_power*factor
    ccf_power_std = ccf_power*factor

    for j,i in enumerate(files):
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
                print(' \n[WARNING] Discrepancy detected between CCFs (%.4f/%.4f), value reset to non-transformed one'%(V1,V2))
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
    
    rvs_std = svrad_phot2['rv']
    fwhms = np.array(fwhms).astype('float')*2.355
    fwhms_std = np.array(fwhms_std).astype('float')*2.355
    
    warning_rv_borders = False
    if np.median(fwhms)>(rv_borders/1.5):
        print(' [WARNING] The CCF is larger than the RV borders for the fit')
        warning_rv_borders = True
    
    if jdb is None:
        jdb = np.arange(len(files))
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
    ccf_infos['filename'] = files

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
    rvs = rvs - np.nanmedian(rvs)
    for n,rv in enumerate(rvs):
        if rv==rv:
            profile = myc.tableXY(vrad-rv,ccf_shifted[:,n],0*vrad)
            profile.interpolate(new_grid=vrad)
            ccf_shifted[:,n] = profile.y
    master_ccf = np.nanmean(ccf_shifted,axis=1)
    ccf_res = ccf_norm - master_ccf[:,np.newaxis]

    export = myf.touch_pickle(dir_root+'WORKSPACE/Analyse_ccf_saved.p')
    export['CCF_'+ccf_name] = {}
    export['CCF_'+ccf_name][sub_dico] = {'ccf_vrad':vrad,'ccf_flux':ccf_norm,'ccf_shifted':ccf_shifted,'ccf_master':master_ccf,'filename':files}
    myf.pickle_dump(export,open(dir_root+'WORKSPACE/Analyse_ccf_saved.p','wb'))

    warning = 0
    if ccf_name=='mask_telluric_o2':
        fwhm_ins = np.nanmedian(ccf_fwhm.y)
        if ins.split('_')[0] in myv.instrument_res_kms.keys():
            ref = myv.instrument_res_kms[ins.split('_')[0]]
            print(' [INFO] Reference value for %s is %.1f km/_s'%(ins,ref))
            if abs(ref - fwhm_ins)>1:
                warning = 1
                print(Fore.YELLOW + '\n [WARNING] Instrumental resolution is not usual (%.1f km/s)'%(fwhm_ins)+Fore.RESET)
        else:
            ref = np.nan

    if (ccf_name=='Garfield')&(np.nanstd(ccf_rv.y)>1000): # SB flag
        warning = 1

    plt.figure(figsize=(9,8))
    plt.axes([0.1,0.72,0.6,0.22])
    med = np.nanmedian(ccf_rv.y)
    ccf_rv.plot() ; plt.ylabel('RV [m/s]') ; plt.axhline(y=med,color='r',label='%.1f'%(med))
    plt.axes([0.1,0.50,0.6,0.22])
    med = np.nanmedian(ccf_fwhm.y)
    ccf_fwhm.plot() ; plt.ylabel('FWHM [km/s]') ; plt.axhline(y=med,color='r',label='%.2f km/s'%(med))
    if ccf_name=='mask_telluric_o2':
        plt.axhline(y=ref,color='k',ls='-.',lw=1)
    plt.legend(loc=3)
    plt.axes([0.1,0.28,0.6,0.22])
    med = np.nanmedian(ccf_contrast.y)
    ccf_contrast.plot() ; plt.ylabel('CT [%]') ; plt.axhline(y=med,color='r',label='%.1f %%'%(med))
    plt.legend(loc=3)
    plt.axes([0.1,0.06,0.6,0.22])
    med = np.nanmedian(ccf_vspan.y)
    ccf_vspan.plot() ; plt.ylabel('VSPAN [m/s]') ; plt.axhline(y=med,color='r',label='%.1f'%(med))
    plt.axes([0.75,0.06,0.22,0.66])
    plt.imshow(ccf_res.T,vmin=-0.05,vmax=0.05,aspect='auto',cmap='seismic') ; 
    plt.axvline(x=len(vrad)*0.5,color='k',ls='-.',lw=1)
    plt.axes([0.75,0.72,0.22,0.22])
    plt.plot(vrad/1000,master_ccf,color='k')
    plt.plot(vrad/1000,ccf_norm,alpha=0.2,color='k')
    plt.axvline(x=0,color='k',ls='-.',lw=1) 
    plt.tick_params(top=True,labeltop=True,labelbottom=False)
    plt.savefig(dir_root+'IMAGES/CCF_summary_%s.pdf'%(ccf_name))
    if warning:
        plt.savefig(dir_root+'WARNING/CCF_summary_%s.pdf'%(ccf_name))

    output = {
        'rv':ccf_rv,
        'contrast':ccf_contrast,
        'fwhm':ccf_fwhm,
        'vspan':ccf_vspan}

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
        mask = np.in1d(np.array(summary['filename']),files)
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
        table = np.load(root+'/Python/Material_snaky/template_star_ATLAS_3800_7000.npy')
        table_columns = pd.read_csv(root+'/Python/Material_snaky/template_star_ATLAS_3800_7000.csv',index_col=0).columns[1:]
        code = [[i,i.split('T')[1].split('_')[0],i.split('g')[1].split('_')[0],i.split('H')[-1].split('_')[0]] for i in table_columns]
        code = np.array(code)
        code[:,-1] = 0.0
        wave = np.unique(np.round(np.arange(3800,7000,0.01),2))
    else:
        table = np.load(root+'/Python/Material_snaky/template_star_SNAKY_3900_6800.npy')
        table_columns = pd.read_csv(root+'/Python/Material_snaky/template_star_SNAKY_3900_6800.csv',index_col=0).columns[1:]
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

def import_star_info(dir_root):
    star = dir_root.split('/data')[0].split('/')[-1]
    sinfo = pd.read_pickle(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star))
    return sinfo

def import_ccf_profile(dir_root,mask_name):
    ccf_profile = pd.read_pickle(dir_root+'WORKSPACE/Analyse_ccf_saved.p')['CCF_%s'%(mask_name)]
    return ccf_profile

def import_ccf(dir_root,mask_name):

    ccf_infos = pd.read_pickle(dir_root+'WORKSPACE/Analyse_ccf.p')['CCF_%s'%(mask_name)]['table']
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

    myf.print_box('\n---- RECIPE : EW OF LINE SPECIES ----\n')

    print('\n [INFO] Metallicity abundances recipe launched')
    print(' [INFO] FWHM = %.2f km/s'%(fwhm))

    rv_range = 3*fwhm
    sigma_3wid = rv_range/2.3556
    grid = np.arange(0,sigma_3wid*1000,100) 
    grid = np.hstack([-grid[1:][::-1],grid])
    
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
        print('\n [INFO] Contrast Z = %.3f | EW = %.1f mA'%(contrast0*100,ew0))
        
        mask1 = np.genfromtxt(root+'/Python/Material_snaky/MASK_CCF/FeIU.txt')
        mask11 = np.array([0.5*(mask1[:,0]+mask1[:,1]),mask1[:,2]]).T
        master.ccf(mask11, weighted=False, rv_range=rv_range,rv_sys=rv_sys*1000,fit_gaussian=False)

        master.ccf_profile.smooth(box_pts=10,replace=False,shape='rectangular')
        master.ccf_profile.y /= np.max(master.ccf_profile.smoothed.y)
        master.ccf_profile.interpolate(new_grid=grid,replace=False)

        contrast1 = 1-np.mean(master.ccf_profile.y_interp)
        contrast1 = np.sum(1-master.ccf_profile.y_interp)*np.diff(grid)[0]/1000
        ew1 = contrast1/3e5*np.mean(mask1[:,0])*1000

        plt.fill_between(grid,master.ccf_profile.y_interp,1,color='g',alpha=0.2,label='%.2f'%(contrast1))
        plt.legend(loc=3)
        plt.savefig(dir_root+'IMAGES/Atmos_FeIU.pdf')
        
        print('\n [INFO] Contrast FeIU = %.3f kms | EW = %.1f mA'%(contrast1,ew1))

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

        for species in ['FeIS','FeIIS','TiI','VI','MnI','NdII','TiII','CrI','NiI','CoI','CaI','SiI','ScII','CaH','LiI']:
            count+=1
            mask2 = np.genfromtxt(root+'/Python/Material_snaky/MASK_CCF/%s.txt'%(species))
            mask22 = np.array([0.5*(mask2[:,0]+mask2[:,1]),mask2[:,2]]).T
            master.ccf(mask22, weighted=False, rv_range=rv_range,rv_sys=rv_sys*1000,fit_gaussian=False)
            master.ccf_profile.smooth(box_pts=10,replace=False,shape='rectangular')
            master.ccf_profile.y /= np.max(master.ccf_profile.smoothed.y)
            master.ccf_profile.interpolate(new_grid=grid,replace=False)
            contrast2 = 1-np.mean(master.ccf_profile.y_interp)
            contrast2 = np.sum(1-master.ccf_profile.y_interp)*np.diff(grid)[0]/1000
            ew2 = contrast2/3e5*np.mean(mask2[:,0])*1000
            plt.fill_between(grid,master.ccf_profile.y_interp,1,color='g',alpha=0.2,label='%.2f'%(contrast2))
            plt.legend(loc=3)
            plt.savefig(dir_root+'IMAGES/Atmos_%s.pdf'%(species))

            print('\n [INFO] Contrast %s = %.3f kms | EW = %.1f mA'%(species,contrast2,ew2))
            
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
        plt.subplots_adjust(hspace=0.35,wspace=0.35,top=0.95,left=0.07,right=0.96)
        plt.savefig(dir_root+'IMAGES/Atmos_all.pdf')
    return Contrast, EW

def yarara_atmos_xgb_spectroscopy(dir_root, star_info, resolution=110000, phot=False):
    
    myf.print_box('\n---- RECIPE : XGB ATMOSPHERIC PARAMETERS ----\n')

    if phot:
        lines = ['FeIU','FeIS','FeIIS','TiI','VI','MnI','NdII','TiII','CaH','Z']
        xgb_file = '/Python/Material_snaky/xgb_model_yarara_atmos_phot.p'
    else:
        lines = ['Ha','NaD','MgI','Hb','FeIU','FeIS','FeIIS','TiI','VI','MnI','NdII','TiII','CaH','Z']
        xgb_file = '/Python/Material_snaky/xgb_model_yarara_atmos.p'
    ew = np.array([star_info['Contrast'][kw] for kw in lines])
    rv_sys = star_info['Rv_sys']['SNAKY']

    print(' [INFO] EW:',np.round(np.hstack(ew.T),2))

    R_ratio = resolution/110000 # CORALIE and ESPRESSO similar on HD10700
    factor = 1 # Update 31.10.24 seems no more useful now that EW is used

    xgb_obj = pickle.load(open(root+xgb_file,'rb'))
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
    teff = int(teff)
    feh = np.round(feh,3)
    logg = np.round(logg,3)

    M, dust, dust = myf.find_stellar_mass_radius(teff, sp_type='G2V')
    M_inf, dust, dust = myf.find_stellar_mass_radius(teff-75, sp_type='G2V')
    M_sup, dust, dust = myf.find_stellar_mass_radius(teff+75, sp_type='G2V')
    dM = 0.5*(M_sup-M_inf)
    R = np.exp(0.5*np.log(10)*(4.437+np.log(M)-logg)) #Smette 2005
    dR = 0.08*np.exp(0.5*np.log(10)*(4.437+np.log(M)-logg))
    M, dM, R, dR, samples_ms, samples_rs = myf.find_stellar_mass_radius_MS(teff, logg, samples=99999) #new function
    M = np.round(M,2)
    R = np.round(R,2)

    BV = -3.684*np.log10(teff) + 14.551 #http://www.isthe.com/chongo/tech/astro/HR-temp-mass-table-byhrclass.html
    BV = BV - 0.04 # solar correction ZP for Sun = 0.65
    BV_std = np.std(-3.684*np.log10(np.random.randn(99999)*75+teff) + 14.551)

    vmicro, vmacro = myf.find_turbulence(teff, logg)
    BV = np.round(BV,3)
    vmicro = np.round(vmicro,2)
    vmacro = np.round(vmacro,2)

    print(' [INFO] Effective temperature = %.0f +/- 70 K'%(teff))
    print(' [INFO] Metallicity [Fe/H] = %.2f +/- 0.07 dex'%(feh))
    print(' [INFO] Log(g) = %.2f +/- 0.07 dex'%(logg))
    print(' [INFO] Ms = %.2f +/- %.2f Msol'%(M,dM))
    print(' [INFO] Rs = %.2f +/- %.2f Rsol'%(R,dR))
    print(' [INFO] BV = %.2f +/- 0.02'%(BV))
    print(' [INFO] Vmic = %.1f km/s '%(vmicro))
    print(' [INFO] Vmac = %.1f km/s '%(vmacro))

    samples_teff = np.random.randn(99999)*70+teff
    samples_feh = np.random.randn(99999)*0.07+feh
    samples_logg = np.random.randn(99999)*0.07+logg

    samples = pd.DataFrame(np.array([samples_ms, samples_rs, samples_teff, samples_logg, samples_feh]).T,columns=['ms','rs','teff','logg','feh'])
    samples.to_csv(dir_root+'WORKSPACE/Analyse_samples.csv')

    if teff<4000:
        teff=4000
    elif teff>7000:
        teff=7000

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
        plt.text(0.5,0.5,'FWHM = %.2f km/s | RV_sys = %.1f km/s\n'%(fwhm,rv_sys)+r'$T_{eff}$'+' = %.0f +/- 70 K  |  logg = %.2f +/- 0.07 dex  |  [Fe/H] = %.2f +/- 0.07 dex\n Ms = %.2f +/- %.2f |  Rs = %.2f +/- %.2f'%(teff,logg,feh, M, dM, R, dR),ha='center',va='center',fontsize=15)
        plt.savefig(dir_root+'IMAGES/Atmos_all.pdf')

    return teff,feh,logg,M,R,BV,vmicro,vmacro


def yarara_vcat(dir_root, sub_dico='matching_diff', Prot=None):

    myf.print_box('\n---- RECIPE : VSINI EXTRACTION ----\n')

    sinfo = import_star_info(dir_root)

    teff = sinfo['Teff']['SNAKY']
    logg = sinfo['Log_g']['SNAKY']
    feh = sinfo['FeH']['SNAKY']
    ins_res = sinfo['FWHM']['O2']

    instrument = dir_root.split('/')[-2]
    ins = instrument.split('_')[0]
    ref_resolution = myv.instrument_res_kms[ins]
    diff = ref_resolution - ins_res

    print(' [INFO] Reference instrument resolution = %.2f km/s'%(ref_resolution))
    print(' [INFO] Telluric measured one = %.2f km/s (Delta = %.2f)'%(ins_res,diff))
    
    if instrument[0:6]=='SOPHIE':
        if abs(diff)>1:
            print(Fore.YELLOW+'\n [WARNING] Resolution is too different from reference value, O2 correction applied. \n'+Fore.RESET)
        else:
            ins_res = ref_resolution
    else:
        ins_res = ref_resolution

    calib_product = 'Calib_HARPN_GKM_vsini_HD10700.p'
    print(' [INFO] Calibration product used : %s'%(calib_product))
    calib = pd.read_pickle(root+'/Python/Material_snaky/'+calib_product)

    calib_curve = {}
    for kw in ['GARFIELD','KITTY']:
        G = myc.tableXY(calib['%s_FWHM'%(kw)],calib['%s_VSINI'%(kw)],0*calib['%s_VSINI'%(kw)])
        if kw=='GARFIELD': # solar correction
            G.y = G.y-0.598/(G.x/6.085)**2
        elif kw=='KITTY':  # solar correction
            G.y = G.y-0.708/(G.x/5.895)**2
        calib_curve[kw] = G
    calib_curve['G2'] = calib_curve['GARFIELD']

    samples_table = pd.read_csv(dir_root+'WORKSPACE/Analyse_samples.csv',index_col=0)

    vmacro = get_vmacro(teff,logg,source='Cretignier+26') #Doyle, Bruntt, or Cretignier
    vmacro_sun = get_vmacro(5775,4.44,source='Cretignier+26')
    for kw in vmacro.keys():
        print(' [INFO] Cretignier+26 vmacro for %s (Teff = %.0f K) = %.1f kms (Sun = %.1f km/s)'%(kw,teff,vmacro[kw],vmacro_sun[kw]))

    vsini_cdf = {}
    samples = []
    num = -1

    ccf_values = import_ccf(dir_root,'G2')
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
        fwhmG = []
        plt.figure()
        for s in saveG:
            c = myc.tableXY(vrad,s,0*s)
            c.clip(min=[-vmax*0.4,None],max=[vmax*0.4,None])
            c.interpolate(new_grid=1000,method='cubic')
            v2 = c.x[c.x>0][np.argmin(abs(c.y-0.5)[c.x>0])]
            v1 = c.x[c.x<0][np.argmin(abs(c.y-0.5)[c.x<0])]
            fwhm = (v2-v1)/1000
            fwhmG.append(fwhm)
        fwhmG = np.array(fwhmG)
        kw = mask.upper()
        ins_calib = ins

        fwhmG = np.sqrt(fwhmG**2-ins_res**2+ref_resolution**2)

        calib_ins = pd.read_csv(root+'/Python/Material_snaky/Table_calib_vsini_%s.csv'%(kw),index_col=0)
        calib_ins = myc.tableXY(calib_ins[ins_calib],calib_ins['HARPN'],0*calib_ins[ins_calib]) #reference HARPN
        calib_ins.order()
        calib_ins.interpolate(new_grid=fwhmG,replace=False,method='linear')
        fwhmG_HARPN = calib_ins.y_interp

        #print(fwhmG_HARPN)
        #fwhmG_HARPN = np.sqrt(fwhmG_HARPN**2 - vmacro[mask]**2 + vmacro_sun[mask]**2)
        #print(fwhmG_HARPN)

        G = calib_curve[kw]
        G.interpolate(new_grid=fwhmG_HARPN,method='linear',replace=False)
        V = G.y_interp

        plt.figure('vsin3')
        plt.subplot(3,1,num+1) ; plt.scatter(ccf_values['rv'].x,V,marker='.',color='C%.0f'%(num))
        plt.ylabel(r'$v$ $\sin$ $i$ [km/s]')
        plt.tick_params(labelbottom=False,direction='inout',top=True)
        ax = plt.gca()
        ax.twinx()
        plt.scatter(ccf_values['rv'].x,fwhmG,marker='.',color='C%.0f'%(num))
        plt.ylabel('CCF FWHM\n%s [km/s]'%(ins))
        plt.subplot(3,1,3) ; plt.scatter(ccf_values['rv'].x,V,marker='.')

        mean_F = np.nanmedian(fwhmG_HARPN)
        std_accuracy = (1-np.exp(-abs(teff-5780)/200))*0.1+0.1 # 200 m/s of bias in general, 250 m/s for the Sun
        print(' [INFO] Accuracy uncertainties = %.0f m/s'%(std_accuracy*1000))
        print(' [INFO] Precision uncertainties = %.0f m/s'%(myf.mad(fwhmG_HARPN)*1000))
        std_F = np.sqrt(myf.mad(fwhmG_HARPN)**2 + std_accuracy**2) 
        plt.figure('vsini2')
        G.interpolate(new_grid=np.sort(np.random.randn(99999)*std_F+mean_F),method='linear',replace=False)
        V = G.y_interp

        sample = V
        sample = sample[sample>0]
        samples.append(sample)
        plt.hist(sample,bins=100,density=True,histtype='step')
        plt.hist(sample,bins=100,density=True,alpha=0.4,color='C%.0f'%(num),label=mask)
        plt.xlabel(r'v $\sin$ i [km/s]')
        plt.figure('dust')
        infos = plt.hist(sample,bins=100,density=True,histtype='step',cumulative=True)
        plt.close('dust')
        infos = myc.tableXY(0.5*(infos[1][1:]+infos[1][:-1]),infos[0],0*infos[0])
        vsini_cdf[kw] = infos
    plt.legend()     
    samples = np.hstack(samples)
    plt.hist(samples,bins=100,density=True,histtype='step',color='k',lw=2)
    plt.title(r'v $\sin$ i = %.2f +/- %.2f km/s'%(np.mean(samples),np.std(samples)))
    print(' [INFO] v sin i = %.2f +/- %.2f km/s'%(np.mean(samples),np.std(samples)))
    plt.savefig(dir_root+'IMAGES/Vsini_CCF.pdf')

    plt.figure('vsin3')
    plt.subplot(3,1,3)
    plt.tick_params(top=True,direction='inout')
    plt.axhline(y=np.mean(samples),color='k',ls=':')
    plt.axhline(y=np.mean(samples)+np.std(samples),color='k',ls='-.')
    plt.axhline(y=np.mean(samples)-np.std(samples),color='k',ls='-.')
    plt.xlabel('Jdb - 2,400,000 [days]')
    plt.ylabel(r'$v$ $\sin$ $i$ [km/s]')
    plt.subplots_adjust(hspace=0,top=0.96,right=0.83)

    plt.savefig(dir_root+'IMAGES/Vsini_CCF_sts.pdf')

    samples = np.random.choice(samples,99999)
    samples_table = pd.read_csv(dir_root+'WORKSPACE/Analyse_samples.csv',index_col=0)
    samples_table['vsini'] = samples
    samples_table.to_csv(dir_root+'WORKSPACE/Analyse_samples.csv')

    plt.figure('dust')
    infos = plt.hist(samples,bins=100,density=True,histtype='step',cumulative=True)
    plt.close('dust')
    infos = myc.tableXY(0.5*(infos[1][1:]+infos[1][:-1]),infos[0],0*infos[0])
    vsini_cdf['ALL'] = infos

    return samples

def yarara_vsini(dir_root, Prot=None, Rs=None):

    vsun = 1.87 ; psun = 27.5

    vmax_veq = 2*pd.read_pickle(glob.glob(dir_root+'STAR_INFO/Stell*')[0])['FWHM']['G2']
    samples_table = pd.read_csv(dir_root+'WORKSPACE/Analyse_samples.csv',index_col=0)

    sample_Rs = np.array(samples_table['rs'])
    sample_vsini = np.array(samples_table['vsini'])

    if Rs is not None:
        sample_Rs = np.random.randn(99999)*0.01+Rs #1% radius uncertainty

    sample_prot90 = psun*sample_Rs/(sample_vsini/vsun) 

    p90m = [
        np.nanpercentile(sample_prot90,50),
        np.nanpercentile(sample_prot90,84)-np.nanpercentile(sample_prot90,50),
        np.nanpercentile(sample_prot90,50)-np.nanpercentile(sample_prot90,16)]

    print(' [INFO] Prot (if i=90) estimated = %.2f [%.2f - %.2f] days '%(p90m[0],p90m[0]-p90m[1],p90m[0]+p90m[2]))

    rm = [
        np.nanpercentile(sample_Rs,50),
        np.nanpercentile(sample_Rs,84)-np.nanpercentile(sample_Rs,50),
        np.nanpercentile(sample_Rs,50)-np.nanpercentile(sample_Rs,16)]

    vm = [
        np.nanpercentile(sample_vsini,50),
        np.nanpercentile(sample_vsini,84)-np.nanpercentile(sample_vsini,50),
        np.nanpercentile(sample_vsini,50)-np.nanpercentile(sample_vsini,16)]

    print(' [INFO] vsini = %.2f [%.2f - %.2f] km/s'%(vm[0],vm[0]-vm[1],vm[0]+vm[2]))

    plt.figure('inclination',figsize=(18,5))
    plt.subplot(1,4,1)
    plt.title(r'Rs = %.2f$^{+%.2f}_{-%.2f}$ Rs'%(rm[0],rm[1],rm[2]))
    pby,pbx = np.histogram(sample_Rs,bins=np.arange(0,2,0.025),density=True)
    pbx = 0.5*(pbx[1:]+pbx[0:-1])
    plt.fill_between(pbx,pby,alpha=0.2)
    plt.plot(pbx,pby)
    plt.xlim(0,2)
    plt.ylim(0,None)
    plt.xlabel(r'Radius [Rs]')

    plt.subplot(1,4,2)
    plt.title(r'v $\sin$ i = %.2f$^{+%.2f}_{-%.2f}$ km/s'%(vm[0],vm[1],vm[2]))
    pby,pbx = np.histogram(sample_vsini,bins=np.arange(0,vmax_veq,vmax_veq/200),density=True)
    pbx = 0.5*(pbx[1:]+pbx[0:-1])
    plt.fill_between(pbx,pby,alpha=0.2)
    plt.plot(pbx,pby)
    plt.axvline(x=vm[0],ls=':',color='k')
    if vm[0]>4:
        plt.xlim(vm[0]*0.5,vm[0]*1.5)
    else:
        plt.xlim(0,7)
    plt.ylim(0,None)
    plt.tick_params(top=True,bottom=True,direction='inout',which='both')
    plt.xlabel(r'v $\sin$ i [km/s]')

    sample_sini = np.sqrt(1-np.random.rand(99999)**2) # np.sin of np.arccos
    sample_prot = np.ravel(sample_prot90*sample_sini)

    if Prot is not None:
        sample_prot = np.random.randn(99999)*(0.10*Prot)+Prot #10% prot uncertainty

    plt.subplot(1,4,3)
    plt.title(r'$P_{90} = %.1f^{+%.1f}_{-%.1f}$ days'%(p90m[0],p90m[1],p90m[2]))

    pby,pbx = np.histogram(sample_prot,bins=np.linspace(0,100,100),density=True)
    pbx = 0.5*(pbx[1:]+pbx[0:-1])
    plt.fill_between(pbx,pby,alpha=0.2,color='k')
    plt.plot(pbx,pby,color='k')

    pby,pbx = np.histogram(sample_prot90,bins=np.linspace(0,100,100),density=True)
    pbx = 0.5*(pbx[1:]+pbx[0:-1])
    plt.fill_between(pbx,pby,alpha=0.2)
    plt.plot(pbx,pby)
    plt.axvline(x=p90m[0],ls=':',color='k')
    if Prot is not None:
        plt.axvline(x=Prot,color='k',ls='-',label=r'$P_{rot}$=%.1f days'%(Prot))
        plt.legend()

    plt.xlim(1,100)
    plt.ylim(0,None)
    plt.xscale('log')
    plt.tick_params(top=True,bottom=True,direction='inout')
    plt.xlabel(r'$P_{rot}$ [days]')

    sample_sininc = np.ravel((sample_vsini/vsun)*(sample_prot/psun)/sample_Rs)
    
    I = np.arcsin(np.nanpercentile(sample_sininc,50))*180/np.pi
    Ii =  np.arcsin(np.nanpercentile(sample_sininc,16))*180/np.pi
    Is = np.arcsin(np.nanpercentile(sample_sininc,84))*180/np.pi

    I = [I,90][int(I!=I)]
    Ii = [Ii,90][int(Ii!=Ii)]
    Is = [Is,90][int(Is!=Is)]

    print(' [INFO] Inclination estimated = %.0f [%.0f - %.0f] degree'%(I,Ii,Is))

    plt.subplot(1,4,4)
    plt.title(r'$i = %.0f^{+%.0f}_{-%.0f}$ [°]'%(I,Is-I,I-Ii))

    iby,ibx = np.histogram(sample_sininc,bins=np.arange(0,1,0.01),density=True)
    ibx = 0.5*(ibx[1:]+ibx[0:-1])
    plt.fill_between(ibx,iby,alpha=0.2)
    plt.plot(ibx,iby,label='[%.0f - %.0f - %.0f]'%(Ii,I,Is))
    plt.axvline(x=np.sin(I*np.pi/180),color='k',ls=':')
    plt.tick_params(top=True,bottom=True,direction='inout',which='both')
    plt.xlim(0,1)
    plt.ylim(0,None)
    plt.xlabel(r'$\sin i$ []')
    plt.subplots_adjust(left=0.05,right=0.97)
    plt.savefig(dir_root+'IMAGES/Vsini_inclination.pdf')

def yarara_activity_index(files, rv_sys, shift_rv, material=None, sub_dico='matching_diff'):

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
    
    grid, flux, err_flux = import_sts(files, rv_shift=shift_rv, err=False, sub_dico=sub_dico)

    all_prox_names = np.array(all_proxies)[:,4]
    proxy_found = ((np.array(all_proxies)[:,0]-np.nanmin(grid))>0)&((np.nanmax(grid))>0)
    all_proxies = list(np.array(all_proxies)[proxy_found])

    flux_ref = np.median(flux,axis=0)
    ratio = flux/(flux_ref+1e-6)

    flux_err = myf.mad(flux - flux_ref,axis=1)
    err_flux = np.ones(len(grid))*flux_err[:,np.newaxis]

    if material is not None:
        flux *= np.array(material['correction_factor'])
        err_flux *= np.array(material['correction_factor'])
        flux_ref *= np.array(material['correction_factor'])

    def find_proxy(vec):
        center = myf.doppler_r(vec[0],rv_sys*1000)[0]
        left = myf.doppler_r(vec[0]-vec[1],rv_sys*1000)[0]
        right = myf.doppler_r(vec[0]+vec[1],rv_sys*1000)[0]
        
        center_idx_proxy = myf.find_nearest(grid,center)[0]
        left_idx_proxy = myf.find_nearest(grid,left)[0]
        right_idx_proxy = myf.find_nearest(grid,right)[0]

        left = myf.doppler_r(vec[0]-vec[3],rv_sys*1000)[0]
        right = myf.doppler_r(vec[0]+vec[3],rv_sys*1000)[0]

        left_idx_cont = myf.find_nearest(grid,left)[0]
        right_idx_cont = myf.find_nearest(grid,right)[0]            
        
        return int(center_idx_proxy), int(left_idx_proxy), int(right_idx_proxy), int(left_idx_cont), int(right_idx_cont)
    
    def extract_proxy(vec):
        c, l, r, l_cont, r_cont = find_proxy(vec)            
        continuum=1
        if r!=l:
            r+=1
        if l_cont!=l:
            r_cont+=1
            continuum = np.hstack([ratio[:,l_cont:l],ratio[:,r:r_cont]]) 
            continuum = np.nanmedian(continuum,axis=1)
            continuum[np.isnan(continuum)] = 1
            continuum[continuum==0] = 1
        proxy = np.sum(flux[:,l:r],axis=1)
        proxy_std = np.sum((err_flux[:,l:r])**2,axis=1)
        proxy_std = np.sqrt(proxy_std)
        norm_proxy = (r - l)
        
        proxy/=continuum
        proxy_std/=continuum
        
        if norm_proxy:
            proxy /= norm_proxy
            proxy_std /= norm_proxy      
            return proxy, proxy_std, l, r
        else:
            return 0*proxy, 0*proxy_std, l, r

    save = {'null':0}
    mask_activity = np.zeros(len(grid))
    for p in all_proxies:
        proxy, proxy_std, l, r= extract_proxy(p)
        mask_activity[l:r] = 1    
        save[p[4]] = proxy
        save[p[4]+'_std'] = proxy_std
    del save['null']
    
    for n in all_prox_names:
        if n not in save.keys():
            save[n] = np.zeros(len(files))
            save[n+'_std'] = np.zeros(len(files))
        
    del ratio
    
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
    tab['filename'] = files

    #teff from Ha and NaD EW
    C = np.nanpercentile(save['NaDC'],50)
    X = np.nanpercentile(save['Ha'],50)
    Y = np.nanpercentile(save['NaD'],50)
    Z = np.nanpercentile(save['MgI'],50)
    H = np.nanpercentile(save['Hb'],50)
    CT = {'NaDC':C,'Ha':X,'NaD':Y,'MgI':Z,'Hb':H}
    return tab, CT, mask_activity


def yarara_correct_continuum_absorption(dir_root, master, star_info):
    
    myf.print_box('\n---- RECIPE : CORRECT ABSORPTION CONTINUUM ----\n')

    ins = dir_root.split('/')[-2].split('_')[0]
    rv_sys = star_info['Rv_sys']['SNAKY']
    feh = star_info['FeH']['SNAKY']
    model = star_info['stellar_template']['SNAKY']

    reject_zones = [[5875,5910]]
    force_zones = [[3916.5,3918.5],[3923,3926],[3927.5,3929.5],[3931,3933],[3932.1,3932.6],[3935.4,3935.8],[3936,3937],[3937.5,3939],[3940,3943],
                    [3958,3960],[3962.5,3964.5],[3965.5,3967],[3966.9, 3967.4],[3969.9, 3970.4],[3971.5,3972],[3972.5,3975],[3980,3980.5],[3982,3984],
                    ] 

    rassine_zones = np.array(myv.rassine_continuum)

    reject_zones = [np.round(myf.doppler_r(i,rv_sys*1000)[0],1) for i in reject_zones]
    force_zones = [np.round(myf.doppler_r(i,rv_sys*1000)[0],1) for i in force_zones]
    rassine_zones = np.round(myf.doppler_r(rassine_zones,rv_sys*1000)[0],2)

    grid = master.x
    master.y[-100:] = 1 #issues on the border right

    parameter = '_'.join(model.split('_')[1:])
    model = model.split('_')[0]
        
    print(' Model selected : %s (%s)'%(model,parameter))

    teff = float(parameter.split('T')[-1].split('_')[0])
    logg = float(parameter.split('g')[-1].split('_')[0])    
    template = import_stellar_template(teff,feh=0.0,logg=logg,model='ATLAS',rv_sys=rv_sys)

    template.interpolate(new_grid=grid,replace=True,method='cubic',interpolate_x=False)
    template.y[template.y>1] = 1
    template.y[template.y<0] = 0

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
    print('\n [INFO] Resolution found R=%.0f \n'%(resolution))

    template_flux = myf.instrBroadGaussFast(template.x,template.y,resolution,maxsig=5.0)
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
    plt.plot(template.x, template_flux, color='r',label='Template (%s, Teff = %s, log(g) = %s)'%(model,parameter[0][1:],parameter[1][1:]))       
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

    if ins=='NEID':
        
        template_empi = import_stellar_template(teff,logg=logg,feh=feh,model='SNAKY',rv_sys=rv_sys)
        template_empi.interpolate(master.x,method='linear',fill_value=np.nan)

        s1 = master.y*correction
        s2 = template_empi.y        

        ratio = s2/s1
        ratio[ratio==0] = 1
        extra_correction = myf.smooth(ratio,100)
        extra_correction[extra_correction!=extra_correction] = 1

        correction = correction*extra_correction

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
    plt.savefig(dir_root+'IMAGES/Correction_absolute_continuum.png')
    
    return template_flux, correction

def yarara_instrumental_resolution(dir_root, files, shift_rv, berv, sub_dico='matching_diff'):
    myf.print_box('\n---- RECIPE : EXTRACTION INSTRUMENTAL RESOLUTION ----\n')

    grid, flux, err_flux = import_sts(files, rv_shift=shift_rv, err=False, sub_dico=sub_dico)    
    berv_mad = myf.mad(berv)

    if berv_mad>3:
        print('\n [INFO] BERV MAD = %.1f km/s'%(berv_mad))
        flux_ref = np.nanmedian(flux,axis=0)

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
        template.interpolate(new_grid=grid,method='linear')        
        flux_ref = template.y

    flux_ref[flux_ref<=0] = 1
    flux_ref[flux_ref>1] = 1

    flux = flux/(flux_ref+1e-8)
    for i in tqdm(np.arange(len(files))):
        f = myc.tableXY(myf.doppler_r(grid,berv[i]*1000)[1],flux[i],0*grid)
        f.interpolate(new_grid=grid,method='linear')
        flux[i] = f.y

    ccf_output = yarara_ccf(dir_root, files, 0, 6, 2.0, 'mask_telluric_o2', spectra=(grid,flux,err_flux), debug=False, wave_max=6800)
    fwhm_ins = ccf_output['fwhm'].y
    FWHM_ins = np.nanmedian(fwhm_ins)
    print('\n [INFO] Instrumental resolution measured by O2 lines = %.1f km/s \n'%(FWHM_ins))
    calib = myc.tableXY([1, 2, 3, 4, 5, 6, 7, 8, 9],[299792,149896,99931,74948,59958,49965,42828,37474,33310])
    calib.interpolate(new_grid=np.array([FWHM_ins]),method='cubic',fill_value=np.nan)
    print(' [INFO] Estimate intrumental resolution = %.0f'%(np.round(calib.y[0],-3)))

    fwhm_ins[fwhm_ins<1] = np.nan
    fwhm_ins[fwhm_ins>10] = np.nan
    
    return fwhm_ins

mhk_c1 = -4.04840205e+01        #calibration with RHK DRS
mhk_c2 = 3927259.0994665725
def mhk_rhk(mhk):
    mhk = np.array(mhk)
    mhk[mhk<-40] = -40
    rhk = np.array(np.log10((mhk-mhk_c1)/mhk_c2))    
    return rhk

def yarara_activity_mhk(dir_root, files, rv_sys, shift_rv, teff, material, proxy, sub_dico='matching_diff'):
    
    myf.print_box('\n---- RECIPE : NEW MHK EXTRACTION ----\n')

    jdb = get_jdb(files,dir_root)
    photosphere = pd.read_pickle(root+'/Python/Material_snaky/Photospheric_profiles_V.p')
    chromosphere = myf.touch_pickle(root+'/Python/Material_snaky/Chromospheric_profiles_V.p')
    
    grid, flux, err_flux = import_sts(files, rv_shift=shift_rv, err=False, sub_dico=sub_dico)

    liste_proxy = [myv.Ca2K,myv.Ca2H]

    save = {}
    for n,l in enumerate(liste_proxy):
        unit_filling = {'CaIIK':0.14,'CaIIH':0.14*0.90}[l[-1]]
        deg_poly = {'CaIIK':0,'CaIIH':0,}[l[-1]]
        name = {'CaIIK':'CaII','CaIIH':'CaII'}[l[-1]]
        nb_line = {'CaII':2}[name]

        print('\n [INFO] Analysis of %s(%s)'%(name,l[-1]))

        fig = plt.figure(name,figsize=(14,8))
        gs = fig.add_gridspec(3, nb_line)
        ax = []
        for i in range(3): 
            ax.append(fig.add_subplot(gs[i,n]))

        temp_correction = myc.tableXY(np.array(material['wave']),np.array(material['correction_factor']))
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
        db.rv_shift(rv_sys,fill_value=np.nan,x_grid=line_wave,replace=False)
        quiet = db.shifted.copy()
        quiet.interpolated = quiet.copy()
        quiet.y_interp = quiet.y
        db = db.shifted
        db.y[db.x<(np.mean(db.x)-1.00)] = np.nan
        db.y[db.x>(np.mean(db.x)+1.00)] = np.nan

        db_E1 = chromosphere[l[-1]]
        loc = myf.find_nearest(db_E1['teff'],teff)[0][0]
        E1 = myc.tableXY(db_E1['vel'],db_E1['model'][loc])
        E1.interpolate(new_grid=wave_vel,replace=True,method='linear')
        E1.y[np.abs(E1.x)>35] = 0
        E1.null()
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
            ref = np.mean(line,axis=0)
            ref_std = 0*ref+0.01

        plt.axes(ax[0])
        mat = myc.table(line.copy())
        plt.title(l[-1],fontsize=16)

        quiet.interpolate(new_grid=line_wave,replace=False)
        calib = myc.tableXY(quiet.y_interp[mask_activity],ref[mask_activity])
        index_vec = np.arange(len(ref))[mask_activity]
        index_vec = index_vec[(~np.isnan(calib.x))&(~np.isnan(calib.y))]

        calib.yerr = calib.yerr*0+ref_std[mask_activity]
        calib.supress_nan()
        for j in range(3):
            calib.fit_line(recenter=False)
            mask = myf.rm_outliers(calib.y-calib.x*calib.lin_slope_w,m=2,kind='mad')[0]
            calib.masked(mask)
            index_vec = index_vec[mask]

        index_vec = np.in1d(np.arange(len(ref)),index_vec)
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
        if len(files)>5:
            db.y[np.isnan(db.y)] = np.median(mat.table,axis=0)[np.isnan(db.y)]
        else:
            db.y[np.isnan(db.y)] = quiet.y_interp[np.isnan(db.y)]
            db.y[np.isnan(db.y)] = np.median(mat.table,axis=0)[np.isnan(db.y)]
        
        plt.plot(3e5*(db.x-center)/center,db.y,color='C2',ls='-',lw=2,label=r'$I_{Q}$($\lambda$,%.0fK)'%(teff))
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

        if len(files)>5:
            uncertainties = mat2.table-np.median(mat2.table,axis=0)
        else:
            uncertainties = mat2.table-quiet.y_interp

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
        plt.savefig(dir_root+'IMAGES/Activity_profiles_%s_model_%s.pdf'%(l[-1],fmodel))
        plt.figure(name,figsize=(14,8))

        index_extracted = mat3.coeff_fitted[:,0]*100
        index_extracted_std = (mat3.coeff_fitted_std[:,0]*100 + offset_std)
        med_precision = np.nanmedian(index_extracted_std)*conversion_filling
        med_accuracy = np.nanmedian(index_extracted_std)*conversion_filling

        print(' [INFO] Med flux uncertainties = %.2f'%(med_err)+'%')
        print(' [INFO] Med filling uncertainties (precision) = %.2f'%(med_precision)+'%')
        print(' [INFO] Med filling uncertainties (accuracy) = %.2f'%(med_accuracy)+'%')
        
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
        plt.savefig(dir_root+'IMAGES/Activity_profiles_%s_%s.pdf'%(name,fmodel))

    #CaII

    index1 =  save['CaIIK']['index']
    index1_std = save['CaIIK']['index_std']

    index2 =  save['CaIIH']['index']
    index2_std = save['CaIIH']['index_std']

    index = (index1*0.5+index2*0.5)/(0.5+0.5)
    index_std = np.sqrt(1/(1/index1_std**2+1/index2_std**2))

    print('\n [INFO] M-index = %.2f +/- %.2f'%(np.median(index),np.median(index_std))+'%')

    save['CaII'] = {}
    save['CaII']['index'] = index
    save['CaII']['index_std'] = index_std
    save['CaII']['snr_core'] = 0.5*(save['CaIIK']['snr_core']+save['CaIIH']['snr_core'])*np.sqrt(2)

    mhk = [np.nanpercentile(save['CaII']['index'],i) for i in [16,50,86]]
    print('\n [INFO] M-index = %.2f +/- %.2f [%.2f -> %.2f] \n'%(mhk[1],np.median(index_std),mhk[0],mhk[2]))

    dico = {'filename':files}
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
    plt.savefig(dir_root+'IMAGES/MHK_RHK.pdf')
    
    return dico, RHK_mean, MHK_mean
