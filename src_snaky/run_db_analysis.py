import os 
from matplotlib.pyplot import table
import numpy as np 
import matplotlib.pylab as plt
import pandas as pd
import pickle
from tqdm import tqdm
import glob as glob
from colorama import Fore

from . import snaky_functions as myf
from . import snaky_classes as myc
from . import snaky_variables as myv
from . import snaky_main as mym
import src_snaky.snaky_utilities as snaky_util

version = mym.__version__
MATERIAL_DIR = myv.MATERIAL_DIR

def check_snaky_processing(output_dir,instrument='*'):

    os.makedirs(output_dir+'/database', exist_ok=True)
    instruments = glob.glob(output_dir+'/*/data/s1d/'+instrument+'/RAW')
    instruments = np.unique([i.split('/')[-2] for i in instruments])
    extract = []
    for instrument in instruments:
        print('\n[INFO] Summary for the instrument : %s \n'%(instrument))
        all_dir = glob.glob(output_dir+'/*/data/s1d/'+instrument+'/RAW')
        stars = [d.split('/data')[0].split('/')[-1] for d in all_dir]
        ins = [d.split('/RAW')[0].split('/')[-1] for d in all_dir]
        code = [i+'_'+j for i,j in zip(stars,ins)]

        all_info = pd.DataFrame(np.array([code,stars,ins]).T,columns=['code','star','ins'])

        kws = [
            'force_pre',
            'force_summary',
            'force_rvsys',
            'force_ccf',
            'force_master',
            'force_atmos',
            'force_resolution',
            'force_vsini',
            'force_abs_continuum',
            'force_activity',
            'force_mhk',
            'force_spectroscopy',
            'force_magcycle',
            ]
        
        for kw in kws:
            all_files2 = glob.glob(output_dir+'/*/data/s1d/'+instrument+'/REDUCTION_INFO/'+kw+'.txt')
            stars2 = [d.split('/data')[0].split('/')[-1] for d in all_files2]
            ins2 = [d.split('/REDUCTION_INFO')[0].split('/')[-1] for d in all_files2]
            code2 = [i+'_'+j for i,j in zip(stars2,ins2)]
            mask = np.in1d(np.array(all_info['code']),np.array(code2))
            all_info[kw.split('_')[1][0:5]] = 'XXXX'
            all_info.loc[mask,kw.split('_')[1][0:5]] = ''
        all_info = all_info.sort_values(by='code').reset_index(drop=True)
        nb_stars = len(all_info)
        stat = [nb_stars]
        print('total = %.0f'%(nb_stars))
        save = []
        for kw in kws:
            nb = int(nb_stars-np.sum(all_info[kw.split('_')[1][0:5]]=='XXXX'))
            percentage = 100*nb/(nb_stars+1e-6)
            if percentage>99:
                print(Fore.GREEN+'%s = %.0f (%.1f%%)'%(kw,nb,percentage)+Fore.RESET)
            else:
                print(Fore.YELLOW+'%s = %.0f (%.1f%%)'%(kw,nb,percentage)+Fore.RESET)
            save.append(nb)
        extract.append([instrument]+[nb_stars]+save)
        print('\n[INFO] File DB created: '+output_dir+'/database/Snaky_processing_db_'+instrument.replace('*','')+'.csv')
        all_info.to_csv(output_dir+'/database/Snaky_processing_db_'+instrument.replace('*','')+'.csv')
    new_name = [k.split('_')[1] for k in kws]
    extract = pd.DataFrame(extract,columns=['ins','Ntot']+new_name)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')  # remove axes
    table = ax.table(cellText=extract.values,colLabels=extract.columns,loc='center',cellLoc='center',colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(extract.columns))))
    plt.tight_layout()
    plt.savefig(output_dir+'/database/Summary_processing.png')

    for k in new_name:
        extract[k] = (extract[k]/extract['Ntot']*100).astype('int')

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')  # remove axes
    table = ax.table(cellText=extract.values,colLabels=extract.columns,loc='center',cellLoc='center',colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(extract.columns))))
    plt.tight_layout()
    plt.savefig(output_dir+'/database/Summary_processing_percentage.png')
    

def check_comp_rqm(output_dir,instrument='*'):
    os.makedirs(output_dir+'/database/processing_stat_%s'%(version), exist_ok=True)
    instruments = glob.glob(output_dir+'/*/data/s1d/'+instrument+'/REDUCTION_INFO/Time_info*_N*_GBP*_GBT*.png')
    instruments = list(np.unique([i.split('/')[-3] for i in instruments]))

    n = np.arange(0,5000)
    x2 = 4000 ; y1 = 60 ; y2 = 50 ; perc = None
    sbatch = [[1,150,300,600,2500],[5,5,8,15,50]]
    if len(instruments):
        instruments.append('*')
        for instrument in instruments[::-1]:
            all_dir = glob.glob(output_dir+'/*/data/s1d/'+instrument+'/REDUCTION_INFO/Time_info*_N*_GBP*_GBT*.png')
            
            stars = [d.split('/data')[0].split('/')[-1] for d in all_dir]
            ins = [d.split('/REDUCTION_INFO')[0].split('/')[-1] for d in all_dir]
            code = np.unique([s+'_'+i for s,i in zip(stars,ins)])
            filenames = [d.split('/')[-1] for d in all_dir]
            N = np.array([f.split('_N')[-1].split('_')[0] for f in filenames]).astype('int')
            GBP = np.array([f.split('_GBP')[-1].split('_')[0] for f in filenames]).astype('float')
            GBT = np.array([f.split('_GBT')[-1][0:-4] for f in filenames]).astype('float')
            exec_time = [f.split('_TIME')[-1].split('_')[0] for f in filenames]
            exec_time_min = np.array([t.split('m')[0] for t in exec_time]).astype('int')
            exec_time_sec = np.array([t.split('m')[1].split('s')[0] for t in exec_time]).astype('float')
            total_time = exec_time_min+exec_time_sec/60

            fig = plt.figure(instrument,figsize=(16,6))
            plt.subplot(1,3,3) ; plt.ylabel('CDF') ; plt.xlabel('Dataset Size N []') ; plt.title('Sample datasets')
            plt.tick_params(top=True,right=True,direction='inout',which='both')
            a,b = np.histogram(N,bins=np.arange(0.5,3000.5)) ; b = 0.5*(b[1:]+b[:-1]) ; a = np.cumsum(a) ; a= a/np.max(a)
            plt.plot(b,a,color='k')
            index = myf.find_nearest(b,np.array(sbatch[0]))[0]
            perc = a[index]
            plt.scatter(np.array(sbatch[0]),perc,color='k',marker='o')
            plt.grid()
            if perc is not None:
                for p in perc:
                    plt.axvline(x=p,ls='-',lw=1,color='k')
            plt.xlim(1,4000)
            plt.xscale('log')
            plt.subplot(1,3,1) ; plt.title('Computational time') ; ax = plt.gca(); plt.xlabel('Dataset size N []') ; plt.ylabel('Execution time [min]')
            plt.tick_params(top=True,right=True,direction='inout',which='both')
            if x2 is None:
                x2 = 4000#np.max(N)*1.1
                y1 = np.max(total_time)*1.2
            plt.xscale('log') ; plt.yscale('log')            
            plt.grid()
            if perc is not None:
                for p in perc:
                    plt.axvline(x=p,ls='-',lw=1,color='k')
            plt.scatter(N,total_time,color='k',alpha=0.8)

            plt.xlim(1,x2) ; plt.ylim(0.6,y1)
            plt.subplot(1,3,2,sharex=ax) ; plt.title('Memory') ; plt.xlabel('Dataset size N []') ; plt.ylabel('RAM required [Gb]')
            plt.tick_params(top=True,right=True,direction='inout',which='both')
            plt.scatter(N,GBP,color='C0',alpha=0.8)
            plt.scatter(N,GBT,color='C1',alpha=0.8)
            plt.plot([1,40,250,700,2500],[7,7,10,20,200],ls='-.',color='gray',label='sbatch config',marker='x')
            plt.plot(sbatch[0],sbatch[1],ls='-.',color='k',label='sbatch config (new)',marker='o')
            plt.plot(n,np.max(np.array([[1.35+n*0.0134],[np.ones(len(n))*3.5]]),axis=0)[0],color='r')
            if y2 is None:
                y2 = np.max(GBP)*1.4
            plt.xlim(1,x2) ; plt.ylim(0.5,y2)
            plt.yscale('log')
            plt.legend()
            plt.grid()
            if perc is not None:
                for p in perc:
                    plt.axvline(x=p,ls='-',lw=1,color='k')
            plt.subplots_adjust(left=0.07,right=0.96)
            if instrument=='*':
                instrument='ALL'
            fig.suptitle('SNAKY (%s) with instrument = %s (Number of datasets = %.0f)'%(version,instrument,len(code)))
            filename = output_dir+'/database/processing_stat_%s/Processing_stat_%s.png'%(version,instrument)
            plt.savefig(filename)
            plt.close(instrument)
            print('Figure created: %s'%(filename))

def create_snaky_db(output_dir, filename='All_stars', stars=['*'], branch=None):
    os.makedirs(output_dir+'/database', exist_ok=True)

    root = '/'.join(output_dir.split('/')[:-1])
    if branch is None:
        branch = output_dir.split('/')[-1]

    sub_dico = 'matching_diff'

    files = []
    for s in stars:
        files.append(glob.glob(root+'/'+branch+'/'+s+'/data/s1d/*/STAR_INFO/Stellar_info*.p'))
    files = np.sort(np.hstack(files))
    files = np.sort(np.unique(files))

    ntot = len(files)

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
        processing = {'matching_diff':'YV0','matching_instrument':'YVA','matching_mad':'YV1'}[sub_dico] 
        code = star+'_'+spectro+'_'+drs+'_'+pipeline

        ra = np.round(myf.get_info_lvl2(info,'Ra','SNAKY'),6)
        dec = np.round(myf.get_info_lvl2(info,'Dec','SNAKY'),6)

        teff = np.round(myf.get_info_lvl2(info,'Teff',pipeline.upper()),0)
        logg = np.round(myf.get_info_lvl2(info,'Log_g',pipeline.upper()),2)
        feh = np.round(myf.get_info_lvl2(info,'FeH',pipeline.upper()),2)
        rhk = np.round(myf.get_info_lvl2(info,'RHK',pipeline.upper()),2)
        mhk = np.round(myf.get_info_lvl2(info,'MHK',pipeline.upper()),1)
        rv_sys = np.round(myf.get_info_lvl2(info,'Rv_sys',pipeline.upper()),2)
        sb1 = np.round(myf.get_info_lvl2(info,'SB1','SNAKY'),0)
        sb2 = np.round(myf.get_info_lvl2(info,'SB2','SNAKY'),0)

        fwhm_ins = np.round(myf.get_info_lvl2(info,'FWHM','O2'),2)
        fwhm_ccf1 = np.round(myf.get_info_lvl2(info,'FWHM','G2'),2)
        fwhm_ccf2 = np.round(myf.get_info_lvl2(info,'FWHM','GARFIELD'),2)
        fwhm_ccf3 = np.round(myf.get_info_lvl2(info,'FWHM','KITTY'),2)
        vsini = np.round(myf.get_info_lvl2(info,'Vsini',pipeline.upper()),2)

        f2 = glob.glob(f.replace('STAR_INFO','WORKSPACE/Analyse_samples.csv*&&').split('&&')[0])
        mhk_err = np.nan ; rhk_err = np.nan
        if len(f2):
            samples = pd.read_csv(f2[0],index_col=0)
            if 'mhk' in samples.keys():
                mhk_err = np.round(np.std(samples['mhk']),2)
                rhk_err = np.round(np.std(samples['rhk']),3)
        infos.append([code,star,spectro,drs,pipeline,processing,ra,dec,teff,logg,feh,mhk,mhk_err,rhk,rhk_err,fwhm_ins,fwhm_ccf1,fwhm_ccf2,fwhm_ccf3,vsini,sb1,sb2])

    infos = pd.DataFrame(infos,columns=['code','star','ins','drs','pipeline','yvx','ra','dec','teff','logg','feh','mhk','mhk_err','rhk','rhk_err','fwhm_ins','fwhm_g2','fwhm_garfield','fwhm_kitty','vsini','sb1','sb2'])

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
    infos = infos.reset_index(drop=True)

    nb_processed = len(np.unique(infos['star']))
    print(infos)

    print('\n [INFO] %.0f datasets'%(len(infos)))
    percentage = 100*nb_processed/ntot
    print('\n [INFO] Nb unique stars processed = %.0f (%.0f%%)\n'%(nb_processed,percentage))

    infos.to_csv(output_dir+'/database/'+filename+'_summary_infos.csv')

    return infos

def get_plim(text=''):
    ax = plt.gca()
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    print(ylim,xlim,text)

def update_atmos_databases(output_dir, xbg_model='_phot_1.8.0_sklearn_1.7.2'):
    file_xgb = pd.read_pickle(MATERIAL_DIR+'/xgb_model_yarara_atmos'+xbg_model+'.p')
    means = file_xgb['mean']
    stds = file_xgb['std']

    all_files = glob.glob(str(output_dir)+'/SNAKY_DB_SPECTRA/*/data/s1d/*/STAR_INFO/Stellar_info*.p')
    keys = means.keys()


def compare_atmos_databases(ref_table, output_dir, csv_file=None, reference='SNAKY'):
    
    if csv_file is None:
        csv_files = glob.glob(str(output_dir)+'/*.csv')
        starnames = []
        for c in csv_files:
            c1 = pd.read_csv(c,index_col=0)
            c1 = snaky_util.format(c1) 
            c1['src_file'] = c.split('/')[-1]
            starnames.append(c1)
        starnames = pd.concat(starnames,axis=0)
        ref_table = starnames.drop_duplicates(subset=['starname'])
        csv_file = 'all_csv_files.csv'
    
    output_db_analysis = output_dir+'/DB_ANALYSIS/'+csv_file.split('/')[-1].replace('.csv','')

    os.makedirs(output_db_analysis, exist_ok=True)

    paths = output_dir+'/SNAKY_DB_SPECTRA/*/data/s1d/ALLINS_MERGED/Atmos_all_parameters.csv'
    all_files = glob.glob(paths)
    mask_file = np.in1d([f.split('/data')[0].split('/')[-1] for f in all_files],ref_table['starname'].values)
    all_files = np.array(all_files)[mask_file]

    db = []
    for f in tqdm(all_files):
        starname = f.split(str(output_dir))[1].split('/data')[0].split('/')[-1]
        file = pd.read_csv(f,index_col=0)
        file.index = [starname]*len(file)
        file['starname'] = starname
        db.append(file)
    db = pd.concat(db,axis=0)

    paths_lit = output_dir+'/SNAKY_DB_SPECTRA/*/data/s1d/ALLINS_MERGED/Atmos_literature.csv'
    all_files_lit = glob.glob(paths_lit)
    mask_file = np.in1d([f.split('/data')[0].split('/')[-1] for f in all_files_lit],ref_table['starname'].values)
    all_files_lit = np.array(all_files_lit)[mask_file]

    db_lit = []
    for f in tqdm(all_files_lit):
        starname = f.split(str(output_dir))[1].split('/data')[0].split('/')[-1]
        file = pd.read_csv(f,index_col=0)
        file.index = [starname]*len(file)
        file['starname'] = starname
        db_lit.append(file)
    db_lit = pd.concat(db_lit,axis=0)

    ref_table['REF_ID'] = ref_table.index

    zero_value = {
        'teff':5775,
        'feh':0.0,
        'logg':4.44,
        'ms':1.0,
        'rs':1.0,
        'vsini':0.0,
        'prot':0.0,
        'mhk':0.0,
        'rhk':-4.7,
        'age':0.0}

    missing_values = {
        'teff_std>':300,
        'feh_std>':0.20,
        'feh=':0.0,
        'logg_std>':0.15,
        'logg=':4.0,
        'rs_std>':1.0,
        'rs>':2.0,
        'vsini>':25,
        'vsini_std>':7,
        'rhk>':0,
        'rhk_std>':1.0,
        'mhk_std>':100,
        'ms>':1.5,
        'ms_std>':1.0,
        'age<':0.1}

    all_kw = ['teff','logg','feh','ms','rs','vsini','rhk','age']

    for kw in missing_values:
        k = kw[0:-1]
        condition = kw[-1]
        if condition=='=':
            db.loc[db[k]==missing_values[kw],k] = np.nan
        elif condition=='>':
            db.loc[db[k]>missing_values[kw],k] = np.nan
        elif condition=='<':
            db.loc[db[k]<missing_values[kw],k] = np.nan

    db = db.loc[db['ins']=='ALLINS_MIXTURE']
    db['ins'] = 'SNAKY'

    db = pd.concat([db,db_lit],axis=0)
    for kw in db.keys():
        try:
            db[kw] = np.round(db[kw].astype(float),[3,0][int(kw[0:4]=='teff')])
        except:
            pass
    db.to_csv(output_db_analysis+'/ALL_atmos_parameters.csv',index=False)

    db_ref = db.loc[db['ins']==reference].copy()
    db_aggregated = db.loc[db['ins']!=reference].copy()
    del db_aggregated['ins'] ; del db_ref['ins']
    db_aggregated = db_aggregated.groupby('starname').agg(['mean','std','count'])

    db_aggregated = db_aggregated.reindex(ref_table["starname"])
    db_ref = db_ref.reindex(ref_table["starname"])

    #TBD implement correction to see

    plt.figure(figsize=(18,8))
    plt.subplots_adjust(left=0.05,right=0.97,top=0.95,bottom=0.10,wspace=0.3,hspace=0.3)
    for counter_params, kw in enumerate(all_kw):
        v1 = db_aggregated[kw]['mean'].values
        v2 = db_ref[kw].values
        v1_err = db_aggregated[kw+'_std']['mean'].values
        v1_std = db_aggregated[kw]['std'].values
        v1_err[v1_err!=v1_err] = 0
        v1_err[v1_err<v1_std] = v1_std[v1_err<v1_std]
        v2_err = db_ref[kw+'_std'].values


        diff_err = np.sqrt(v1_err**2+v2_err**2)
        diff = v2-v1

        delta = np.nanmedian(diff)
        med_err = np.nanmedian(diff_err)
        disp1 = myf.mad(diff)

        outliers = np.abs(diff-delta)>7*disp1
        diff[outliers] = np.nan
        diff_err[outliers] = np.nan

        if med_err<disp1:
            disp = np.sqrt(disp1**2-med_err**2)
        else:
            disp = -np.sqrt(med_err**2-disp1**2)

        precision = 0 if kw in ['teff'] else 2
        fmt_str = f'%.{precision}f'

        alpha_data = [0.5,0.2][len(v2)>150]

        bins = {'TEFF': np.arange(3000, 10001, 100),
                'LOGG': np.arange(0, 5.01, 0.1),
                'FEH': np.arange(-2.5, 1.01, 0.25),
                'MS': np.arange(0, 2.01, 0.1),
                'RS': np.arange(0, 2.01, 0.1),
                'RHK': np.arange(-5.5, -2.99, 0.1),
                'AGE': np.arange(0, 10.01, 0.5),
                'VSINI': np.arange(0, 25, 2)}[kw.upper()]

        v2_binned = np.digitize(v2, bins)
        v2_binned_median = np.array([np.nanmedian(v2[v2_binned == i]) for i in range(1, len(bins))])
        diff_binned = [diff[v2_binned == i] for i in range(1, len(bins))]
        diff_binned_median = np.array([np.nanmedian(d) for d in diff_binned])
        diff_binned_std = np.array([myf.mad(d) for d in diff_binned])
        diff_binned_count = np.array([np.sum(~np.isnan(d)) for d in diff_binned])

        diff_binned_median[diff_binned_count<10] = np.nan
        diff_binned_std[diff_binned_count<10] = np.nan
        ref_error = np.nanmedian(diff_err)

        diff_binned_std[diff_binned_std==0] = ref_error/np.sqrt(diff_binned_count[diff_binned_std==0])
        diff_binned_std[diff_binned_std!=diff_binned_std] = ref_error/np.sqrt(diff_binned_count[diff_binned_std!=diff_binned_std])

        deg = 1
        N = np.sum(~np.isnan(diff))
        if np.sum(~np.isnan(diff))>10:
            coeff = np.polyfit(v2[~np.isnan(diff)],diff[~np.isnan(diff)],deg)
            #print(kw,coeff)
            coeff2 = np.polyfit(v2[~np.isnan(diff)],diff[~np.isnan(diff)],3)
            print(kw,[f"{x:.3e}" for x in coeff2])
            disp2 = myf.mad(diff-np.polyval(coeff, v2))
            zeroline = -coeff[1]/coeff[0]
        else:
            coeff = np.zeros(deg+1)
            disp2 = myf.mad(diff)
            zeroline = 0

        plt.subplot(2,4,counter_params+1)
        plt.errorbar(v2,diff,yerr=diff_err,alpha=alpha_data, capsize=0, ls='',marker='o', markersize=3, color='C0',zorder=1)
        plt.errorbar(v2_binned_median, diff_binned_median, yerr=diff_binned_std, alpha=1.0, capsize=0, ls='', marker='o', markersize=6, color='k',zorder=10)
        plt.axhline(y=0,ls='-',color='k',lw=1,alpha=0.4)
        plt.xlabel('%s (%s)'%(kw.upper(),reference),fontsize=12)
        plt.ylabel('%s - MEAN'%(reference),fontsize=12)
        ax = plt.gca()
        xlim = ax.get_xlim()
        y1 = np.polyval(coeff, xlim[0])
        y2 = np.polyval(coeff, xlim[1])
        plt.plot([xlim[0],xlim[1]],[y1,y2],ls='-',color='k',lw=2,zorder=10,label='S = %.3f | Z = %.2f'%(coeff[0],zeroline))
        plt.legend(fontsize=10)
        #plt.title('N = %.0f | '%(N)+r'$\delta$ = ' + fmt_str%(delta) + ' | ' + r'S = ' + '%.2f'%(coeff[0])+'\n ' + r'$\Delta$ = ' + fmt_str%(disp1) + ' | ' + r'$\sigma_{+}$ = ' + fmt_str%(disp),fontsize=10)
        plt.title('N = %.0f | '%(N)+r'$\delta$ = ' + fmt_str%(delta) + r' | $\sigma$ = ' + fmt_str%(disp1),fontsize=12)
    plt.subplots_adjust(bottom=0.10,left=0.16)
    plt.savefig(output_db_analysis+'/ALL_param_%s_with_MEAN.png'%(reference),dpi=200)
        

    source = np.unique(db['ins'])
    db2 = {}
    for s in source:
        df_small = db.loc[db['ins']==s]
        df_small = df_small.reindex(ref_table["starname"])
        db2[s] = df_small
        db2[s]['ins'] = s

    source = np.array(list(db2.keys()))
    source = list(np.setdiff1d(source,[reference]))+[reference]

    statistic = {}
    for counter_params, kw in enumerate(all_kw):
        print('\n[INFO] Analysis of the parameter: %s'%(kw.upper()))

        source2 = source.copy()
        for s in source:
            Nstar = np.sum(~np.isnan(db2[s][kw].values))
            if Nstar==0:
                source2.remove(s)
        i1,i2 = np.triu_indices(len(source2))

        fig = plt.figure(figsize=(18,10))
        fig.suptitle(kw.upper())
        gs = fig.add_gridspec(len(source2), len(source2), hspace=0.0, wspace=0.0)
        mini = np.nanmin(db2[reference][kw])
        maxi = np.nanmax(db2[reference][kw])
        span = maxi-mini
        mini -= 0.10*span ; maxi += 0.10*span

        ax0 = fig.add_subplot(gs[len(source2)-1,0])
        plt.axis('off')

        params = []

        count=0
        for j1,j2 in zip(i1,i2):
            count+=1
            if j1==j2:
                ax = fig.add_subplot(gs[j1, j2],sharex=ax0)
            else:
                ax = fig.add_subplot(gs[j1, j2],sharex=ax0,sharey=ax0)
            
            s1 = source2[j1]
            s2 = source2[j2]
            v1 = db2[s1][kw].values ; v1_err = db2[s1][kw+'_std'].values
            v2 = db2[s2][kw].values ; v2_err = db2[s2][kw+'_std'].values
            N = len((v1+v2)[~np.isnan(v1+v2)])

            offset = np.nan
            rpearson = np.nan
            disp = np.nan
            disp2 = np.nan

            if (N>10) & (j1!=j2):
                offset = np.nanmedian(v1-v2)
                mask = ~np.isnan(v1+v2)
                weights = 1.0 / (v1_err[mask]**2 + v2_err[mask]**2)
                weights[np.isinf(weights)] = 0
                coeff = np.polyfit(v1[mask], v2[mask], 1)
                rpearson = np.corrcoef(v1[mask], v2[mask])[0, 1]
                diff = v1-v2
                v1_err[v1_err!=v1_err] = 0
                v2_err[v2_err!=v2_err] = 0
                diff_err = np.sqrt(v1_err**2 + v2_err**2)
                med_err = np.nanmedian(diff_err)
                disp = myf.mad(diff)
                if med_err<disp:
                    disp = np.sqrt(disp**2-med_err**2)
                else:
                    disp = -np.sqrt(med_err**2-disp**2)

                disp2 = myf.mad(v2[mask]-np.polyval(coeff, v1[mask]))
            else:
                coeff = [1,0]
            params.append([j1,j2,offset,coeff[0],coeff[1],N, rpearson, disp, disp2])

            alpha_data = [0.5,0.2][N>150]
            
            plt.xlim(mini,maxi)
            plt.tick_params(labelleft=False,labelbottom=False,direction='inout',which='both',top=True,right=True)
            if j1==j2:
                plt.xlabel(s1,fontsize=11)
                plt.hist(v1,np.linspace(mini,maxi,30),color='C0',alpha=0.8)
                plt.tick_params(labelbottom=True)
                plt.text(0.05,0.95,'%.0f'%(len(v1[~np.isnan(v1)])),transform=ax.transAxes,fontsize=8, va='top')
            else:
                plt.errorbar(v1, v2, xerr=v1_err, yerr=v2_err, alpha=alpha_data, capsize=0, ls='',marker='o', markersize=3, color='C0')
                plt.plot([mini,maxi],[mini,maxi],ls='-',color='k',lw=1)
                plt.ylim(mini,maxi)
                if N>10:
                    plt.text(0.95,0.05,'%.2f\n%.2f'%(offset,coeff[0]),transform=ax.transAxes,fontsize=8,va='bottom',ha='right',weight='bold')
                    plt.text(0.05,0.95,'%.0f\n%.2f'%(N,rpearson),transform=ax.transAxes,fontsize=8,va='top', ha='left')
        plt.subplots_adjust(wspace=0.0,hspace=0.0,top=0.89,bottom=0.05,right=0.99,left=0.03)

        params = pd.DataFrame(params,columns=['i1','i2','offset','coeff','intercept','N','rpearson','disp','disp2'])
        statistic[kw] = params

        precision = 0 if kw in ['teff'] else 2
        for i in range(len(source2)):
            p = params.loc[(params['i1']==i)|(params['i2']==i)]
            offset, coeff, disp = p['offset'].values, p['coeff'].values, p['disp'].values
            
            offset[np.array(p['i1']==i)] = -offset[np.array(p['i1']==i)]
            coeff[np.array(p['i1']==i)] = 1.0/coeff[np.array(p['i1']==i)]

            delta = np.nanmean(offset)
            slope = np.nanmean(coeff)
            disp = np.nanmean(disp)

            plt.subplot(gs[0,i])
            fmt_str = f'%.{precision}f'
            plt.title(r'$\delta$ = ' + fmt_str%(delta) + '\n' + r'S = ' + '%.2f'%(slope)+ '\n' + r'$\sigma_{+}$ = ' + fmt_str%(disp),fontsize=10)
        plt.savefig(output_db_analysis+'/meta_%s_with_LIT.png'%(kw.upper()),dpi=200)

    plt.figure(figsize=(15,10))
    plt.subplots_adjust(hspace=0.15,wspace=0.0,top=0.90,left=0.08,right=0.96)
    col=0
    for kw in all_kw:

        stat = statistic[kw]

        col+=1
        v1 = db2[reference][kw].values
        v1_err = db2[reference][kw+'_std'].values
        v1_err[v1_err!=v1_err] = 0
        ax = plt.subplot(len(all_kw),len(source),1+(col-1)*len(source))
        if col==1:
            plt.title(reference.replace('+','\n20')+'\n(N=%.0f)'%(len(v1[~np.isnan(v1)])),fontsize=11)
        plt.ylabel(kw.upper())
        plt.tick_params(labelbottom=False, right=True, top=True, direction='inout')

        mini = np.nanmin(db2[reference][kw])
        maxi = np.nanmax(db2[reference][kw])
        span = maxi-mini
        mini -= 0.10*span ; maxi += 0.10*span
        
        plt.hist(v1[~np.isnan(v1)],np.linspace(mini,maxi,30),alpha=0.5,color='C0',orientation='horizontal')

        row=0
        for s in np.setdiff1d(source,[reference]):
            row+=1
            #print(s)
            v2 = db2[s][kw].values
            v2_err = db2[s][kw+'_std'].values
            v2_err[v2_err!=v2_err] = 0
            N = np.sum(v2==v2)

            alpha_data = [0.5,0.2][N>150]

            v_err = np.sqrt(v1_err**2 + v2_err**2)
            plt.subplot(len(all_kw),len(source),1+(col-1)*len(source)+row,sharey=ax)
            plt.tick_params(labelbottom=False,labelleft=False,right=True,top=True,direction='inout')

            plt.errorbar(v2, v1, xerr=v2_err, yerr=v_err, alpha=alpha_data, capsize=0, ls='',marker='o', markersize=3, color='C0')
            if np.sum(~np.isnan(v1+v2))!=0:
                plt.plot([mini,maxi],[mini,maxi],ls='-',color='k',lw=1)
            plt.xlim(mini,maxi) ; plt.ylim(mini,maxi)
            if col==1:
                author = s.split('+')[0]
                fontsize=10
                if len(author)>9:
                    author = author[0:7]+'...'
                    fontsize=10
                if len(s.split('+'))>1:
                    year = s.split('+')[1]
                    year = '20'+year
                else:
                    year = ''
                title = author+'\n'+year+'\n(N=%.0f)'%(N)
                plt.title(title,fontsize=fontsize)
            
        plt.savefig(output_db_analysis+'/analysis_meta_%s.png'%(reference),dpi=200)
            


def analyse_snaky_results(ref_table, output_dir, csv_file, reference, flag_outliers=True, trim_outliers=True, residual=True):

    output_db_analysis = output_dir+'/DB_ANALYSIS/'+csv_file.split('/')[-1].replace('.csv','')

    os.makedirs(output_db_analysis, exist_ok=True)

    paths = output_dir+'/SNAKY_DB_SPECTRA/*/data/s1d/ALLINS_MERGED/Atmos_all_parameters.csv'
    all_files = glob.glob(paths)
    mask_file = np.in1d([f.split('/data')[0].split('/')[-1] for f in all_files],ref_table['starname'].values)
    all_files = np.array(all_files)[mask_file]

    db = []
    for f in tqdm(all_files):
        starname = f.split(str(output_dir))[1].split('/data')[0].split('/')[-1]
        file = pd.read_csv(f,index_col=0)
        file.index = [starname]*len(file)
        file['starname'] = starname
        db.append(file)
    db = pd.concat(db,axis=0)

    paths_lit = output_dir+'/SNAKY_DB_SPECTRA/*/data/s1d/ALLINS_MERGED/Atmos_literature.csv'
    all_files_lit = glob.glob(paths_lit)
    mask_file = np.in1d([f.split('/data')[0].split('/')[-1] for f in all_files_lit],ref_table['starname'].values)
    all_files_lit = np.array(all_files_lit)[mask_file]

    db_lit = []
    for f in tqdm(all_files_lit):
        starname = f.split(str(output_dir))[1].split('/data')[0].split('/')[-1]
        file = pd.read_csv(f,index_col=0)
        file.index = [starname]*len(file)
        file['starname'] = starname
        db_lit.append(file)
    db_lit = pd.concat(db_lit,axis=0)

    ref_source = reference.upper()
    db_lit = db_lit[db.keys()]
    db = pd.concat([db,db_lit.loc[db_lit['ins']==ref_source]],axis=0)

    ref_table['REF_ID'] = ref_table.index

    zero_value = {
        'teff':5775,
        'feh':0.0,
        'logg':4.44,
        'ms':1.0,
        'rs':1.0,
        'vsini':0.0,
        'prot':0.0,
        'mhk':0.0,
        'rhk':-4.7,
        'age':0.0}

    missing_values = {
        'teff_std>':300,
        'feh_std>':0.20,
        'feh=':0.0,
        'logg_std>':0.15,
        'logg=':4.0,
        'rs_std>':1.0,
        'rs>':2.0,
        'vsini>':25,
        'vsini_std>':7,
        'rhk>':0,
        'rhk_std>':1.0,
        'mhk_std>':100,
        'ms>':1.5,
        'ms_std>':1.0}

    for kw in missing_values:
        k = kw[0:-1]
        condition = kw[-1]
        if condition=='=':
            db.loc[db[k]==missing_values[kw],k] = np.nan
        elif condition=='>':
            db.loc[db[k]>missing_values[kw],k] = np.nan
        elif condition=='<':
            db.loc[db[k]<missing_values[kw],k] = np.nan

    source = np.unique(db['ins'])
    db2 = {}
    for s in source:
        df_small = db.loc[db['ins']==s]
        df_small = df_small.reindex(ref_table["starname"])
        db2[s] = df_small
        db2[s]['ins'] = s

    ref = db2[ref_source]
    source = np.array(list(db2.keys())) ; source = source[source!=ref_source]

    plt.figure('fig_all',figsize=(16,8))

    if residual:
        ext = '_residual'
    else:
        ext = ''

    mini = {}
    maxi = {}

    all_kw = ['teff','logg','feh','ms','rs','vsini','rhk','age']
    for counter_params, kw in enumerate(all_kw):
        print('\n[INFO] Analysis of the parameter: %s'%(kw.upper()))

        yref = db2[ref_source][kw].copy() ; yref_err = db2[ref_source][kw+'_std'].copy()
        y1 = db2['ALLINS_MIXTURE'][kw].copy() ; y1_err = db2['ALLINS_MIXTURE'][kw+'_std'].copy()
        y2 = db2['ALLINS_BAYESIAN'][kw].copy() ; y2_err = db2['ALLINS_BAYESIAN'][kw+'_std'].copy()

        yref_err[yref_err!=yref_err] = 0

        delta = y1 - yref
        delta_err = np.sqrt(y1_err**2 + yref_err**2)
        delta_err[delta_err!=delta_err] = y1_err[delta_err!=delta_err]
        index = np.arange(len(db2[ref_source]))

        plt.figure(figsize=(16,8))
        plt.axes([0.06,0.10,0.9,0.2]) ; ax = plt.gca()
        plt.errorbar(index, -delta, yerr=delta_err, fmt='o',label='ALLINS_BAYESIAN')
        plt.axhline(y=0,ls='-',color='k',alpha=0.3,lw=1,zorder=100)
        plt.xlabel('Stars index')
        plt.grid(alpha=0.40,axis='x')
        plt.tick_params(top=True,direction='inout')

        mini_val = np.nanmin(yref) ; maxi_val = np.nanmax(yref)
        if mini_val!=mini_val:
            mini_val = np.nanmin(y2) ; maxi_val = np.nanmax(y2)
        span = maxi_val-mini_val
        mini[kw] = np.round(mini_val-0.1*span,2)
        maxi[kw] = np.round(maxi_val+0.1*span,2)

        diff_mad = myf.mad(delta)
        diff_med = np.nanmedian(delta)

        plt.ylabel(r'$\Delta$ %s'%(kw.upper()))
        plt.axhline(y=diff_med,ls='--',color='r',alpha=0.5,lw=1,zorder=100,label=r'$\delta$ = %.2f | $\Delta$ = %.2f'%(diff_med,diff_mad))
        plt.legend()

        plt.axes([0.06,0.3,0.9,0.68],sharex=ax)

        plottype='bar'
        if plottype=='errorbar':
            plt.errorbar(index,yref,yerr=yref_err,fmt='o',label='%s'%(ref_source),color='C0',capsize=2)
            plt.errorbar(index,y1,yerr=y1_err,fmt='o',label='ALLINS_MIXTURE',color='k',capsize=0)
            plt.errorbar(index,y2,yerr=y2_err,fmt='o',label='ALLINS_BAYESIAN',color='C1',capsize=2)
        else:
            baseline = zero_value[kw]
            bars = plt.bar(np.arange(len(y1))-0.15,y1-baseline,yerr=y1_err,bottom=baseline,label='ALLINS_MIXTURE',error_kw={'ecolor': '#5a9bd4'},width=0.6)
            plt.bar(np.arange(len(y1))+0.15,yref-baseline, yerr=yref_err, bottom=baseline,alpha=0.7,label='%s'%(ref_source),error_kw={'ecolor': '#b85600'},width=0.6)
            # Add labels at the outermost of the two bars

            if len(ref)<120:
                mask = np.arange(len(y1))
            else:
                outliers = abs(delta).sort_values().dropna()[-10:].keys()
                mask = np.where(np.in1d(np.array(delta.keys()),outliers))[0]

            for i, v1, v1_err, vref, vref_err, name in zip(mask, y1[mask], y1_err[mask], yref[mask],  yref_err[mask], ref['starname'].values[mask]):

                # Outer edges of the two error bars, measured relative to baseline
                y1_edge = v1 + v1_err if v1 >= baseline else v1 - v1_err
                vref_edge = vref + vref_err if vref >= baseline else vref - vref_err
                if vref_edge!=vref_edge:
                    vref_edge = y1_edge

                # Pick whichever value/error bar extends furthest from baseline
                if abs(y1_edge - baseline) >= abs(vref_edge - baseline):
                    ypos = y1_edge
                else:
                    ypos = vref_edge

                offset = 3 if ypos >= baseline else -3
                va = 'bottom' if ypos >= baseline else 'top'

                plt.annotate(
                    name,
                    xy=(i, ypos),
                    xytext=(0, offset),
                    textcoords='offset points',
                    ha='center',
                    va=va,
                    rotation=90,
                    fontsize=8
                )

        plt.legend()
        med = np.nanmedian(y1)

        if plottype=='errorbar':
            diff_mad = np.nanpercentile(delta,95)-np.nanpercentile(delta,5)
            sup3 = np.nanmedian(y2)+5*diff_mad
            inf3 = np.nanmedian(y2)-5*diff_mad
            delta = abs(delta)
            discrepant = delta.dropna().sort_values(ascending=False)[0:int(0.05*len(delta))]
            
            if mini[kw]==mini[kw]:
                plt.ylim(mini[kw],maxi[kw])

            ax = plt.gca()
            for counter,starname in enumerate(db2[ref_source].index):
                r = np.nanmean(db2['ALLINS_MIXTURE'].loc[starname,kw])
                dr = np.nanmean(db2['ALLINS_MIXTURE'].loc[starname,kw+'_std'])
                if r<med:
                    value = r-0.25*span-dr
                    if starname in discrepant.index:
                        plt.annotate(starname, xy=(counter, r - dr - span*0.02), xytext=(counter, mini[kw]), fontsize=6, ha='center', va='bottom', rotation=90, arrowprops=dict(arrowstyle='-', color='k', lw=1, alpha=0.35),)
                else:
                    value = r+0.25*span+dr
                    if starname in discrepant.index:
                        plt.annotate(starname, xy=(counter, r + dr + span*0.02),  xytext=(counter, maxi[kw]), fontsize=6, ha='center', va='top', rotation=90, arrowprops=dict(arrowstyle='-', color='k', lw=1, alpha=0.35),)
        plt.ylabel(kw.upper())
        plt.tick_params(top=True,direction='inout',labelbottom=False)
        plt.xlim(-1,len(ref)+1)
        ax_figure = plt.gca()
        ylim_plot = ax_figure.get_ylim()
        ysup_new = 0.5*(ylim_plot[1]+ylim_plot[0])+1.2*0.5*(ylim_plot[1]-ylim_plot[0])
        yinf_new = 0.5*(ylim_plot[1]+ylim_plot[0])-1.2*0.5*(ylim_plot[1]-ylim_plot[0])
        plt.ylim(yinf_new,ysup_new)
        plt.grid(alpha=0.40,axis='x')
        plt.savefig(output_db_analysis+'/comp_%s.png'%(kw.upper()),dpi=200)
        
        fig = plt.figure(kw,figsize=(15,8))

        max_disp = 0
        axes = {}
        for n, s in enumerate(source):
            for figure in ['fig_all',kw]:
                if figure=='fig_all':
                    plt.figure('fig_all')
                    grid_loc = counter_params
                    n_rows=2
                    n_cols=4
                    s_title = ''

                else:
                    plt.figure(kw)
                    grid_loc = n
                    n_rows = 2
                    n_cols = max(1, int(np.ceil(len(source)/2)))
                    s_title = s

                if (figure=='fig_all')&(s!='ALLINS_MIXTURE'):
                    continue

                if figure not in axes:
                    plt.subplot(n_rows, n_cols, grid_loc+1) ; ax = plt.gca()
                    axes[figure] = ax
                else:
                    plt.subplot(n_rows, n_cols, grid_loc+1, sharex=axes[figure],sharey=axes[figure])

                if (figure=='fig_all')&(s!='ALLINS_MIXTURE'):
                    plt.xlim(mini[kw],maxi[kw])
                    plt.ylim(mini[kw],maxi[kw])

                if n>=len(source)//2-1:
                    if residual:
                        if figure!='fig_all':
                            plt.ylabel('SNAKY - %s'%(ref_source))
                    else:
                        plt.xlabel('SNAKY') ; lim = plt.xlim()            

                if figure=='fig_all':
                    plt.xlabel('%s %s'%(ref_source, kw.upper()))
                    if residual:
                        plt.ylabel('SNAKY - %s'%(ref_source))
                    else:
                        plt.ylabel('SNAKY')

                plt.title(
                    r'%s'%(s_title),
                    fontsize=10
                )

                if n%(len(source)//2)==0:
                    if figure!='fig_all':
                        plt.ylabel('%s %s'%(ref_source, kw.upper()))
                else:
                    if figure!='fig_all':
                        plt.tick_params(labelleft=False)

                if residual:
                    if figure=='fig_all':
                        plt.axhline(y=0,ls='-',color='k',alpha=0.3,lw=1)    
                    else:
                        plt.axvline(x=0,ls='-',color='k',alpha=0.3,lw=1)
                else:
                    plt.plot([mini[kw],maxi[kw]],[mini[kw],maxi[kw]],ls='-',color='k',alpha=0.3,lw=1)

                plt.tick_params(top=True,right=True,direction='inout',which='both')

                ref.loc[ref[kw+'_std']!=ref[kw+'_std'],kw+'_std'] = 0.0

                df = db2[s]
                ref_vector = ref[kw]
                if len(ref_vector.dropna())==0:
                    continue

                diff = df[kw]-ref_vector
                disp = myf.mad(diff)
                max_disp = np.max([disp,max_disp])
                diff_std = np.sqrt(df[kw+'_std']**2+ref[kw+'_std']**2)
                diff_std[diff_std==0] = myf.mad(ref_vector)*0.05
                med_err = np.nanmedian(diff_std)

                if med_err<disp:
                    extra_jitter = np.sqrt(disp**2-med_err**2)
                else:
                    extra_jitter = np.sqrt(med_err**2-disp**2)

                n_sigma = diff/diff_std
                abs_sigma = np.abs(n_sigma)
                percentage_20 = (1-5/len(abs_sigma))*100
                outliers = abs_sigma>np.nanpercentile(abs_sigma, percentage_20)
                if not flag_outliers:
                    outliers = np.zeros(len(diff)).astype('bool')

                if (trim_outliers)&(residual==True):
                    diff[outliers] = 3*disp*np.sign(diff[outliers])

                x = diff+ref_vector*(1-int(residual))
                y = ref_vector
                xerr = diff_std
                yerr = ref[kw+'_std']

                stat = pd.DataFrame(np.array([x,y,xerr,outliers]).T,columns=['diff','ref','diff_std','outliers'])
                stat = stat.dropna()

                if len(stat)>0:
                    stat = stat[stat['outliers']==0]

                if len(stat)>7:
                    weights = 1.0 / stat['diff_std'].values**2
                    coeff = np.polyfit(stat['ref'],stat['diff'],2,w=weights)
                    trend = np.polyval(coeff,stat['ref'])
                    #print('[INFO] Fit coefficients on %s:'%(kw.upper()),coeff)
                    if figure=='fig_all':
                        plt.scatter(stat['ref'],trend,alpha=1.0,color='k',zorder=100,label='S=%.3f | I=%.3f'%(coeff[0],coeff[1]))
                    else:
                        plt.scatter(trend, stat['ref'],alpha=1.0,color='k',marker='.',lw=1,zorder=100,label='S=%.3f | I=%.3f'%(coeff[0],coeff[1]))
                    if not residual:
                        plt.legend(fontsize=8,loc='upper left',framealpha=0.5)
                else:
                    coeff = [1,0]

                trend = np.polyval(coeff,x)
                res = y - trend
                disp2 = myf.mad(res)
                
                if figure=='fig_all':
                    x,y,xerr,yerr = y,x,yerr,xerr

                alpha_data = [0.5,0.2][len(x)>150]

                plt.errorbar(x[~outliers],y[~outliers],xerr=xerr[~outliers],yerr=yerr[~outliers],fmt='o',alpha=alpha_data)

                if np.sum(outliers):
                    plt.errorbar(x[outliers],y[outliers],xerr=xerr[outliers],yerr=yerr[outliers],alpha=1,color='r',fmt='x')

                plt.title(
                    (
                        r'%s' '\n'
                        r'$\delta$ = %.2f' '\n'
                        r'$\Delta_{0}$ = %.2f | '
                        r'$\Delta_{1}$ = %.2f | '
                        r'$\sigma_{+}$ = %.2f'
                    ) % (s_title, np.nanmedian(diff), disp, disp2, extra_jitter),
                    fontsize=10
                )
                
            if (max_disp==max_disp)&(residual):
                plt.xlim(-max_disp*5,max_disp*5)
            else:
                plt.xlim(mini[kw],maxi[kw])
                plt.ylim(mini[kw],maxi[kw])
            plt.subplots_adjust(wspace=0.15,hspace=0.50,left=0.06,right=0.98)
            plt.savefig(output_db_analysis+'/param_%s_with_%s%s.png'%(kw.upper(),ref_source,ext),dpi=200)

    plt.figure('fig_all')
    for counter_params, kw in enumerate(all_kw):
        plt.subplot(2,4,counter_params+1)

        plt.xlim(mini[kw],maxi[kw])
        if not residual:
            plt.ylim(mini[kw],maxi[kw])

    plt.subplots_adjust(wspace=0.30,hspace=0.30,left=0.06,right=0.98,top=0.95,bottom=0.10)
    plt.savefig(output_db_analysis+'/ALL_param_with_%s%s.png'%(ref_source,ext),dpi=200)

    for kw in db2.keys():
        db2[kw]['starname'] = db2[kw].index
        db2[kw]['index_num'] = np.arange(len(db2[kw]))

    db_stacked = pd.concat([db2[kw] for kw in db2.keys()],axis=0)
    db_stacked.index = np.arange(len(db_stacked))
    db_stacked = pd.merge(db_stacked,ref_table[['starname','REF_ID']],on='starname',how='left') 

    export_table = db_stacked.loc[db_stacked['ins']=='ALLINS_MIXTURE']
    export_table.index = np.array(export_table['REF_ID'])
    export_table.to_csv(output_db_analysis+'/SNAKY_final_atmos_parameters.csv')

    return db_stacked


def compare_snaky_atmos(output_dir, stars=['*']):
    all_files = []
    for s in stars:
        files = glob.glob(output_dir+'/'+s+'/data/s1d/*/WORKSPACE/Analyse_samples.csv')
        all_files.append(files)
    all_files = np.hstack(all_files)

    stars = np.sort(np.unique([f.split('/data')[0].split('/')[-1] for f in all_files]))

    for s in tqdm(stars):
        count = -1
        files = glob.glob(output_dir+'/'+s+'/data/s1d/*/WORKSPACE/Analyse_samples.csv')
        plt.figure(figsize=(18,6))
        plt.subplots_adjust(left=0.06,right=0.96,hspace=0.60,top=0.95,bottom=0.15,wspace=0.30)
        for f in files:
            ins = f.split('/WORKSPACE')[0].split('/')[-1]
            code = ins[0]+ins.split('_')[0][-2:]+'_'+ins.split('_')[1]
            count += 1
            table = pd.read_csv(f,index_col=0)
            borders = {'ms':[0,3],'rs':[0,3],'teff':[3000,8000],'logg':[3.5,5.0],'feh':[-1.5,0.5],'vsini':[0,10],'mhk':[-50,200],'rhk':[-6,-4]}
            variables = ['ms','rs','teff','logg','feh','vsini','mhk','rhk']
            save = {kw:[] for kw in variables}
            for j,kw in enumerate(variables):
                if kw in table.keys():
                    plt.subplot(2,4,j+1)
                    plt.boxplot(np.array(table[kw]),positions=[count],showfliers=False,labels=[code],widths=[0.5])
                    plt.ylabel(kw,fontsize=14)
                    plt.xticks(rotation=45,ha='right')
                    save[kw].append(np.array(table[kw]))

        for j,kw in enumerate(variables):
            plt.subplot(2,4,j+1)
            plt.boxplot(np.ravel(save[kw]),positions=[count+2],showfliers=False,labels=['ALL'],widths=[0.5],patch_artist=True,boxprops=dict(facecolor='lightsteelblue',edgecolor='black',linewidth=1.))
            if kw=='teff':
                plt.title('%s = %.0f +/- %.0f'%(kw,np.median(save[kw]),myf.mad(np.ravel(save[kw]))))
            else:
                plt.title('%s = %.2f +/- %.2f'%(kw,np.median(save[kw]),myf.mad(np.ravel(save[kw]))))
            plt.xticks(rotation=90,ha='center')

            plt.savefig(output_dir+'/'+s+'/data/s1d/ALLINS_MERGED/Atmos_all_instrument.pdf')
        if len(stars)>5:
            plt.close('all')


def create_snaky_finch_db(output_dir, filename='All_stars', stars=['*'], infos=None, branch=None):
    if infos is None:
        infos = create_snaky_db(output_dir, filename=filename, stars=stars, branch=branch)
    stars = np.sort(np.unique(infos['star']))

    root = '/'.join(output_dir.split('/')[:-1])
    if branch is None:
        branch = output_dir.split('/')[-1]

    tgrid = 61041+182.5*np.arange(21)
    infos = []
    for s in tqdm(stars):
        if os.path.exists(output_dir+'/'+s+'/data/s1d/ALLINS_MERGED/Pmag_FINCH_info.p'):
            info = pd.read_pickle(output_dir+'/'+s+'/data/s1d/ALLINS_MERGED/Pmag_FINCH_info.p')
            if 'Phase_side' in info.keys():
                info['Phase_pred_side'] = info['Phase_side']
            Pmag_model = pd.read_csv(output_dir+'/'+s+'/data/s1d/ALLINS_MERGED/Finch_MHK_GP_model.csv',index_col=0)
            model = myc.tableXY(Pmag_model['jdb'],Pmag_model['proxy'],Pmag_model['proxy_std'])
            model.interpolate(new_grid=tgrid,method='linear')
            liste = [s,info['Pmag'],info['Kmag_mean'],info['Kmag_amp'],info['Kmag_pred'],info['Phase_pred'],info['Phase_pred_side']]
            liste = liste + list(np.round(model.y,1)) + list(np.round(model.yerr,1))
            infos.append(liste)
    table = pd.DataFrame(infos,columns=['star','Pmag','Kmean','Kamp','Kpred','Lpred','Lside']+['MHK_%.1f'%(i) for i in np.arange(2026,2036.1,0.5)]+['MHK_err_%.1f'%(i) for i in np.arange(2026,2036.1,0.5)])
    table.to_csv(output_dir+'/database/'+filename+'_FINCH_mag.csv')

    infos2 = []
    for s in tqdm(stars):
        if os.path.exists(output_dir+'/'+s+'/data/s1d/ALLINS_MERGED/Finch_MHK.csv'):
            tab = pd.read_csv(output_dir+'/'+s+'/data/s1d/ALLINS_MERGED/Finch_MHK.csv',index_col=0)
            tab['star'] = s
            tab = tab[['star','species','jdb','proxy','proxy_std','qc']]
            infos2.append(tab)
    infos2 = pd.concat(infos2)
    infos2.to_csv(output_dir+'/database/'+filename+'_FINCH.csv')


def create_snaky_rv_db(output_dir,filename='All_stars', stars=['*'], infos=None, anonymous=False, branch=None):
    root = '/'.join(output_dir.split('/')[:-1])
    if branch is None:
        branch = output_dir.split('/')[-1]

    if infos is None:
        infos = create_snaky_db(output_dir, filename=filename, stars=stars, branch=branch)
    stars = np.sort(np.unique(infos['star']))

    infos = [] ; infos_p = []
    for s in tqdm(stars):
        infos2 = []
        plt.figure('rv')
        for f in glob.glob(output_dir+'/'+s+'/data/s1d/*/STAR_INFO/Stellar_info_*p'): 
            info = pd.read_pickle(f)
            star = f.split('/data')[0].split('/')[-1]
            instrument = f.split('/STAR_INFO')[0].split('/')[-1]
            spectro = instrument.split('_')[0]
            drs = instrument.split('_')[1]
            pipeline = f.split('/'+star)[0].split('/')[-1]
            code = star+'_'+spectro+'_'+drs

            try:
                rv_sys = info['Rv_sys']['SNAKY']
                summary = pd.read_csv(f.split('STAR_INFO/')[0]+'WORKSPACE/Analyse_summary.csv',index_col=0)
            except:
                print(' [ERROR] %s has no RV_SYS'%(s))
                summary = {}
            
            if 'ccf_rv_G2' in summary.keys():
                if 'rv_shift' not in summary.keys():
                    summary['rv_shift'] = 0
                if spectro[0:4]=='NEID':
                    summary['rv_shift'] = summary['rv_shift']-summary['ccf_rv_mask_telluric_o2']/1000 - 0.740

                jdb = np.array(summary['jdb'])
                rv = rv_sys+summary['rv_shift']+summary['ccf_rv_G2']/1000
                rv_std = np.ones(len(rv))*30/1000 #30 m/s SNAKY RV uncertainties

                table = pd.DataFrame(np.array([jdb,rv,rv_std]).T,columns=['jdb','rv','rv_std'])
                table['code'] = code
                table['star'] = star
                table['ins'] = spectro
                table['drs'] = drs
                table = table[['code','star','ins','drs','jdb','rv','rv_std']]
                infos2.append(table)
                plt.errorbar(jdb,rv,yerr=rv_std,color=None,label=instrument,capsize=0,ls='',marker='o',zorder=100)

        if len(infos2):
            plt.xlabel('Jdb - 2,400,000 [days]')
            plt.ylabel('RV [km/s]')  
            plt.legend()

            infos2 = pd.concat(infos2)
            infos2 = infos2.reset_index(drop=True)
            infos.append(infos2)
            
            if anonymous:
                japanese_names = ["太郎", "花子", "健", "美咲", "翔", "愛", "直樹", "由美", "大輔", "陽菜", "拓也", "彩", "悠斗", "真央", "和也", "玲奈", "誠", "さくら", "隼人", "結衣", "隆", "千尋", "慎一", "奈々", "智也", "美穂", "剛", "麻衣", "亮", "久美子",'宮崎','なおみ']
                matching = {i:k for i,k in zip(np.unique(infos2['ins']), np.random.choice(japanese_names,len(np.unique(infos2['ins'])),replace=False))}
                for ins in np.unique(infos2['ins']):
                    infos2.loc[infos2['ins']==ins,'ins'] = matching[ins]

                infos2['rv'] = infos2['rv']+np.random.randn(len(infos2))*0.03
                infos2['jdb'] = infos2['jdb']+np.random.randn(len(infos2))*1
                plt.scatter(infos2['jdb'],infos2['rv'],color='k',zorder=10)
                infos_p.append(infos2)
            plt.savefig(output_dir+'/'+s+'/data/s1d/ALLINS_MERGED/RV.png')
            plt.close('rv')
            
            #infos2[['ins','jdb','rv','rv_std']].to_csv(output_dir+'/'+s+'/data/s1d/ALLINS_MERGED/RV_anonymous.csv')

    infos = pd.concat(infos)
    infos = infos.reset_index(drop=True)
    infos.to_csv(root+'/Snaky/database/'+filename+'_RV_infos.csv')

    if anonymous:
        infos_p = pd.concat(infos_p)
        infos_p = infos_p.reset_index(drop=True)
        infos_p.to_csv(root+'/Snaky/database/PUBLIC_'+filename+'_RV_infos.csv')

def create_snaky_ccf_db(output_dir, filename='All_stars', stars=['*'], infos=None, branch=None):
    root = '/'.join(output_dir.split('/')[:-1])
    if branch is None:
        branch = output_dir.split('/')[-1]

    if infos is None:
        infos = create_snaky_db(output_dir, filename=filename, stars=stars)
    files = [root+'/'+branch+'/'+i+'/data/s1d/'+j+'_'+k+'/STAR_INFO/Stellar_info*.p' for i,j,k in np.array(infos[['star','ins','drs']])]

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
    all_ccf = (np.array(all_ccf)*1e4).astype('int16')
    np.save(output_dir+'/database/'+filename+'_ccf.npy',all_ccf)

def create_snaky_spec_db(output_dir, filename='All_stars', stars=['*'], infos=None, wave_min=6100,wave_max=6200, branch=None):
    root = '/'.join(output_dir.split('/')[:-1])
    if branch is None:
        branch = output_dir.split('/')[-1]

    if infos is None:
        infos = create_snaky_db(output_dir, filename=filename, stars=stars, branch=branch)
    files = [root+'/'+branch+'/'+i+'/data/s1d/'+j+'_'+k+'/STAR_INFO/Stellar_info*.p' for i,j,k in np.array(infos[['star','ins','drs']])]

    wgrid = np.round(np.arange(wave_min,wave_max,0.01),2)
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
    np.save(output_dir+'/database/'+filename+'_spec_%.0f_%.0f.npy'%(wave_min,wave_max),all_spec)


def plot_starinfo(output_dir, ins='*', xvar='Teff_SNAKY', yvar='FWHM_G2', cvar=None, svar=None, branch=None):
    """
    Search inside thr Stellar_info*.p SNAKY files and query the values according to a two levels search of the variables:
    xvar = 'X_Y' implies -> sinfo[X][Y]
    """
    
    root = '/'.join(output_dir.split('/')[:-1])
    if branch is None:
        branch = output_dir.split('/')[-1]

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
    plt.xlabel(xvar)
    plt.ylabel(yvar)

def plot_fwhm(output_dir, ccf_mask='mask_telluric_o2', xvar='jdb', yvar='fwhm', alpha=0.4, color='k', branch='Snaky',marker='o', label=''):
    root = '/'.join(output_dir.split('/')[:-1])
    if branch is None:
        branch = output_dir.split('/')[-1]

    all_files = glob.glob(root+'/'+branch+'/*/data/s1d/*/WORKSPACE/Analyse_ccf.p')
    var = []
    for f in all_files:
        print(f)
        f2 = f.replace('/WORKSPACE/Analyse_ccf.p','/STAR_INFO/Ste*')
        if xvar=='jdb':
            v = 0
        elif xvar=='':
            v = 0
        else:
            v = pd.read_pickle(glob.glob(f2)[0])[xvar][branch.upper()]
        var.append(v)

    good = 0
    values = []
    for v,f in zip(var,all_files):
        tab = pd.read_pickle(f)
        try:
            ccf = tab['CCF_'+ccf_mask]['table']
            values.append(np.array(ccf[yvar]))
            if xvar!='':
                if xvar=='jdb':
                    x = ccf['jdb']
                else:
                    x = np.ones(len(ccf[yvar]))*v
                if good==0:
                    plt.scatter(x*np.nan, ccf[yvar], color=color, alpha=1.0, marker=marker, label=label)
                    plt.ylabel('%s [km/s]'%(yvar.upper()))
                plt.scatter(x, ccf[yvar], color=color, alpha=alpha, marker=marker)
                good = 1
        except:
            pass
    
    if (xvar=='')&(len(values)!=0):
        values = np.hstack(np.hstack(values))
        values = values[values==values]
        pby,pbx = np.histogram(values,bins=np.arange(0,10,0.025),density=True)
        pbx = 0.5*(pbx[1:]+pbx[0:-1])
        plt.fill_between(pbx,pby,alpha=alpha,color=color,label=label)
        plt.plot(pbx,pby,color='k')

def fix_dir_root(output_dir, instrument='SOPHIE-HE_1.0',stars='*'): # to be repaired
    pouet
    root = '/'.join(output_dir.split('/')[:-1])
    if branch is None:
        branch = output_dir.split('/')[-1]

    a = glob.glob(output_dir+'/'+stars+'/data/s1d/*/WORKSPACE/Analyse_summary.csv')
    for file in a:
        star = file.split('/data')[0].split('/')[-1]
        dir_root = file.split('WORKSPACE')[0]
        ins = dir_root.split('/')[-2]
        summary = pd.read_csv(file,index_col=0)
        dace_file = file.replace('WORKSPACE/Analyse_summary.csv','DACE_TABLE/Dace_extracted_table.csv')
        dace = pd.read_csv(dace_file,index_col=0)
        mask = (np.array(summary['ins'])==instrument)

        if os.path.exists(dace['fileroot'][0]):
            raw_visible = True
        else:
            raw_visible = False
            print(Fore.YELLOW+' [INFO] The path root of the RAW is not more visible'+Fore.RESET)

        if (sum(mask)!=0)&(ins!=instrument):
            print(Fore.YELLOW+' [WARNING] Some spectra with the wrong instrument found for %s'%(star)+Fore.RESET)
            mym.create_snaky_dir(output_dir,star,instrument)
            pickle.dump(mym.import_star_info(dir_root),open(dir_root.replace(ins,instrument)+'/STAR_INFO/Stellar_info_'+star+'.p','wb'))
            os.system('touch '+root+'/Snaky/'+star+'/data/s1d/'+instrument+'/REDUCTION_INFO/force_pre.txt')
            
            if raw_visible:
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

                mask_dace = myf.in1d(dace['fileroot'],np.array(all_files))

                new_dace = dace.loc[mask_dace].reset_index(drop=True)
                new_dace['ins'] = instrument
                new_dace['fileroot'] = np.array([n.replace('/'+ins+'/','/'+instrument+'/') for n in new_dace['fileroot']])
                new_dace.to_csv(dace_file.replace('/'+ins+'/','/'+instrument+'/'))
                
                old_dace = dace.loc[~mask_dace].reset_index(drop=True)
                old_dace.to_csv(dace_file)

            new_summary = summary.loc[mask].reset_index(drop=True)
            new_summary['filename'] = np.array([n.replace('/'+ins+'/','/'+instrument+'/') for n in new_summary['filename']])
            new_summary.to_csv(file.replace('/'+ins+'/','/'+instrument+'/'))

            old_summary = summary.loc[~mask].reset_index(drop=True)
            old_summary.to_csv(file)

            rassine_files = glob.glob(file.replace('Analyse_summary.csv','RASSINE*.p'))
            for k in np.setdiff1d(rassine_files,np.array(old_summary['filename'])):
                os.system('rm '+k)
            
            if np.sum(~mask_dace)==0:
                print(star)
                os.system('rm -rf '+file.split('/WORKSPACE')[0])
