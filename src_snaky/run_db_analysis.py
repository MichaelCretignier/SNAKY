import os 
import numpy as np 
import matplotlib.pylab as plt
import pandas as pd
import pickle
from tqdm import tqdm
import glob as glob

from . import snaky_functions as myf
from . import snaky_classes as myc
from . import snaky_variables as myv

def check_snaky_processing(output_dir,instrument='*'):

    os.makedirs(output_dir+'/database', exist_ok=True)
    instruments = glob.glob(output_dir+'/*/data/s1d/'+instrument+'/RAW')
    instruments = np.unique([i.split('/')[-2] for i in instruments])
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
        for kw in kws:
            nb = int(nb_stars-np.sum(all_info[kw.split('_')[1][0:5]]=='XXXX'))
            print('%s = %.0f (%.1f%%)'%(kw,nb,100*nb/(nb_stars+1e-6)))

        print('\n[INFO] File DB created: '+output_dir+'/database/Snaky_processing_db_'+instrument.replace('*','')+'.csv')
        all_info.to_csv(output_dir+'/database/Snaky_processing_db_'+instrument.replace('*','')+'.csv')


def create_snaky_db(output_dir, filename='All_stars_summary_infos.csv', stars=['*']):
    os.makedirs(output_dir+'/database', exist_ok=True)
    
    ntot = len(stars)

    files = []
    for s in stars:
        files.append(glob.glob(output_dir+'/'+s+'/data/s1d/*/STAR_INFO/Stellar_info*.p'))
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
    print('\n [INFO] Nb unique stars processed = %.0f (%.0f%%)\n'%(nb_processed,100*nb_processed/ntot))

    infos.to_csv(output_dir+'/database/'+filename)

    return infos

def compare_snaky_atmos(stars=['*']):
    all_files = []
    for s in stars:
        files = glob.glob(root+'/Snaky/'+s+'/data/s1d/*/WORKSPACE/Analyse_samples.csv')
        all_files.append(files)
    all_files = np.hstack(all_files)

    stars = np.sort(np.unique([f.split('/data')[0].split('/')[-1] for f in all_files]))

    for s in tqdm(stars):
        count = -1
        files = glob.glob(root+'/Snaky/'+s+'/data/s1d/*/WORKSPACE/Analyse_samples.csv')
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

            plt.savefig(root+'/Snaky/'+s+'/data/s1d/ALLINS_MERGED/Atmos_all_instrument.pdf')
        if len(stars)>5:
            plt.close('all')


def create_snaky_finch_db(filename='All_stars', stars=['*'], infos=None):
    if infos is None:
        infos = create_snaky_db(filename=filename, stars=stars, branch='Snaky')
    stars = np.sort(np.unique(infos['star']))

    tgrid = 61041+182.5*np.arange(21)
    infos = []
    for s in tqdm(stars):
        if os.path.exists(root+'/Snaky/'+s+'/data/s1d/ALLINS_MERGED/Pmag_FINCH_info.p'):
            info = pd.read_pickle(root+'/Snaky/'+s+'/data/s1d/ALLINS_MERGED/Pmag_FINCH_info.p')
            if 'Phase_side' in info.keys():
                info['Phase_pred_side'] = info['Phase_side']
            Pmag_model = pd.read_csv(root+'/Snaky/'+s+'/data/s1d/ALLINS_MERGED/Finch_MHK_GP_model.csv',index_col=0)
            model = myc.tableXY(Pmag_model['jdb'],Pmag_model['proxy'],Pmag_model['proxy_std'])
            model.interpolate(new_grid=tgrid,method='linear')
            liste = [s,info['Pmag'],info['Kmag_mean'],info['Kmag_amp'],info['Kmag_pred'],info['Phase_pred'],info['Phase_pred_side']]
            liste = liste + list(np.round(model.y,1)) + list(np.round(model.yerr,1))
            infos.append(liste)
    table = pd.DataFrame(infos,columns=['star','Pmag','Kmean','Kamp','Kpred','Lpred','Lside']+['MHK_%.1f'%(i) for i in np.arange(2026,2036.1,0.5)]+['MHK_err_%.1f'%(i) for i in np.arange(2026,2036.1,0.5)])
    table.to_csv(root+'/Snaky/database/'+filename+'_FINCH_mag.csv')

    infos2 = []
    for s in tqdm(stars):
        if os.path.exists(root+'/Snaky/'+s+'/data/s1d/ALLINS_MERGED/Finch_MHK.csv'):
            tab = pd.read_csv(root+'/Snaky/'+s+'/data/s1d/ALLINS_MERGED/Finch_MHK.csv',index_col=0)
            tab['star'] = s
            tab = tab[['star','species','jdb','proxy','proxy_std','qc']]
            infos2.append(tab)
    infos2 = pd.concat(infos2)
    infos2.to_csv(root+'/Snaky/database/PRIVATE_'+filename+'_FINCH.csv')



def create_snaky_rv_db(filename='All_stars', stars=['*'], infos=None, anonymous=False):
    if infos is None:
        infos = create_snaky_db(filename=filename, stars=stars, branch='Snaky')
    stars = np.sort(np.unique(infos['star']))

    infos = [] ; infos_p = []
    for s in tqdm(stars):
        infos2 = []
        plt.figure('rv')
        for f in glob.glob(root+'/Snaky/'+s+'/data/s1d/*/STAR_INFO/Stellar_info_*p'): 
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
            plt.savefig(root+'/Snaky/'+s+'/data/s1d/ALLINS_MERGED/RV.png')
            plt.close('rv')
            
            infos2[['ins','jdb','rv','rv_std']].to_csv(root+'/Snaky/'+s+'/data/s1d/ALLINS_MERGED/RV_anonymous.csv')


    infos_p = pd.concat(infos_p)
    infos_p = infos_p.reset_index(drop=True)
    infos_p.to_csv(root+'/Snaky/database/'+filename+'_RV_infos.csv')

    infos = pd.concat(infos)
    infos = infos.reset_index(drop=True)
    infos.to_csv(root+'/Snaky/database/PRIVATE_'+filename+'_RV_infos.csv')

def create_snaky_ccf_db(filename='All_stars', stars=['*'], branch='Snaky', infos=None):

    if infos is None:
        s = create_snaky_db(filename=filename, stars=stars, branch=branch)
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
    np.save(root+'/Snaky/database/'+filename+'_ccf.npy',all_ccf)

def create_snaky_spec_db(filename='All_stars', stars=['*'], branch='Snaky', infos=None, wave_min=6100,wave_max=6200):
    if infos is None:
        infos = create_snaky_db(filename=filename, stars=stars, branch=branch)
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
    np.save(root+'/Snaky/database/'+filename+'_spec_%.0f_%.0f.npy'%(wave_min,wave_max),all_spec)


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

def plot_fwhm(dir_root, ccf_mask='mask_telluric_o2', xvar='jdb', yvar='fwhm', alpha=0.4, color='k', branch='Snaky',marker='o', label=''):
    all_files = glob.glob(dir_root+'WORKSPACE/Analyse_ccf.p')
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

def create_finch_db(dir_root):

    files = np.sort(glob.glob(dir_root+'/*/data/s1d/*/WORKSPACE/Analyse_summary.csv'))

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