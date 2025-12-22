import getopt
import snaky_variables as myv
import snaky_functions as myf
import snaky_classes as myc
import snaky_main as mym
import pandas as pd
import numpy as np 
import matplotlib.pylab as plt
import pickle
import os
import glob as glob
import sys
import time

from colorama import Fore

"""

SNAKY — Spectroscopic Novel Analysis Kit of Yarara

Sequence:

force_pre = force_pre,                       #1.  Read the spectrum in fits
force_summary = force_summary,               #2.  Extract header information
force_rvsys = force_rvsys,                   #3.  Compute the systemic RV
force_ccf = force_ccf,                       #4.  Compute the RVs
force_master = force_master,                 #5.  Create the master
force_resolution = force_resolution,         #6.  Compute the instrumental resolution with 02
force_atmos = force_atmos,                   #7.  Compute the atmospheric parameters
force_vsini = force_vsini,                   #8.  Compute the vsini
force_abs_continuum = force_abs_continuum,   #9.  Correct the blue continuum
force_activity = force_activity,             #10. Compute chromospheric activity index
force_mhk = force_mhk,                       #11. Compute the MHK activity index
force_spectroscopy = force_spectroscopy,     #12. Compute the master spectrum SRF
force_cleaning = force_cleaning,             #13. Clean subproducts

"""

cwd = os.getcwd()
root = '/'.join(cwd.split('/')[:-1])

ins = 'SOPHIE_0.5'
star = 'HD101'
chunck = 0
multiprocess = 5
sub_dico = 'matching_diff'
light_ram = True
use_yarara = False
begin = 0
end = 0
debug = False
dec = None
ra = None

if len(sys.argv)>1:
    optlist,args =  getopt.getopt(sys.argv[1:],'i:n:N:y:s:b:e:H:S:d:r:')
    for j in optlist:
        if j[0] == '-n':
            chunck = int(j[1])
        elif j[0] == '-N':
            multiprocess = int(j[1])
        elif j[0] == '-s':
            star = j[1]     
        elif j[0] == '-S':
            sub_dico = 'matching_'+j[1]
        elif j[0] == '-i':
            ins = j[1]
        elif j[0] == '-y':
            use_yarara = bool(int(j[1]))
        elif j[0] == '-b':
            begin = int(j[1])
        elif j[0] == '-e':
            end = int(j[1])
        elif j[0] == '-H':
            debug = bool(int(j[1]))
        elif j[0] == '-d':
            dec = float(j[1])
        elif j[0] == '-r':
            ra = float(j[1])  

steps = np.arange(begin,end+1,1).astype('int')

#### main reduction

def reduce(
        star,
        ins = 'SOPHIE_0.5',
        dec = None, # in degree
        ra = None, # in degree
        sub_dico = 'matching_diff',
        use_yarara = False,
        debug = False, 
        force_pre = False, 
        force_summary = False, 
        force_rvsys = False,
        force_ccf = False,
        force_master = False,
        force_resolution = False,
        force_atmos = False,
        force_vsini = False,
        force_abs_continuum = False,
        force_activity = False,
        force_mhk = False,
        force_spectroscopy = False,
        force_magcycle = False,
        force_cleaning = False,
        ):

    myf.print_box('\n---- Launching reduction %s with instrument %s  ----\n'%(star,ins))
    time_start = time.time()

    dir_root = root+'/Snaky/'+star+'/data/s1d/'+ins+'/'
    print(Fore.CYAN+" [INFO] (root directory) dir_root = '"+dir_root+"' \n"+Fore.RESET)

    star, ins = mym.create_snaky_dir(star,ins)

    if use_yarara:
        print(' [INFO] Loading YARARA workspace...')
        data_found = int(len(glob.glob(dir_root.replace('Snaky','Yarara')+'WORKSPACE/RASSINE*'))!=0)

        os.system('cp '+dir_root.replace('Snaky','Yarara')+'DACE_TABLE/Dace_extracted_table.csv '+dir_root+'DACE_TABLE')
        os.system('cp '+dir_root.replace('Snaky','Yarara')+'WORKSPACE/Analyse_summary.csv '+dir_root+'WORKSPACE')
        os.system('cp '+dir_root.replace('Snaky','Yarara')+'STAR_INFO/Stellar_info_%s.p '%(star)+dir_root+'STAR_INFO')
        summary = mym.import_summary(dir_root)
        if 'snr_computed' in summary.keys():
            summary['snr'] = summary['snr_computed']
        sinfo = mym.import_star_info(dir_root)
        ra = mym.ra_to_deg(sinfo['Ra']['fixed'].replace(' ',''))
        dec = mym.dec_to_deg(sinfo['Dec']['fixed'].replace(' ',''))
        dace_summary = pd.read_csv(dir_root+'DACE_TABLE/Dace_extracted_table.csv',index_col=0)
        dace_summary['RA'] = np.round(ra,6) ; dace_summary['DEC'] = np.round(dec,6)
        dace_summary.to_csv(dir_root+'DACE_TABLE/Dace_extracted_table.csv')
        
        force_pre = False
        force_summary = False
    else:
        sub_dico = 'matching_diff'

    if sub_dico != 'matching_diff':
        force_resolution = None

    #read fits files an create spectra normalised
    files = np.sort(glob.glob(dir_root+'RAW/*.fits'))
    if ins[0:6]=='SOPHIE':
        for f in files:
            mym.read_sophie(f,force=force_pre)
    elif (ins=='HARPS_3.5')|(ins=='HARPS03_3.5')|(ins=='HARPS15_3.5'):
        for f in files:
            mym.read_sophie(f,force=force_pre)
    elif (ins.split('_')[0][0:5]=='HARPS')|(ins.split('_')[0]=='HARPN')|(ins.split('_')[0]=='ESPRESSO'):
        for f in files:
            mym.read_espresso(f,force=force_pre)
    elif ins.split('_')[0]=='NEID':
        for f in files:
            mym.read_neid(f,force=force_pre)

    #extract time information from headers
    if (not os.path.exists(dir_root+'DACE_TABLE/Dace_extracted_table.csv'))|force_summary:
        if not len(files):
            print(Fore.YELLOW+' [EMERGENCY STOP] No spectra found in the RAW directory %s'%(dir_root)+Fore.RESET)
            print('\n')
            force_pre, force_summary, force_rvsys, force_ccf, force_master, force_atmos, force_resolution, force_vsini,force_abs_continuum, force_activity ,force_mhk,force_spectroscopy ,force_cleaning = [False]*13           
        else:
            summary = mym.extract_header(files, ins, debug=debug, dec=dec, ra=ra)
            summary['fileroot'] = files
            summary['ins'] = ins
            summary.to_csv(dir_root+'DACE_TABLE/Dace_extracted_table.csv')

    files = np.sort(glob.glob(dir_root+'WORKSPACE/RASSINE*.p'))
    if (not os.path.exists(dir_root+'WORKSPACE/Analyse_summary.csv'))|force_summary:
        dace_summary = pd.read_csv(dir_root+'DACE_TABLE/Dace_extracted_table.csv',index_col=0)
        berv = np.array(dace_summary['berv_computed'])
        inss = np.array(dace_summary['ins'])
        jdb = []
        for f in files:
            arcfiles = pd.read_pickle(f)['parameters']['arcfiles']
            temp = []
            for arc in arcfiles:
                val = dace_summary.loc[dace_summary['fileroot']==arc,'rjd']
                temp.append(val)
            jdb.append(np.mean(temp))
        
        if len(jdb)!=len(files):
            jdb = np.arange(len(files))

        flag = np.zeros(len(files)).astype('int')
        if debug:
            mym.snaky_help()
            print(files,'\n',inss,'\n',jdb,'\n',berv,'\n',flag)
        summary = pd.DataFrame(np.array([files, inss, jdb, berv, flag]).T,columns=['filename','ins','jdb','berv','flag'])
        summary.to_csv(dir_root+'WORKSPACE/Analyse_summary.csv')

        star_info = {
            'Name':star,
            'Ra':{'fixed':0.0},
            'Dec':{'fixed':0.0},
            'Teff':{'fixed':5775},
            'FeH':{'fixed':0.0},
            'FWHM':{'fixed':6.0},
            'Rv_sys':{'fixed':0.0},
            'Contrast':{'fixed':0.4},
            'CCF_beta':{'fixed':2.0},
            'SB2':{'fixed':0.0},
            'EW':{'fixed':0.0},
            'Mstar':{'fixed':1.0},
            'Rstar':{'fixed':1.0},
            'Teff':{'fixed':5775},
            'Log_g':{'fixed':4.44},
            'FeH':{'fixed':0.0},
            'BV':{'fixed':0.66},
            'Vmicro':{'fixed':1.0},
            'Vmacro':{'fixed':1.0},
            'stellar_template':{'fixed':'MARCS_T5750_g4.5'},
            'Vsini':{'fixed':2.0},
            'RHK':{'fixed':-5.00},
            'MHK':{'fixed':0.0},
            'Prot':{'fixed':25.0},
            'Pmag':{'fixed':11.0},
            }

        sinfo = myf.touch_pickle(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star))
        for kw in star_info:
            sinfo[kw] = star_info[kw]
        pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))

    summary = mym.import_summary(dir_root)
    files = np.array(summary['filename'])
    if 'flag1' not in summary.keys():
        wave_grid, sts, sts_err = mym.import_sts(files, sub_dico=sub_dico)
        anomalous = np.sum((sts>1.02)|(sts<0),axis=1)*100/len(wave_grid)
        anomalous = np.round(anomalous,0).astype('int')
        summary['anomalous'] = anomalous
        kept = (anomalous<5)

        print(' [INFO] Number of good spectra = %.0f'%(sum(kept)))
        print(' [INFO] Number of anomalous spectra = %.0f'%(len(kept)-sum(kept)))
        if sum(kept)==0:
            print(Fore.YELLOW+' [WARNING] No good spectra (Emergency stop)'+Fore.RESET)
            print(' [INFO] criterion = ',anomalous)
            print('\n')
            force_pre, force_summary, force_rvsys, force_ccf, force_master, force_atmos, force_resolution, force_vsini,force_abs_continuum, force_activity ,force_mhk,force_spectroscopy ,force_cleaning = [False]*13           

        summary['flag1'] = 0
        summary.loc[summary.index[~kept],'flag1'] = 1
        summary.to_csv(dir_root+'WORKSPACE/Analyse_summary.csv')
    
    summary = mym.import_summary(dir_root)
    if force_rvsys:
        teff,feh = mym.yarara_flux_density(files)
        
        rv_sys = []
        for f in files:
            spec = mym.import_spectrum(f,sub_dico=sub_dico)
            rv_sys1 = mym.yarara_rough_rv_sys(spec,teff=teff,verbose=debug)
            rv_sys.append(rv_sys1)
        rv_sys = np.array(rv_sys)
        rv_sys_std = np.nanstd(rv_sys)
        rv_sys_approx = np.round(np.nanmedian(rv_sys),2)
        print('\n [INFO] Final aproximated RV_sys = %.1f +/- %.1f kms'%(rv_sys_approx,rv_sys_std))

        if debug:
            mym.yarara_check_rv_sys(spec, 15, rv_sys_approx, dir_root=dir_root)

        anomalous = np.array(summary['anomalous'])
        spec  = mym.import_spectrum(files[np.argmin(anomalous)],sub_dico=sub_dico)
        sinfo2 = mym.yarara_check_rv_sys_wrapper(dir_root,spec,rv_sys_approx)

        dace_summary = pd.read_csv(dir_root+'DACE_TABLE/Dace_extracted_table.csv',index_col=0)
        ra_deg = np.nanmedian(dace_summary['RA'])
        dec_deg = np.nanmedian(dace_summary['DEC'])
        
        fwhm, rv_sys, contrast, beta_gnd, sb_flag, ccf = sinfo2
        sinfo = mym.import_star_info(dir_root)
        sinfo = myf.update_info_lvl2(sinfo,'Rv_sys','SNAKY',rv_sys)
        sinfo = myf.update_info_lvl2(sinfo,'CCF_beta','SNAKY',beta_gnd)
        sinfo = myf.update_info_lvl2(sinfo,'SB2','SNAKY',sb_flag)
        sinfo = myf.update_info_lvl2(sinfo,'Ra','SNAKY',ra_deg)
        sinfo = myf.update_info_lvl2(sinfo,'Dec','SNAKY',dec_deg)
        sinfo = myf.update_info_lvl2(sinfo,'Teff','fixed',teff)
        sinfo = myf.update_info_lvl2(sinfo,'CCF_beta','SNAKY',beta_gnd)
        sinfo = myf.update_info_lvl2(sinfo,'FeH','fixed',feh)
        sinfo = myf.update_info_lvl2(sinfo,'FWHM','fixed',fwhm)
        sinfo = myf.update_info_lvl2(sinfo,'Contrast','SNAKY',contrast)
        pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))

    sinfo = mym.import_star_info(dir_root)
    if force_ccf:
        kept = np.array(1-summary['flag1'])
        if sum(kept)!=0:
            sub = summary.loc[summary['flag1']==0]
            files = np.array(sub['filename'])
            ccf_output = mym.yarara_ccf(dir_root, files, rv_sys, fwhm, beta_gnd, 'G2', debug=debug)
            ct = ccf_output['contrast'].y
            med_ct = np.nanmedian(ct)
            kept2 = abs(ct-med_ct)<2
            print(' [INFO] Number of good spectra (after CCF check) = %.0f'%(sum(kept2)))
            print(' [INFO] Number of bad spectra (after CCF check) = %.0f'%(len(kept)-sum(kept)+len(kept2)-sum(kept2)))
            if np.sum(kept2)==0:
                kept2 = np.ones(len(ct)).astype('bool')
            summary['flag2'] = 0
            summary.loc[sub.index[~kept2],'flag2'] = 1
            summary.to_csv(dir_root+'WORKSPACE/Analyse_summary.csv')

    summary = mym.import_summary(dir_root)
    if force_ccf:
        kept = np.array(1-summary['flag1'])*np.array(1-summary['flag2'])
        if sum(kept)!=0:
            files = np.array(summary.loc[kept==1,'filename'])
            ccf_output = mym.yarara_ccf(dir_root, files, rv_sys, fwhm, beta_gnd, 'G2', debug=debug, sub_dico=sub_dico)
            sinfo['FWHM']['G2'] = np.round(np.nanmedian(ccf_output['fwhm'].y),2)
            ccf_output1 = mym.yarara_ccf(dir_root, files, rv_sys, fwhm, beta_gnd, 'Kitty', ccf_oversampling=3, debug=debug, sub_dico=sub_dico)
            sinfo['FWHM']['KITTY'] = np.round(np.nanmedian(ccf_output1['fwhm'].y),2)
            ccf_output2 = mym.yarara_ccf(dir_root, files, rv_sys, fwhm, beta_gnd, 'Garfield', ccf_oversampling=3, debug=debug, sub_dico=sub_dico)
            sinfo['FWHM']['GARFIELD'] = np.round(np.nanmedian(ccf_output2['fwhm'].y),2)
            pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))

    if force_master:
        ccf_output = mym.import_ccf(dir_root,'G2')
        master = mym.master_spectrum(ccf_output['filename'],ccf_output['rv'].y,0)
        material = {'wave':master.x,'reference_spectrum':master.y}
        pickle.dump(material,open(dir_root+'WORKSPACE/Analyse_material.p','wb'))
        #sinfo = yarara_check_rv_sys_wrapper(dir_root,master,0) #check if on 0
    
    if force_atmos:
        master = mym.import_master(dir_root)
        fwhm = sinfo['FWHM']['G2']
        CT,EW = mym.yarara_iron_lines(dir_root, master, fwhm, rv_sys=rv_sys)
        for kw in CT.keys():
            sinfo['Contrast'][kw] = CT[kw]
        for kw in EW.keys():
            sinfo['EW'][kw] = EW[kw]
        pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))

        atmos = mym.yarara_atmos_xgb_spectroscopy(dir_root, sinfo, resolution=80000, phot=True)
        teff,feh,logg,M,R,BV,vmicro,vmacro = atmos

        suffixe = 'ATLAS_T%.0f_g%.1f'%(np.round(teff,-2),np.round(logg,1))
        print(' [INFO] Atmospheric model set to : %s'%(suffixe))

        sinfo = myf.update_info_lvl2(sinfo,'Mstar','SNAKY',M)
        sinfo = myf.update_info_lvl2(sinfo,'Rstar','SNAKY',R)
        sinfo = myf.update_info_lvl2(sinfo,'Teff','SNAKY',teff)
        sinfo = myf.update_info_lvl2(sinfo,'FeH','SNAKY',feh)
        sinfo = myf.update_info_lvl2(sinfo,'Log_g','SNAKY',logg)
        sinfo = myf.update_info_lvl2(sinfo,'BV','SNAKY',BV)
        sinfo = myf.update_info_lvl2(sinfo,'Vmicro','SNAKY',vmicro)
        sinfo = myf.update_info_lvl2(sinfo,'Vmacro','SNAKY',vmacro)
        sinfo = myf.update_info_lvl2(sinfo,'stellar_template','SNAKY',suffixe)

        pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))

    if force_resolution:
        summary = mym.import_summary(dir_root)
        ccf_output = mym.import_ccf(dir_root,'G2')
        files = ccf_output['filename']
        shift_rv = ccf_output['rv'].y
        berv = np.array(summary.loc[np.in1d(summary['filename'],files),'berv'])
        fwhm_ins = mym.yarara_instrumental_resolution(dir_root, files, shift_rv, berv)
        if ins[0:6]=='SOPHIE':
            output = np.array([files,fwhm_ins]).T
            newins = np.array([['SOPHIE-HE_0.5','SOPHIE_0.5'][i<5] for i in fwhm_ins])
            output[:,-1] = newins
            loc = [np.where(summary['filename']==f)[0][0] for f in output[:,0]]
            summary.loc[loc,'ins'] = output[:,-1]
        
        sinfo['FWHM']['O2'] = np.round(np.nanmedian(fwhm_ins),2)            
        pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))
        summary.to_csv(dir_root+'WORKSPACE/Analyse_summary.csv')

    summary = mym.import_summary(dir_root)
    sinfo = mym.import_star_info(dir_root)

    if force_vsini:
        teff = sinfo['Teff']['SNAKY']
        logg = sinfo['Log_g']['SNAKY']
        feh = sinfo['FeH']['SNAKY']
        ins_res = sinfo['FWHM']['O2']
        vsini = mym.yarara_vcat(dir_root, teff, logg, ins, ins_res=ins_res, sub_dico=sub_dico) 
        mym.yarara_vsini(dir_root, Prot=None, Rs=None)
        sinfo = myf.update_info_lvl2(sinfo,'Vsini','SNAKY',np.round(np.nanmean(vsini),2))
        pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))

    material = mym.import_material(dir_root)
    if ('correction_factor' not in material.keys())|(force_abs_continuum):
        master = mym.import_master(dir_root)
        template_flux, correction = mym.yarara_correct_continuum_absorption(dir_root, master, sinfo)
        material['stellar_template'] = template_flux
        material['correction_factor'] = correction
        pickle.dump(material,open(dir_root+'WORKSPACE/Analyse_material.p','wb'))

    if force_activity:
        sinfo = mym.import_star_info(dir_root)
        rv_sys = sinfo['Rv_sys']['SNAKY']
        kept = np.array(1-summary['flag1'])*np.array(1-summary['flag2'])
        files = np.array(summary.loc[kept==1,'filename'])
        ccf_output = mym.import_ccf(dir_root,'G2')
        tab_proxies, CT, mask_activity = mym.yarara_activity_index(ccf_output['filename'], rv_sys, ccf_output['rv'].y, material=material)
        material['activity_proxies'] = mask_activity
        pickle.dump(material,open(dir_root+'WORKSPACE/Analyse_material.p','wb'))

        for kw in CT.keys():
            sinfo['Contrast'][kw] = np.round(CT[kw],5)
        pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))
        for kw in tab_proxies:
            if (kw!='filename')&(kw in summary.keys()):
                del summary[kw]
        summary = pd.merge(summary,tab_proxies,on='filename',how='left')
        summary.to_csv(dir_root+'WORKSPACE/Analyse_summary.csv')

    if force_mhk:
        sinfo = mym.import_star_info(dir_root)
        rv_sys = sinfo['Rv_sys']['SNAKY']
        teff = sinfo['Teff']['SNAKY']
        material = mym.import_material(dir_root)
        ccf_output = mym.import_ccf(dir_root,'G2')   
        summary = mym.import_summary(dir_root)
        proxy = np.array(summary.loc[np.in1d(summary['filename'],ccf_output['filename']),'CaII'])
        dico, rhk, mhk = mym.yarara_activity_mhk(dir_root, ccf_output['filename'], rv_sys, ccf_output['rv'].y, teff, material, proxy)
        
        for kw in ['RHK','RHK_std','MHK','MHK_std']:
            if kw in summary.keys():
                del summary[kw]
        summary = pd.merge(summary,dico[['filename','RHK','RHK_std','MHK','MHK_std']],on='filename',how='left')
        summary.to_csv(dir_root+'WORKSPACE/Analyse_summary.csv')

        sinfo = myf.update_info_lvl2(sinfo,'RHK','SNAKY',np.round(rhk,3))
        sinfo = myf.update_info_lvl2(sinfo,'MHK','SNAKY',np.round(mhk,1))

        prot = myf.conv_rhk_prot(sinfo['RHK']['SNAKY'], sinfo['BV']['SNAKY'])
        prot_vsini = np.round(sinfo['Rstar']['SNAKY']*25/(sinfo['Vsini']['SNAKY']/2),1)
        prot1 = np.round(prot[2],1)
        prot2 = np.round(prot[0],1)
        prot1 = np.max([prot1,1])
        prot2 = np.max([prot2,1])
        prot_vsini = np.min([prot_vsini,100])
        sinfo = myf.update_info_lvl2(sinfo,'Prot','Mamaj+08', prot1)
        sinfo = myf.update_info_lvl2(sinfo,'Prot','Noyes+84', prot2)
        sinfo = myf.update_info_lvl2(sinfo,'Prot','VSINI', prot_vsini)
        pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))

        mym.plot_mhk(dir_root)
        mym.create_finch_db(dir_root=dir_root)

    if force_spectroscopy:
        material = mym.import_material(dir_root)
        sinfo = mym.import_star_info(dir_root)
        rv_sys = sinfo['Rv_sys']['SNAKY']
        master = myc.tableXY(myf.doppler_r(material['wave'],rv_sys*1000)[1],material['reference_spectrum'],0*material['wave'])
        master.interpolate(new_grid=material['wave'],method='linear')

        spectroscopy = {
            'wave':master.x,
            'flux':master.y,
            }

        master = myc.tableXY(myf.doppler_r(material['wave'],rv_sys*1000)[1],material['reference_spectrum']*material['correction_factor'],0*material['wave'])
        master.interpolate(new_grid=material['wave'],method='linear',fill_value=0)

        spectroscopy['flux_corrected'] = master.y

        for kw in sinfo.keys():
            if 'fixed' in sinfo[kw]:
                extracted = sinfo[kw]
                del extracted['fixed']
                spectroscopy[kw] = extracted
        pickle.dump(spectroscopy,open(dir_root+'WORKSPACE/Analyse_spectroscopy.p','wb'))

    if force_magcycle:
        finch_output = mym.yarara_finch(dir_root, rm_source=['DACE'], offset_instrument='no!', ext='')
        sinfo = myf.update_info_lvl2(sinfo,'Pmag','SNAKY', finch_output[1])
        pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))
        pickle.dump({
            'Starname':star,
            'Pmag':finch_output[1],
            'Kmag_mean':finch_output[2],
            'Kmag_amp':finch_output[3],
            'Kmag_pred':finch_output[4],
            'Phase_pred':finch_output[5],
            'Phase_side':finch_output[6]}, 
            open(dir_root.replace(ins+'/','ALLINS_MERGED/Pmag_FINCH_info.p'),'wb'))

    time_end = time.time()
    duration = np.round((time_end-time_start)/60,2)
    tag_duration = str(int(duration//1))+'m'+str(int((duration%1)*60))+'s'
    print(Fore.CYAN+"\n [INFO] Processing achieved in "+tag_duration+" of dir_root = '"+dir_root+"' \n"+Fore.RESET)

    if force_cleaning:
        mym.clean_light_dir(dir_root)

force_init = bool(np.sum(steps==0))
force_pre = bool(np.sum(steps==1))
force_summary = bool(np.sum(steps==2))
force_rvsys = bool(np.sum(steps==3))
force_ccf = bool(np.sum(steps==4))
force_master = bool(np.sum(steps==5))
force_atmos = bool(np.sum(steps==6))
force_resolution = bool(np.sum(steps==7))
force_vsini = bool(np.sum(steps==8))
force_abs_continuum = bool(np.sum(steps==9))
force_activity = bool(np.sum(steps==10))
force_mhk = bool(np.sum(steps==11))
force_spectroscopy = bool(np.sum(steps==12))
force_magcycle = bool(np.sum(steps==13))
force_cleaning = bool(np.sum(steps==14))

if (len(steps)==1)&(np.sum(steps==0)):
    star, ins = mym.create_snaky_dir(star,ins)
else:
    if chunck!=0: #multiprocessing via multiterminal
        files = glob.glob(root+'/Snaky/*/data/s1d/%s/RAW/*.fits'%(ins))
        stars = np.sort(np.unique([f.split('Snaky/')[-1].split('/')[0] for f in files]))
        to_process = np.array_split(stars,multiprocess)[chunck-1]
        print('[INFO] The following stars will be processed:')
        print(to_process)
        for star in to_process:
            reduce(
                star,
                ins = ins,
                debug = debug,
                ra = ra,
                dec = dec,
                force_pre = force_pre,
                force_summary = force_summary,
                force_rvsys = force_rvsys,
                force_ccf = force_ccf,
                force_master = force_master,
                force_atmos = force_atmos,
                force_resolution = force_resolution,
                force_vsini = force_vsini,
                force_abs_continuum = force_abs_continuum,
                force_activity = force_activity,
                force_mhk = force_mhk,  
                force_spectroscopy = force_spectroscopy,
                force_magcycle = force_magcycle,      
                force_cleaning = force_cleaning,
                )
            plt.close('all')
    else:
        reduce(
            star,
            ins = ins,
            debug = debug, 
            ra = ra,
            dec = dec,
            use_yarara = use_yarara,
            sub_dico = sub_dico,
            force_pre = force_pre,                       #1
            force_summary = force_summary,               #2
            force_rvsys = force_rvsys,                   #3
            force_ccf = force_ccf,                       #4
            force_master = force_master,                 #5
            force_atmos = force_atmos,                   #6
            force_resolution = force_resolution,         #7
            force_vsini = force_vsini,                   #8
            force_abs_continuum = force_abs_continuum,   #9
            force_activity = force_activity,             #10
            force_mhk = force_mhk,                       #11
            force_spectroscopy = force_spectroscopy,     #12
            force_magcycle = force_magcycle,             #13
            force_cleaning = force_cleaning,             #14
            )
    
if False:
    pass
    #run snaky.py -s HD128621 -y 1 -i CORALIE14_3.8 -b 2 -e 12