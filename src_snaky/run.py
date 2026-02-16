import getopt
import pandas as pd
import numpy as np 
import matplotlib.pylab as plt
import pickle
import os
import glob as glob
import sys
import time
import tracemalloc
import psutil

from colorama import Fore

from . import snaky_variables as myv
from . import snaky_functions as myf
from . import snaky_classes as myc
from . import snaky_main as mym

# =============================================================================
# MEMORY AND TIME MONITORING
# =============================================================================

tracemalloc.start()
top_stats = []
memory_history = [[-99.9,0,0,0]]
begin = time.time()
time_step = {'begin':begin}
timestamp_reduction = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

debug_mode = 0

def monitor_ram(stage=0):
    snapshot = tracemalloc.take_snapshot()
    top_stats.append(snapshot.statistics('traceback'))
    mem = tracemalloc.get_traced_memory()
    memory_history.append([stage]+[np.round(i/1e9,4) for i in mem]+[psutil.virtual_memory().percent])
    print(Fore.CYAN + '\n [INFO] RAM [Gb] allocated now and peak value : ',memory_history[-1],Fore.RESET+'')
    tracemalloc.clear_traces()

def write_progress(stage, step, time_step, savefile=None):

    #monitoring memory
    monitor_ram(stage=stage)
    myf.print_ram(step=step+'='+str(stage))

    now = time.time()
    time_step[step] = now 

    table_time = pd.DataFrame(time_step.values(),index=time_step.keys(),columns=['time_abs'])
    dt = np.hstack([0,np.diff(table_time['time_abs'])])
    table_time['time_step_min'] = dt
    table_time['frac_time']=100*table_time['time_step_min']/np.sum(table_time['time_step_min'])
    table_time['time_step_min'] /= 60 #convert in minutes
    table_time['stage'] = np.array(memory_history)[:,0]
    table_time['RAM_gb'] = np.array(memory_history)[:,1]
    table_time['RAM_gb_peak'] = np.array(memory_history)[:,2]
    table_time['RAM_filled'] = np.array(memory_history)[:,3]

    table_time[['time_step_min','frac_time','RAM_gb','RAM_gb_peak']] = np.round(table_time[['time_step_min','frac_time','RAM_gb','RAM_gb_peak']],2)

    if savefile is not None:
        table_time.to_csv(savefile)

#### main reduction

class SnakyError(Exception):
    pass

class run():
    def __init__(self, job_id=0):
        self.sy_output_dir = myv.WORKSPACE+'/'
        self.sy_job_id = job_id
        #if job_id!=0:
        #    for ccf_mask in ['G2','Garfield','Kitty','Magicat']:
        #        time.sleep(np.random.randint(0,5)) #to avoid conflicts in case of parallel runs
        #        os.system('cp '+myv.MATERIAL_DIR+'/MASK_CCF/CCF_'+ccf_mask+'.fits '+myv.MATERIAL_DIR+'/MASK_CCF/CCF_'+ccf_mask+'_N%.0f.fits'%(job_id))
        #        print(' [INFO] CCF mask %s for job_id = %.0f copied!'%(ccf_mask,job_id))
    
    def set_output_dir(self,outputdir):
        self.sy_output_dir = outputdir

    def format(self,debug=False):
        dir_root = self.sy_dir_root
        starname = self.sy_starname
        ins = self.sy_instrument
        if starname.split('_')[0]!='Sun':
            print(' [INFO] Formatting SNAKY with basic minimal information...')
            dace_table = mym.import_dace_table(dir_root)
            files = np.array(dace_table['fileroot'])
            sinfo = mym.import_star_info(dir_root)
            if self.sy_rassine_db:
                pouet #TBD
            else:
                query = mym.extract_header(files, ins, debug=debug, dec=None, ra=None)
                sinfo['Ra']['fixed'] = np.median(query['RA'])
                sinfo['Dec']['fixed'] = np.median(query['DEC'])
                pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(starname),'wb'))
            query['ins'] = ins
            dace_table = pd.concat([dace_table,query],axis=1)
            dace_table.to_csv(dir_root+'DACE_TABLE/Dace_extracted_table.csv')

    def set_starinfo(self):
        starinfo = self.sy_dir_root+'STAR_INFO/Stellar_info_%s.p'%(self.sy_starname)
        template = myv.star_info.copy()
        template['Name'] = self.sy_starname
        pickle.dump(template,open(starinfo,'wb'))

    def set_dataset(self, starname, ins, files, sub_dico='matching_diff'):
        starname,ins = mym.create_snaky_dir(self.sy_output_dir,starname,ins)
        self.sy_starname = starname
        self.sy_instrument = ins
        self.sy_dir_root = self.sy_output_dir+'Snaky/'+starname+'/data/s1d/'+ins+'/'
        dir_root = self.sy_dir_root
       
        print(Fore.CYAN+" [INFO] (root directory) dir_root = '"+dir_root+"' \n"+Fore.RESET)
        #if self.sy_job_id!=0:
        #    for ccf_mask in ['G2','Garfield','Kitty','Magicat']:
        #        os.system('mv '+myv.MATERIAL_DIR+'/MASK_CCF/CCF_'+ccf_mask+'_N%.0f.fits'%(self.sy_job_id)+' '+dir_root+'CCF_MASK/')

        self.sy_files = files
        self.sy_sub_dico = sub_dico

        file_test = self.sy_files[0]
        self.sy_rassine_db = False
        self.sy_yarara_db = False

        if file_test.split('/')[-1][0:8]=='RASSINE_':
            self.sy_rassine_db = True
            if len(file_test.split('/Yarara/'))==2:
                self.sy_yarara_db = True

        #read fits files an create spectra normalised
        check_files =  np.array([os.path.exists(f) for f in self.sy_files])

        if np.product(check_files)==0:
            print(Fore.YELLOW+' [EMERGENCY STOP] All the spectra were not found'+Fore.RESET)
            print(' [INFO] Missing files:\n')
            for f in np.array(files)[~check_files]:
                print(f)            
            raise SnakyError('All the spectra indicated were not found.')

    def init_workspace(self):
        dir_root = self.sy_dir_root

        self.set_starinfo()

        if self.sy_yarara_db:
            file_test = self.sy_files[0]
            self.copy_yarara(file_test.split('WORKSPACE/RASSINE')[0])

        if not self.sy_yarara_db:
            table = pd.DataFrame(self.sy_files,columns=['fileroot'])
            table.to_csv(dir_root+'DACE_TABLE/Dace_extracted_table.csv')
            self.format()

    def set_summary(self, debug=False):
        dir_root = self.sy_dir_root
        files = self.sy_rassine_files
        ins = self.sy_instrument
        if self.sy_yarara_db==False:
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
                print(len(files),len(inss),len(jdb),len(berv),len(flag))
            summary = pd.DataFrame(np.array([files, inss, jdb, berv, flag]).T,columns=['filename','ins','jdb','berv','flag'])
            summary.to_csv(dir_root+'WORKSPACE/Analyse_summary.csv')
        else:
            summary = pd.read_csv(dir_root+'WORKSPACE/Analyse_summary.csv',index_col=0)
            self.sy_rassine_files = np.array(summary['filename'].values)    

    def copy_yarara(self,yarara_root):
        dir_root = self.sy_dir_root
        star = self.sy_starname

        print(' [INFO] Loading YARARA workspace...')
        data_found = int(len(glob.glob(yarara_root+'WORKSPACE/RASSINE*'))!=0)

        os.system('cp '+yarara_root+'DACE_TABLE/Dace_extracted_table.csv '+dir_root+'DACE_TABLE')
        os.system('cp '+yarara_root+'WORKSPACE/Analyse_summary.csv '+dir_root+'WORKSPACE')
        os.system('cp '+yarara_root+'STAR_INFO/Stellar_info_%s.p '%(star)+dir_root+'STAR_INFO')
        summary = mym.import_summary(dir_root)
        if 'snr_computed' in summary.keys():
            summary['snr'] = summary['snr_computed']
        summary.to_csv(dir_root+'WORKSPACE/Analyse_summary.csv')
        sinfo = mym.import_star_info(dir_root)
        if 'DRS' in sinfo['Ra'].keys():
            sinfo['Ra']['fixed'] = sinfo['Ra']['DRS']
            sinfo['Dec']['fixed'] = sinfo['Dec']['DRS']
            pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))
            ra = sinfo['Ra']['DRS']
            dec = sinfo['Dec']['DRS']
        else:
            ra = mym.ra_to_deg(sinfo['Ra']['fixed'].replace(' ',''))
            dec = mym.dec_to_deg(sinfo['Dec']['fixed'].replace(' ',''))
        dace_summary = pd.read_csv(dir_root+'DACE_TABLE/Dace_extracted_table.csv',index_col=0)
        dace_summary['RA'] = np.round(ra,6) ; dace_summary['DEC'] = np.round(dec,6)
        dace_summary.to_csv(dir_root+'DACE_TABLE/Dace_extracted_table.csv')

    def preprocess(self):
        ins = self.sy_instrument
        files = self.sy_files
        dir_root = self.sy_dir_root

        if self.sy_rassine_db==False:
            if ins[0:6]=='SOPHIE':
                for f in files:
                    mym.read_sophie(f,dir_root,force=True)
            elif (ins=='HARPS_3.5')|(ins=='HARPS03_3.5')|(ins=='HARPS15_3.5'):
                for f in files:
                    mym.read_sophie(f,dir_root,force=True)
            elif (ins.split('_')[0][0:5]=='HARPS')|(ins.split('_')[0]=='HARPN')|(ins.split('_')[0]=='ESPRESSO'):
                for f in files:
                    mym.read_espresso(f,dir_root,force=True)
            elif ins[0:4]=='NEID':
                for f in files:
                    mym.read_neid(f,dir_root,force=True)
            else:
                for f,w0,dw in zip(files,cval1,cdelta1):
                    mym.read_static(f,dir_root,w0,dw,force=True)
        else:
            print(' [INFO] No preprocessing needed, spectra already in RASSINE format.')

        self.sy_rassine_files = np.sort(glob.glob(dir_root+'WORKSPACE/RASSINE*.p'))

    def check_spectra(self):
        files = self.sy_rassine_files
        sub_dico = self.sy_sub_dico
        summary = mym.import_summary(self.sy_dir_root)

        wave_grid, sts, sts_err = mym.import_sts(files, sub_dico=sub_dico)
        anomalous = np.sum((sts>1.02)|(sts<0),axis=1)*100/len(wave_grid)
        anomalous = np.round(anomalous,0).astype('int')

        summary['anomalous'] = anomalous
        if np.min(anomalous)<5:
            kept = (anomalous<5)
        elif np.min(anomalous)<10:
            kept = (anomalous<10)
        else:
            kept = (anomalous<15)

        print(' [INFO] Number of good spectra = %.0f'%(sum(kept)))
        print(' [INFO] Number of anomalous spectra = %.0f'%(len(kept)-sum(kept)))
        print(' [INFO] criterion = ',anomalous)

        summary['flag1'] = 0
        summary.loc[summary.index[~kept],'flag1'] = 1
        summary.to_csv(self.sy_dir_root+'WORKSPACE/Analyse_summary.csv')

        if np.sum(summary['flag1']==0)==0:
            print(Fore.YELLOW+' [WARNING] No good spectra (Emergency stop)'+Fore.RESET)
            print('\n')
            raise SnakyError('No valid spectra detected (Emergency stop)')
        
    def compute_rv_sys(self,debug=False):
        dir_root = self.sy_dir_root
        sub_dico = self.sy_sub_dico
        star = self.sy_starname

        summary = mym.import_summary(dir_root)
        files = np.array(summary['filename'])[summary['flag1']==0]

        teff,feh,fluxD,warning_hole = mym.yarara_flux_density(files)
        plt.savefig(dir_root+'IMAGES/Teff_approximated.png')
        if warning_hole:
            plt.figure('warning')
            plt.savefig(dir_root+'WARNING/WARNING_Flux_density.png')
            plt.close()
        
        rv_sys = []
        for f in files:
            spec = mym.import_spectrum(f,sub_dico=sub_dico)
            rv_sys1 = mym.yarara_rough_rv_sys(spec,teff=teff,verbose=debug)
            rv_sys.append(rv_sys1)
        rv_sys = np.array(rv_sys)

        rv_sys[abs(rv_sys-np.nanmedian(rv_sys))>50] = np.nan
        rv_sys_std = np.nanstd(rv_sys)
        rv_sys_approx = np.round(np.nanmedian(rv_sys),2)
        print('\n [INFO] Final aproximated RV_sys = %.1f +/- %.1f kms'%(rv_sys_approx,rv_sys_std))

        if debug:
            mym.yarara_check_rv_sys(spec, 15, rv_sys_approx, dir_root=dir_root)

        mask = np.ones(len(rv_sys)).astype('bool')
        if len(rv_sys)>10:
            mask_outliers = abs(rv_sys-np.nanmedian(rv_sys))/myf.mad(rv_sys)
            if np.sum(~(mask_outliers>5))!=0:
                mask = ~(mask_outliers>5)
        
        anomalous = np.array(summary['anomalous'])[mask]
        spec  = mym.import_spectrum(files[mask][np.argmin(anomalous)],sub_dico=sub_dico)
        
        sb_flag2 = False
        if rv_sys_std>10: 
            print(Fore.YELLOW+' [WARNING] RV_SYS RMS high (%.1f km/s), SB flag'%(rv_sys_std)+Fore.RESET)
            sb_flag2 = True
            pd.DataFrame(np.array([files,rv_sys]).T,columns=['files','rv_sys']).to_csv(dir_root+'WARNING/RV_SYS_JITTER.csv')
            rv_sys_approx = mym.yarara_rough_rv_sys(spec,teff=teff,verbose=debug)
        print('\n [INFO] RV_sys initial guess = %.1f +/- %.1f kms'%(rv_sys_approx,rv_sys_std))
        sinfo2,sb_flag1 = mym.yarara_check_rv_sys_wrapper(dir_root, spec, rv_sys_approx, ccf_tag='')

        dace_summary = pd.read_csv(dir_root+'DACE_TABLE/Dace_extracted_table.csv',index_col=0)
        ra_deg = np.nanmedian(dace_summary['RA'])
        dec_deg = np.nanmedian(dace_summary['DEC'])
        
        fwhm, rv_sys, contrast, beta_gnd, sb_flag, rcorr, ccf = sinfo2
        if abs(rv_sys)>600:
            rv_sys=0
        if fwhm>300:
            fwhm=300 
        if teff>8500:
            fwhm=200

        sinfo = mym.import_star_info(dir_root)
        sinfo = myf.update_info_lvl2(sinfo,'FluxD','P05',fluxD[0])
        sinfo = myf.update_info_lvl2(sinfo,'FluxD','P10',fluxD[1])
        sinfo = myf.update_info_lvl2(sinfo,'FluxD','P15',fluxD[2])
        sinfo = myf.update_info_lvl2(sinfo,'FluxD','P20',fluxD[3])
        sinfo = myf.update_info_lvl2(sinfo,'FluxD','P25',fluxD[4])
        sinfo = myf.update_info_lvl2(sinfo,'Rv_sys','SNAKY',rv_sys)
        sinfo = myf.update_info_lvl2(sinfo,'CCF_beta','SNAKY',beta_gnd)
        sinfo = myf.update_info_lvl2(sinfo,'Ra','SNAKY',ra_deg)
        sinfo = myf.update_info_lvl2(sinfo,'Dec','SNAKY',dec_deg)
        sinfo = myf.update_info_lvl2(sinfo,'Teff','fixed',teff)
        sinfo = myf.update_info_lvl2(sinfo,'Teff','FluxD',teff)
        sinfo = myf.update_info_lvl2(sinfo,'CCF_beta','SNAKY',beta_gnd)
        sinfo = myf.update_info_lvl2(sinfo,'FeH','fixed',feh)
        sinfo = myf.update_info_lvl2(sinfo,'FWHM','fixed',fwhm)
        sinfo = myf.update_info_lvl2(sinfo,'Contrast','SNAKY',contrast)

        try:
            sb_flag1 = sb_flag1|mym.yarara_check_sb(dir_root)
        except:
            pass

        sinfo = myf.update_info_lvl2(sinfo,'SB1','SNAKY',int(sb_flag1))
        sinfo = myf.update_info_lvl2(sinfo,'SB2','SNAKY',int(sb_flag2))
        pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))

        if rcorr<0.75:
            print(Fore.YELLOW+' [EMERGENCY STOP] The CCF is unusual. The spectra is unusual.'+Fore.RESET)
            print('\n')
            raise SnakyError('The CCF has not a proper shape.')

        if sb_flag1|sb_flag2:
            print(Fore.YELLOW+' [WARNING] Spectroscopy binary detected, pipeline not designed for them.'+Fore.RESET)
            print('\n')

    def compute_ccf(self, debug=False):
        dir_root = self.sy_dir_root
        sub_dico = self.sy_sub_dico
        star = self.sy_starname

        sinfo = mym.import_star_info(dir_root)
        summary = mym.import_summary(dir_root)

        kept = np.array(1-summary['flag1']) 
        fwhm = sinfo['FWHM']['fixed']
        rv_sys = sinfo['Rv_sys']['SNAKY']
        beta_gnd = sinfo['CCF_beta']['SNAKY']
        if sum(kept)!=0:
            sub = summary.loc[summary['flag1']==0]
            files = np.array(sub['filename'])
            ccf_output = mym.yarara_ccf(dir_root, files, rv_sys, fwhm, beta_gnd, 'G2', debug=debug, sub_dico=sub_dico, ccf_tag='', save=False)
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

        sinfo = mym.import_star_info(dir_root)
        summary = mym.import_summary(dir_root)
        kept = np.array(1-summary['flag1'])*np.array(1-summary['flag2'])
        if sum(kept)!=0:
            files = np.array(summary.loc[kept==1,'filename'])
            ccf_output = mym.yarara_ccf(dir_root, files, rv_sys, fwhm, beta_gnd, 'Magicat', debug=debug, sub_dico=sub_dico, ccf_tag='')
            ccf_output = mym.yarara_ccf(dir_root, files, rv_sys, fwhm, beta_gnd, 'G2', debug=debug, sub_dico=sub_dico, ccf_tag='')
            sinfo['FWHM']['G2'] = np.round(np.nanmedian(ccf_output['fwhm'].y),2)
            ccf_output1 = mym.yarara_ccf(dir_root, files, rv_sys, fwhm, beta_gnd, 'Kitty', ccf_oversampling=3, debug=debug, sub_dico=sub_dico, ccf_tag='',rv_shift=ccf_output['rv'].y)
            sinfo['FWHM']['KITTY'] = np.round(np.nanmedian(ccf_output1['fwhm'].y),2)
            ccf_output2 = mym.yarara_ccf(dir_root, files, rv_sys, fwhm, beta_gnd, 'Garfield', ccf_oversampling=3, debug=debug, sub_dico=sub_dico, ccf_tag='',rv_shift=ccf_output['rv'].y)
            sinfo['FWHM']['GARFIELD'] = np.round(np.nanmedian(ccf_output2['fwhm'].y),2)
            if (np.std(ccf_output['rv'].y)>1000)&(np.median(ccf_output['fwhm'].y)<30):
                sinfo = myf.update_info_lvl2(sinfo,'SB2','SNAKY',1)
                print(Fore.YELLOW+' [EMERGENCY STOP] Spectroscopy binary detected'+Fore.RESET)
                print('\n')
                #force_pre, force_summary, force_rvsys, force_ccf, force_master, force_atmos, force_resolution, force_vsini,force_abs_continuum, force_activity ,force_mhk, force_spectroscopy, force_magcycle, force_cleaning = [False]*14   
            pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))

    def compute_master(self):
        dir_root = self.sy_dir_root
        try:
            ccf_output = mym.import_ccf(dir_root,'G2')
            files = ccf_output['filename']
            rv = ccf_output['rv'].y
        except:
            summary = mym.import_summary(dir_root)
            rv_sys = mym.import_star_info(dir_root)['Rv_sys']['SNAKY']
            files = summary['filename']
            rv = np.ones(len(files))*rv_sys
        master = mym.master_spectrum(files,rv,0)
        material = {'wave':master.x,'reference_spectrum':master.y}
        pickle.dump(material,open(dir_root+'WORKSPACE/Analyse_material.p','wb'))
        #sinfo = yarara_check_rv_sys_wrapper(dir_root,master,0) #check if on 0

    def compute_atmos(self,debug=False):
        dir_root = self.sy_dir_root
        star = self.sy_starname
        sinfo = mym.import_star_info(dir_root)
        master = mym.import_master(dir_root)
        try:
            fwhm = sinfo['FWHM']['G2']
        except:
            fwhm = sinfo['FWHM']['fixed']
        
        ccf_output = mym.import_ccf(dir_root,'G2')
        rv = ccf_output['rv'].y
        rv_sys_correction = np.nanmedian(rv)/1000

        rv_sys = sinfo['Rv_sys']['SNAKY'] - rv_sys_correction
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

    def compute_resolution(self,debug=False):
        dir_root = self.sy_dir_root
        star = self.sy_starname
        ins = self.sy_instrument

        sinfo = mym.import_star_info(dir_root)
        summary = mym.import_summary(dir_root)
        try:
            files = mym.import_ccf(dir_root,'G2')['filename']
        except:
            files = summary['filename']
        berv = np.array(summary.loc[np.in1d(summary['filename'],files),'berv'])
        fwhm_ins, berv_output = mym.yarara_instrumental_resolution(dir_root, files, np.zeros(len(berv)), berv.copy())
        summary = mym.import_summary(dir_root) # to reload updated table
        if np.sum(berv!=berv_output)!=0:
            summary.loc[np.in1d(np.array(summary['filename']),files),'berv_computed'] = berv_output
        output = np.array([files,fwhm_ins]).T
        if ins[0:6]=='SOPHIE':
            newins = np.array([[ins.replace('-HE',''),ins.replace('-HE','').replace('_','-HE_')][int(i>5)] for i in fwhm_ins])
            output[:,-1] = newins
            loc = [np.where(summary['filename']==f)[0][0] for f in output[:,0]]
            summary.loc[loc,'ins'] = output[:,-1]
        if ins[0:4]=='NEID':
            newins = np.array([[ins.replace('-HE',''),ins.replace('-HE','').replace('_','-HE_')][int(i>4)] for i in fwhm_ins])
            output[:,-1] = newins
            loc = [np.where(summary['filename']==f)[0][0] for f in output[:,0]]
            summary.loc[loc,'ins'] = output[:,-1]
        ins_res = np.round(np.nanmedian(fwhm_ins),2)
        sinfo = myf.update_info_lvl2(sinfo,'FWHM','O2',ins_res)
        pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))
        summary.to_csv(dir_root+'WORKSPACE/Analyse_summary.csv')

    def compute_vsini(self, Prot=None, Rs=None, debug=False):
        dir_root = self.sy_dir_root
        star = self.sy_starname
        sub_dico = self.sy_sub_dico

        sinfo = mym.import_star_info(dir_root)
        vsini = mym.yarara_vcat(dir_root, sub_dico=sub_dico, debug=debug, std_bias_kms=0.1) 
        mym.yarara_vsini(dir_root, Prot=Prot, Rs=Rs)
        sinfo = myf.update_info_lvl2(sinfo,'Vsini','SNAKY',np.round(np.nanmean(vsini),2))
        pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))

    def compute_abs_continuum(self, debug=False):
        dir_root = self.sy_dir_root
        star = self.sy_starname
        material = mym.import_material(dir_root)
        template_flux, correction = mym.yarara_correct_continuum_absorption(dir_root)
        material['stellar_template'] = template_flux
        material['correction_factor'] = correction
        pickle.dump(material,open(dir_root+'WORKSPACE/Analyse_material.p','wb'))

    def compute_activity(self, debug=False):
        dir_root = self.sy_dir_root
        star = self.sy_starname
        summary = mym.import_summary(dir_root)
        material = mym.import_material(dir_root)
        sinfo = mym.import_star_info(dir_root)
        rv_sys = sinfo['Rv_sys']['SNAKY']
        fwhm = sinfo['FWHM']['fixed']
        kept = np.array(1-summary['flag1'])*np.array(1-summary['flag2'])
        files = np.array(summary.loc[kept==1,'filename'])
        ccf_output = mym.import_ccf(dir_root,'G2')
        tab_proxies, CT, mask_activity = mym.yarara_activity_index(ccf_output['filename'], rv_sys, ccf_output['rv'].y, material=material, fwhm=fwhm)
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

    def compute_mhk(self, debug=False):
        dir_root = self.sy_dir_root
        star = self.sy_starname
        summary = mym.import_summary(dir_root)
        sinfo = mym.import_star_info(dir_root)

        ccf_output = mym.import_ccf(dir_root,'G2')
        rv = ccf_output['rv'].y
        rv_sys_correction = np.nanmedian(rv)/1000

        rv_sys = sinfo['Rv_sys']['SNAKY'] - rv_sys_correction
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

    def compute_spectroscopy(self, debug=False):
        dir_root = self.sy_dir_root
        star = self.sy_starname
        material = mym.import_material(dir_root)
        sinfo = mym.import_star_info(dir_root)
        rv_sys = sinfo['Rv_sys']['SNAKY']
        master = myc.tableXY(myf.doppler_r(material['wave'],rv_sys*1000)[1],material['reference_spectrum'],0*material['wave'])
        master.interpolate(new_grid=material['wave'],method='linear')

        spectroscopy = {'wave':master.x,'flux':master.y,}
        master = myc.tableXY(myf.doppler_r(material['wave'],rv_sys*1000)[1],material['reference_spectrum']*material['correction_factor'],0*material['wave'])
        master.interpolate(new_grid=material['wave'],method='linear',fill_value=0)

        spectroscopy['flux_corrected'] = master.y

        for kw in sinfo.keys():
            if 'fixed' in sinfo[kw]:
                extracted = sinfo[kw]
                del extracted['fixed']
                spectroscopy[kw] = extracted
        pickle.dump(spectroscopy,open(dir_root+'WORKSPACE/Analyse_spectroscopy.p','wb'))

    def compute_mag_cycle(self, debug=False, rm_source=['DACE','Yu+23']):
        dir_root = self.sy_dir_root
        star = self.sy_starname
        try:
            sinfo = mym.import_star_info(dir_root)
            finch_output = mym.yarara_finch(dir_root, rm_source=rm_source, offset_instrument='no!', ext='_fix_model')
            sinfo = myf.update_info_lvl2(sinfo,'Pmag','SNAKY', finch_output[1])
            pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))
            pickle.dump({
                'Starname':star,
                'Pmag':finch_output[1],
                'Kmag_mean':finch_output[2],
                'Kmag_amp':finch_output[3],
                'Kmag_pred':finch_output[4],
                'Phase_pred':finch_output[5],
                'Phase_pred_side':finch_output[6]}, 
                open(dir_root.replace(ins+'/','ALLINS_MERGED/Pmag_FINCH_info.p'),'wb'))
            finch_output = mym.yarara_finch(dir_root, rm_source=rm_source, offset_instrument='yes', automatic_fit=True, ext='_free_model', predict_samples=[2026,2036])
        except:
            pass

    def cleaning(self):
        mym.clean_light_dir(self.sy_dir_root)

    def reset(self):
        os.system('rm -f '+self.sy_dir_root+'/IMAGES/*')
        os.system('rm -f '+self.sy_dir_root+'/WORKSPACE/Analyse_*')
        os.system('rm -f '+self.sy_dir_root+'/WARNING/*.png')
        os.system('rm -f '+self.sy_dir_root+'/STAR_INFO/*')
        os.system('rm -f '+self.sy_dir_root+'/CCF_MASK/*.fits')


    def reduce(self,
            steps,
            automatic_db = True,
            debug = False, 
            Prot = None,
            Rs = None,
            ):
        


        star = self.sy_starname
        ins = self.sy_instrument
        dir_root = self.sy_self.dir_root

        myf.print_box('\n---- Launching reduction %s with instrument %s  ----\n'%(star,ins))
        time_start = time.time()
        
        filename_time = dir_root + '/REDUCTION_INFO/Time_informations_reduction_snaky_%s.csv'%(timestamp_reduction)

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
        force_reset = bool(np.sum(steps==666))
        force_format = bool(np.sum(steps==777))

        if automatic_db:
            print(' [INFO] Automatic sequence build...')
            force_pre = bool(1-mym.check_force_pre(dir_root))&force_pre
            force_summary = bool(1-mym.check_force_summary(dir_root))&force_summary
            force_rvsys = bool(1-mym.check_force_rvsys(dir_root))&force_rvsys
            force_ccf = bool(1-mym.check_force_ccf(dir_root))&force_ccf
            force_master = bool(1-mym.check_force_master(dir_root))&force_master
            force_atmos = bool(1-mym.check_force_atmos(dir_root))&force_atmos
            force_resolution = bool(1-mym.check_force_resolution(dir_root))&force_resolution
            force_vsini = bool(1-mym.check_force_vsini(dir_root))&force_vsini
            force_abs_continuum = bool(1-mym.check_force_abs_continuum(dir_root))&force_abs_continuum
            force_activity = bool(1-mym.check_force_activity(dir_root))&force_activity
            force_mhk = bool(1-mym.check_force_mhk(dir_root))&force_mhk
            force_spectroscopy = bool(1-mym.check_force_spectroscopy(dir_root))&force_spectroscopy
            force_magcycle = bool(1-mym.check_force_magcycle(dir_root))&force_magcycle
            print(' [INFO] Automatic sequence done!\n')

        write_progress(0, 'init', time_step, savefile=filename_time)

        if force_pre: #1
            if not self.sy_rassine_db:
                self.preprocess()
            write_progress(1, 'pre', time_step, savefile=filename_time)
        qc = mym.check_force_pre(dir_root)

        if force_summary: #2
            if not self.sy_yarara_db:
                self.init_summary()

        try:
            mym.check_and_update_path(dir_root)
            summary = mym.import_summary(dir_root)
            files = np.array(summary['filename'])
        except:
            pass

        if force_summary: #2
            self.check_spectra()
            write_progress(2, 'summary', time_step, savefile=filename_time)
        qc = mym.check_force_summary(dir_root)

        if force_rvsys: #3
            self.compute_rv_sys()
            write_progress(3, 'rv_sys', time_step, savefile=filename_time)
        qc = mym.check_force_rvsys(dir_root)

        try:
            teff = mym.import_star_info(dir_root)['Teff']['FluxD']
            if teff>7500:
                force_ccf, force_vsini, force_activity, force_mhk, force_magcycle = [False]*5
            if teff<4000:
                force_activity, force_mhk, force_magcycle = [False]*3
        except:
            pass

        if force_ccf: #4
            self.compute_ccf()
            write_progress(4, 'ccf', time_step, savefile=filename_time)
        qc = mym.check_force_ccf(dir_root)

        if force_master: #5
            self.compute_master()
            write_progress(5, 'master', time_step, savefile=filename_time)
        qc = mym.check_force_master(dir_root)

        if force_atmos: #6
            self.compute_atmos()
            write_progress(6, 'atmos', time_step, savefile=filename_time)
        qc = mym.check_force_atmos(dir_root)

        if self.sy_sub_dico != 'matching_diff':
            force_resolution = False

        if force_resolution: #7
            self.compute_resolution()
            write_progress(7, 'resolution', time_step, savefile=filename_time)
        qc = mym.check_force_resolution(dir_root)

        if force_vsini: #8
            self.compute_vsini()
            write_progress(8, 'vsini', time_step, savefile=filename_time)
        qc = mym.check_force_vsini(dir_root)

        if force_abs_continuum: #9
            self.compute_abs_continuum()
            write_progress(9, 'abs_continuum', time_step, savefile=filename_time)
        qc = mym.check_force_abs_continuum(dir_root)

        if force_activity: #10
            self.compute_activity()
            write_progress(10, 'activity', time_step, savefile=filename_time)
        qc = mym.check_force_activity(dir_root)

        if force_mhk: #11
            self.compute_mhk()
            write_progress(11, 'mhk', time_step, savefile=filename_time)
        qc = mym.check_force_mhk(dir_root)

        if force_spectroscopy: #12
            self.compute_spectroscopy()
            write_progress(12, 'spectroscopy', time_step, savefile=filename_time)
        qc = mym.check_force_spectroscopy(dir_root)

        if force_magcycle: #13
            self.compute_mag_cycle()
            write_progress(13, 'mag_cycle', time_step, savefile=filename_time)
        qc = mym.check_force_magcycle(dir_root)

        try:
            mym.compare_snaky_atmos(stars=[star])
        except:
            pass

        time_end = time.time()
        duration = np.round((time_end-time_start)/60,2)
        tag_duration = str(int(duration//1))+'m'+str(int((duration%1)*60))+'s'
        print(Fore.CYAN+"\n [INFO] Processing achieved in "+tag_duration+" of dir_root = '"+dir_root+"' \n"+Fore.RESET)

        if force_cleaning: #14
            self.cleaning()
