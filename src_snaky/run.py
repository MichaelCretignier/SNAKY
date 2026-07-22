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
import re
import gc

from colorama import Fore

from . import snaky_variables as myv
from . import snaky_functions as myf
from . import snaky_classes as myc
from . import snaky_main as mym

try:
    from memory_profiler import profile
except:
    pass

# =============================================================================
# MEMORY AND TIME MONITORING
# =============================================================================

__version__ = mym.__version__

#### main reduction

class SnakyError(Exception):
    pass

class start():
    def __init__(self, job_id=0, verbose=True, debug=False):
        self.sy_output_dir = myv.WORKSPACE+'/'
        self.sy_job_id = job_id
        self.warning_printed = 0
        self.prd_ext = ''
        self.debug = debug
        self.sy_user_object = {'Name':None, 'Ra':None, 'Dec':None, 'Rv_sys':None, 'Prot':None, 'Rs':None, 'Ms':None, 'Teff':None, 'Log_g':None, 'FeH':None, 'RHK':None, 'Vsini':None, 'stellar_template':None, 'reference':None}
        self.missing_file = False
        if not verbose:
            myv.VERBOSE = False

    def set_output_dir(self,outputdir=None):

        if outputdir is None:
            outputdir = myv.WORKSPACE
            print(' [INFO] No output directory specified, using default: '+outputdir)

        if outputdir[-1]!='/':
            outputdir = outputdir+'/'
        self.sy_output_dir = outputdir

    def format(self,ra=None,dec=None):
        dir_root = self.sy_dir_root
        starname = self.sy_starname
        ins = self.sy_instrument
        source = self.sy_source_files
        if starname.split('_')[0]!='Sun':
            myv.vprint(' [INFO] Formatting SNAKY with basic minimal information...')
            dace_table = mym.import_dace_table(dir_root)
            files = np.array(dace_table['fileroot'])
            sinfo = mym.import_star_info(dir_root)
            query = mym.extract_header(files, ins, debug=self.debug, dec=dec, ra=ra, sources=source)
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

    def get_atmos_db(self):
        dir_root = self.sy_dir_root
        ins = self.sy_instrument
        filename = dir_root.replace(ins,'ALLINS_MERGED')+'Atmos_all_instruments.csv'
        if os.path.exists(filename):
            db = pd.read_csv(filename,index_col=0)

            myv.vprint('\n [INFO] Current SNAKY atmos DB:\n')
            myv.vprint(db)

            db_values = db.loc[db['ins']=='ALLINS_MERGED']
            print(Fore.GREEN+'\n [QUERY] Atmospheric database queried:'+Fore.RESET)
            teff = int(db_values['teff'])
            logg = float(db_values['logg'])
            feh = float(db_values['feh'])
            ms = float(db_values['ms'])
            rs = float(db_values['rs'])
            self.set_star(teff=teff,logg=logg,feh=feh,ms=ms,rs=rs)
        else:
            print(Fore.YELLOW+' [ERROR] The atmospheric databases is not yet existing.'+Fore.RESET)

    def set_star(self, ra=None, dec=None, rv_sys=None, prot=None, rs=None, ms=None, teff=None, logg=None, feh=None, rhk=None, vsini=None, age=None, reference='user', stellar_template=None):
        sy_user_object = {
            'Name'  : self.sy_starname,
            'Ra'    : ra,
            'Dec'   : dec,
            'Rv_sys': rv_sys,
            'Prot'  : prot,
            'Rs'    : rs,
            'Ms'    : ms,
            'Teff'  : teff,
            'Log_g'  : logg,
            'FeH'   : feh,
            'RHK'   : rhk,
            'Vsini' : vsini,
            'Age'   : age,
            'stellar_template' : None,
            'reference': reference
        }
        
        for kw in sy_user_object:
            if sy_user_object[kw] is not None:
                if sy_user_object[kw]==sy_user_object[kw]:
                    self.sy_user_object[kw] = sy_user_object[kw]
                    myv.vprint(' [INFO] Stellar parameters updated: %s = %s'%(kw,str(sy_user_object[kw])))
        
    def estimate_computation_time(self):
        N = len(self.sy_files)

        time_per_rassine = 15  # 15s per continuum normlisation
        snaky1 = 60            # time to process snaky with N=1
        snaky100 = 38        # time to process snaky with N=100

        rassine_processing = time_per_rassine*N
        snaky_processing = snaky1+(N/100)*snaky100

        total_time = rassine_processing*(1-self.sy_rassine_db) + snaky_processing
        minutes = int(total_time//60)
        secondes = int(total_time-60*minutes)

        total_time_required = str(minutes)+'m'+str(secondes)+'s'
        rassine_time_required = str(int(rassine_processing//60))+'m'+str(int(rassine_processing - 60*(rassine_processing//60)))+'s'
        snaky_time_required = str(int(snaky_processing//60))+'m'+str(int(snaky_processing - 60*(snaky_processing//60)))+'s'

        print(Fore.CYAN+' [INFO] For N=%.0f spectra:'%(N))

        print("\n [INFO] RASSINE computation time: %s %s"%(rassine_time_required,['','(SKIPPED)'][int(self.sy_rassine_db)]))
        print(" [INFO] SNAKY computation time: "+snaky_time_required)
        print("\n [INFO] Total computation time estimated: "+total_time_required+" \n"+Fore.RESET)
        
        self.sy_time_required_est = total_time_required

    def set_dataset(self, starname, ins, files, sub_dico='matching_diff', source=None):
        
        sizes = np.array([os.path.getsize(f) for f in files])
        files = list(np.array(files)[sizes>0])
        if np.sum(sizes==0):
            print(Fore.YELLOW+" [WARNING] %.0f files are empty! \n"%(np.sum(sizes==0))+Fore.RESET)        
        
        if len(files)==0:
            raise SnakyError('The input list of files is empty')

        starname,ins = mym.create_snaky_dir(self.sy_output_dir,starname,ins)
        self.sy_starname = starname
        self.sy_instrument = ins
        self.sy_dir_root = self.sy_output_dir+starname+'/data/s1d/'+ins+'/'
        dir_root = self.sy_dir_root
       
        self.set_star()

        print(Fore.CYAN+" [INFO] (root directory) dir_root = '"+dir_root+"' \n"+Fore.RESET)
        if sub_dico=='matching_mad':
            print(Fore.CYAN+" [INFO] Processing YV1 dataset! \n"+Fore.RESET)        
        elif sub_dico=='matching_instrument':
            print(Fore.CYAN+" [INFO] Processing YVA dataset! \n"+Fore.RESET)        

        self.sy_files = files
        self.sy_sub_dico = sub_dico
        if (type(source)==str)|(source is None):
            self.sy_source_files = [source]*len(files)
        else:
            self.sy_source_files = source

        self.sy_rassine_db = False
        self.sy_yarara_db = False

        if len(files)!=0:
            file_test = self.sy_files[0]
            if file_test.split('/')[-1][0:8]=='RASSINE_':
                self.sy_rassine_db = True
                parent = '/'.join(file_test.split('/')[:-1])
                check_summary = os.path.exists(parent+'/Analyse_summary.csv')
                if (len(file_test.split('/Yarara/'))==2)&(check_summary):
                    self.sy_yarara_db = True

                mask = np.ones(len(self.sy_files)).astype('bool')
                for n,f in enumerate(self.sy_files):
                    if not sub_dico in pd.read_pickle(f).keys():
                        mask[n] = False
                if sum(mask)!=len(mask):
                    print(Fore.YELLOW+' [WARNING] Only %.0f/%.0f contains the correct sub_dico'%(sum(mask),len(mask))+Fore.RESET)
                    self.sy_yarara_db = False
                self.sy_files = list(np.array(self.sy_files)[mask])

            #read fits files an create spectra normalised
            check_files =  np.array([os.path.exists(f) for f in self.sy_files])

            if np.prod(check_files)==0:
                print(Fore.YELLOW+' [EMERGENCY STOP] All the spectra were not found'+Fore.RESET)
                print(' [INFO] Missing files:\n')
                for f in np.array(files)[~check_files]:
                    print(f)            
                raise SnakyError('All the spectra indicated were not found.')

        self.estimate_computation_time()

    def init_workspace(self, ra=None, dec=None, copy_rassine_files=True, rv_mode='RV'):
        dir_root = self.sy_dir_root

        self.set_starinfo()
        self.sy_rv_mode = rv_mode

        if (copy_rassine_files)&(self.sy_rassine_db):
            myv.vprint(' [INFO] Copying RASSINE files...')
            for f in self.sy_files:
                filename = f.split('/')[-1]
                if not os.path.exists(dir_root+'WORKSPACE/'+filename):
                    os.system('cp '+f+' '+dir_root+'WORKSPACE/')
            
        if self.sy_yarara_db:
            file_test = self.sy_files[0]
            self.copy_yarara(file_test.split('WORKSPACE/RASSINE')[0])
        else:
            if self.sy_rassine_db:
                self.sy_files = list(np.sort(glob.glob(dir_root+'WORKSPACE/RASSINE*.p')))

        if not self.sy_yarara_db:
            table = pd.DataFrame(self.sy_files,columns=['fileroot'])
            table.to_csv(dir_root+'DACE_TABLE/Dace_extracted_table.csv')
            self.format(ra=ra, dec=dec)

    def set_summary(self):
        dir_root = self.sy_dir_root
        files = self.sy_rassine_files
        ins = self.sy_instrument
        if (self.sy_yarara_db==False)&(self.sy_rassine_db==False):
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
            if self.debug:
                mym.snaky_help()
                print(files,'\n',inss,'\n',jdb,'\n',berv,'\n',flag)
                print(len(files),len(inss),len(jdb),len(berv),len(flag))
            summary = pd.DataFrame(np.array([files, inss, jdb, berv, flag]).T,columns=['filename','ins','jdb','berv','flag'])
            summary.to_csv(dir_root+'WORKSPACE/Analyse_summary.csv')
        else:
            if (self.sy_rassine_db==True)&(self.sy_yarara_db==False):
                dace_summary = pd.read_csv(dir_root+'DACE_TABLE/Dace_extracted_table.csv',index_col=0)
                summary = dace_summary[['fileroot','ins','rjd','berv']]
                summary['flag'] = 0
                summary = summary.rename(columns={'rjd':'jdb','fileroot':'filename'})
                summary.to_csv(dir_root+'WORKSPACE/Analyse_summary.csv')
            else:
                summary = pd.read_csv(dir_root+'WORKSPACE/Analyse_summary.csv',index_col=0)
            self.sy_rassine_files = np.array(summary['filename'].values)    
        if self.debug:
            print(summary)

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
        
        files1 = np.array(summary['filename'])
        files2 = np.array(self.sy_files)

        raw_dir = os.path.dirname(files2[0])

        files1 = np.array([i.split('/')[-1] for i in files1])
        files2 = np.array([i.split('/')[-1] for i in files2])

        mask = myf.in1d(files1,files2)
        summary = summary.loc[mask].reset_index(drop=True)
        summary['filename'] = np.array([raw_dir+'/'+i for i in files1[mask]])

        summary.to_csv(dir_root+'WORKSPACE/Analyse_summary.csv')
        sinfo = mym.import_star_info(dir_root)
        if 'DRS' in sinfo['Ra'].keys():
            sinfo['Ra']['fixed'] = sinfo['Ra']['DRS']
            sinfo['Dec']['fixed'] = sinfo['Dec']['DRS']
            ra = sinfo['Ra']['DRS']
            dec = sinfo['Dec']['DRS']
        else:
            ra = mym.ra_to_deg(sinfo['Ra']['fixed'].replace(' ',''))
            dec = mym.dec_to_deg(sinfo['Dec']['fixed'].replace(' ',''))
        dace_summary = pd.read_csv(dir_root+'DACE_TABLE/Dace_extracted_table.csv',index_col=0)
        dace_summary['RA'] = np.round(ra,6)
        dace_summary['DEC'] = np.round(dec,6)
        dace_summary.to_csv(dir_root+'DACE_TABLE/Dace_extracted_table.csv')

        for kw in myv.star_info:
            if kw not in sinfo.keys():
                sinfo[kw] = myv.star_info[kw]
        pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))

    def preprocess(self):
        ins = self.sy_instrument
        files = self.sy_files
        dir_root = self.sy_dir_root
        sources = self.sy_source_files
        dace_table = mym.import_dace_table(dir_root)

        rv_shift = np.zeros(len(files))
        if ins.split('_')[0]=='UVES':
            df = pd.DataFrame(files,columns=['fileroot'])
            rv_shift = pd.merge(df,dace_table[['fileroot','berv_computed']],how='left')['berv_computed'].values*1000
        
        if self.sy_rassine_db==False:
            if ins[0:5]=='PEPSI':
                new_summary = mym.pepsi_summary(files,self.sy_dir_root)
                files = new_summary['fileroot'].values
                self.sy_files = files
                self.init_workspace(rv_mode = self.sy_rv_mode)
                dace_table = mym.import_dace_table(dir_root)

            file_opened = []

            for f,source,rv in zip(files,sources,rv_shift):
                if ins[0:6]=='SOPHIE':
                    qc = mym.read_sophie(f,dir_root,force=True)
                elif (ins=='HARPS_3.5')|(ins=='HARPS03_3.5')|(ins=='HARPS15_3.5'):
                    qc = mym.read_sophie(f,dir_root,force=True)
                elif (ins.split('_')[0][0:5]=='HARPS')|(ins.split('_')[0]=='HARPN')|(ins.split('_')[0]=='ESPRESSO'):
                    if source=='ESO':
                        qc = mym.read_eso(f,dir_root,ins.split('_')[0],force=True)
                    elif source=='IA2':
                        qc = mym.read_ia2(f,dir_root,force=True)
                    elif source=='YARARA':
                        qc = mym.read_yarara(f,dir_root,force=True)
                    else:
                        qc = mym.read_espresso(f,dir_root,force=True)
                elif ins[0:5]=='PEPSI':
                    qc = mym.read_pepsi(f,dir_root,force=True)
                elif ins[0:4]=='HERM':
                    qc = mym.read_hermes(f,dir_root,force=True)
                elif ins[0:4]=='NEID':
                    qc = mym.read_neid(f,dir_root,force=True)
                elif source=='GR8':
                    qc = mym.read_gr8(f, dir_root, ins.split('_')[0],force=True)
                elif source=='ESO':
                    qc = mym.read_eso(f, dir_root, ins.split('_')[0], rv_shift=rv, force=True)
                else:
                    qc = 0
                    pass
                    #mym.read_static(f,dir_root,w0,dw,force=True)

                file_opened.append(qc)
            file_opened = np.array(file_opened).astype('bool')
            nb_crash = len(file_opened)-np.sum(file_opened)
            if len(file_opened)!=np.sum(file_opened):
                print(Fore.YELLOW+'\n[WARNING] %.0f files have not been opened correctly and will be removed from the processing...'%(nb_crash)+Fore.RESET)
                print(file_opened.astype('int'))
                self.sy_files = list(np.array(self.sy_files)[file_opened])
                self.sy_source_files = list(np.array(self.sy_source_files)[file_opened])
                dace_table = dace_table.loc[file_opened].reset_index(drop=True)
                dace_table.to_csv(dir_root+'DACE_TABLE/Dace_extracted_table.csv')

            self.sy_rassine_files = np.sort(glob.glob(dir_root+'WORKSPACE/RASSINE*.p'))
            myv.vprint(' [INFO] Venting DACE table in RASSINE files...')
            for rf in self.sy_rassine_files:
                file_to_update = pd.read_pickle(rf)
                arcfile = file_to_update['parameters']['arcfiles'][0]
                if arcfile is not None:
                    entry = dace_table.loc[dace_table['fileroot']==arcfile]
                    file_to_update['parameters']['jdb'] = entry['rjd'].values[0]
                    if entry['berv'].values[0]==entry['berv'].values[0]:
                        file_to_update['parameters']['berv'] = entry['berv'].values[0]
                    else:
                        file_to_update['parameters']['berv'] = entry['berv_computed'].values[0]
                    file_to_update['parameters']['SNR_5500'] = entry['snr'].values[0]
                    pickle.dump(file_to_update,open(rf,'wb'))
        else:
            myv.vprint(' [INFO] No preprocessing needed, spectra already in RASSINE format.')
            self.sy_rassine_files = self.sy_files

    def load_data(self, use_material=False):

        material = None
        rv_sys = 0
        if use_material:
            try:
                material = mym.import_material(self.sy_dir_root)
                correction_factor = material['correction_factor']
                sinfo = mym.import_star_info(self.sy_dir_root)
                rv_sys = sinfo['Rv_sys']['SNAKY']
            except:
                material = None
        
        summary = mym.import_summary(self.sy_dir_root)
        self.sy_rassine_files = np.array(summary['filename'])
        try:
            self.sy_sts_wave, self.sy_sts_flux = mym.create_sts(
                self.sy_rassine_files, 
                sub_dico = self.sy_sub_dico, 
                material = material,
                rv_sys = rv_sys,
                )
        except FileNotFoundError:
            self.missing_file = True
    
    def check_spectra(self):
        files = self.sy_rassine_files
        sub_dico = self.sy_sub_dico
        summary = mym.import_summary(self.sy_dir_root)

        wave_grid, sts, sts_err = mym.import_sts((self.sy_sts_wave,self.sy_sts_flux,files), sub_dico=sub_dico, scale=False)

        del sts_err

        if self.sy_instrument!='UVES':
            anomalous = np.sum((sts>1.02*10000)|(sts<0.01),axis=1)*100/len(wave_grid)
        else:
            anomalous = np.sum((sts>1.02*10000)|((sts<0.01)&(sts!=0.0)),axis=1)*100/len(wave_grid)
        anomalous = np.round(anomalous,0).astype('int')

        empty = np.sum(sts==0.0,axis=1)*100/len(wave_grid)
        empty = np.round(empty,0).astype('int')

        anomalous[empty>75] = 100

        del wave_grid
        del sts

        gc.collect()

        summary['anomalous'] = anomalous
        if np.min(anomalous)<5:
            kept = (anomalous<5)
        elif np.min(anomalous)<10:
            kept = (anomalous<10)
        else:
            kept = (anomalous<[15,40][int(self.sy_instrument.split('_')[0] in ['UVES'])])

        myv.vprint(' [INFO] Number of good spectra = %.0f'%(sum(kept)))
        myv.vprint(' [INFO] Number of anomalous spectra = %.0f'%(len(kept)-sum(kept)))
        myv.vprint(' [INFO] criterion = ',anomalous)

        summary['flag1'] = 0
        summary.loc[summary.index[~kept],'flag1'] = 1
        summary.to_csv(self.sy_dir_root+'WORKSPACE/Analyse_summary.csv')

        if np.sum(summary['flag1']==0)==0:
            print(Fore.YELLOW+' [WARNING] No good spectra (Emergency stop)'+Fore.RESET)
            print('\n')
            raise SnakyError('No valid spectra detected (Emergency stop)')
        
    #@profile
    def compute_rv_sys(self):
        dir_root = self.sy_dir_root
        star = self.sy_starname

        summary = mym.import_summary(dir_root)
        mask_flag0 = (summary['flag1']==0)
        files = np.array(summary['filename'][mask_flag0])

        if len(files)==0:
            print(Fore.YELLOW+' [WARNING] No spectra good enough (Emergency stop)'+Fore.RESET)
            print('\n')
            raise SnakyError('No valid spectra detected (Emergency stop)')

        anomalous = np.array(summary['anomalous'])
        teff,feh,fluxD,warning_hole = mym.yarara_flux_density(dir_root,(self.sy_sts_wave,self.sy_sts_flux[mask_flag0], files))
        
        rv_sys = []
        for n in np.arange(len(summary)):
            spec = myc.tableXY(self.sy_sts_wave/100.,self.sy_sts_flux[n]/10000.,0*self.sy_sts_wave)
            rv_sys1 = mym.yarara_rough_rv_sys(spec,teff=teff,verbose=False)
            rv_sys.append(rv_sys1)
        rv_sys = np.array(rv_sys)
        
        if len(rv_sys)>2:
            mask_out = abs(rv_sys-np.nanmedian(rv_sys))>50
            if np.sum(~mask_out)!=0:
                rv_sys[mask_out] = np.nan
            rv_sys_approx = np.round(np.nanmedian(rv_sys),2)
        else:
            rv_sys_approx = rv_sys[np.argmin(anomalous)]
        
        rv_sys_std = np.nanstd(rv_sys)
        myv.vprint('\n [INFO] Final aproximated RV_sys = %.1f +/- %.1f kms'%(rv_sys_approx,rv_sys_std))

        mask = np.ones(len(rv_sys)).astype('bool')
        if len(rv_sys)>10:
            mask_outliers = abs(rv_sys-np.nanmedian(rv_sys))/myf.mad(rv_sys)
            if np.sum(~(mask_outliers>5))!=0:
                mask = ~(mask_outliers>5)
        
        anomalous = anomalous[mask]

        #spec  = mym.import_spectrum(files[mask][np.argmin(anomalous)],sub_dico=sub_dico)
        #print(np.argmin(anomalous))
        spec = myc.tableXY(self.sy_sts_wave/100.,self.sy_sts_flux[mask][np.argmin(anomalous)]/10000.,0*self.sy_sts_wave)

        sb_flag2 = False
        if rv_sys_std>10: 
            print(Fore.YELLOW+' [WARNING] RV_SYS RMS high (%.1f km/s), SB flag'%(rv_sys_std)+Fore.RESET)
            sb_flag2 = True
            files = [np.array(summary['filename'])]
            pd.DataFrame(np.array([files[-1],rv_sys]).T,columns=['files','rv_sys']).to_csv(dir_root+'WARNING/RV_SYS_JITTER.csv')
            rv_sys_approx = mym.yarara_rough_rv_sys(spec,teff=teff,verbose=self.debug)
        
        if rv_sys_approx!=rv_sys_approx:
            rv_sys_approx=0

        myv.vprint('\n [INFO] RV_sys initi' \
        'al guess = %.1f +/- %.1f kms'%(rv_sys_approx,rv_sys_std))
        sinfo2,sb_flag1 = mym.yarara_check_rv_sys_wrapper(dir_root, spec, rv_sys_approx, ccf_tag='')

        dace_summary = pd.read_csv(dir_root+'DACE_TABLE/Dace_extracted_table.csv',index_col=0)
        ra_deg = np.nanmedian(dace_summary['RA'])
        dec_deg = np.nanmedian(dace_summary['DEC'])
        
        fwhm, rv_sys, contrast, beta_gnd, sb_flag, rcorr, ccf = sinfo2
        if abs(rv_sys)>600:
            rv_sys=0
        if fwhm>300:
            fwhm=300 
        if (teff>8500)&(fwhm<200):
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

        if (rcorr<0.75)&(teff<7500):
            print(Fore.YELLOW+' [EMERGENCY STOP] The CCF is unusual (R = %.2f). The spectra is unusual.'%(rcorr)+Fore.RESET)
            print('\n')
            raise SnakyError('The CCF has not a proper shape.')

        if sb_flag1|sb_flag2:
            print(Fore.YELLOW+' [WARNING] Spectroscopy binary detected, pipeline not designed for them.'+Fore.RESET)
            print('\n')

    #@profile
    def compute_ccf(self, rv_mode=None):
        dir_root = self.sy_dir_root
        sub_dico = self.sy_sub_dico
        star = self.sy_starname
        ins = self.sy_instrument.split('_')[0]

        if rv_mode is None: #either RV or EPRV
            rv_mode = self.sy_rv_mode

        sinfo = mym.import_star_info(dir_root)
        summary = mym.import_summary(dir_root)

        if ins!='UVES':
            kept = np.array(1-summary['flag1']) 
        else:
            kept = (summary['anomalous']<70).astype('int')
        
        fwhm = sinfo['FWHM']['fixed']
        rv_sys = sinfo['Rv_sys']['SNAKY']
        beta_gnd = sinfo['CCF_beta']['SNAKY']
        if sum(kept)!=0:
            mask = kept.astype('bool')
            sub = summary.loc[mask]
            files = np.array(sub['filename'])
            files = (self.sy_sts_wave,self.sy_sts_flux[mask], files)
            ccf_output = mym.yarara_ccf(dir_root, files, rv_sys, fwhm, beta_gnd, 'G2', debug=self.debug, sub_dico=sub_dico, ccf_tag='', save=False, rv_mode=rv_mode)
            ct = ccf_output['contrast'].y
            med_ct = np.nanmedian(ct)
            kept2 = abs(ct-med_ct)<[2,10][int(ins=='UVES')]
            myv.vprint(' [INFO] Number of good spectra (after CCF check) = %.0f'%(sum(kept2)))
            myv.vprint(' [INFO] Number of bad spectra (after CCF check) = %.0f'%(len(kept)-sum(kept)+len(kept2)-sum(kept2)))
            if np.sum(kept2)==0:
                kept2 = np.ones(len(ct)).astype('bool')
            summary['flag2'] = 0
            summary.loc[sub.index[~kept2],'flag2'] = 1
            summary.to_csv(dir_root+'WORKSPACE/Analyse_summary.csv')

        #pickle.dump({
        #    'dir_root':dir_root,
        #    'selfsy_sts_wave':self.sy_sts_wave,
        #    'selfsy_sts_flux':self.sy_sts_flux,
        #    'files':files,
        #    'mask':mask,
        #    'rv_sys':rv_sys,
        #    'fwhm':fwhm,
        #    'beta_gnd':beta_gnd,
        #    },open('/Users/cretignier/Desktop/Snaky/TEST/compute_ccf/export.p','wb'))

        sinfo = mym.import_star_info(dir_root)
        summary = mym.import_summary(dir_root)
        kept = kept*np.array(1-summary['flag2'])
        if sum(kept)!=0:
            mask = np.array(kept==1)
            files = np.array(summary.loc[mask,'filename'])
            files = (self.sy_sts_wave,self.sy_sts_flux[mask],files)
            ccf_output = mym.yarara_ccf(dir_root, files, rv_sys, fwhm, beta_gnd, 'Magicat', debug=self.debug, sub_dico=sub_dico, ccf_tag='')
            del ccf_output
            ccf_output = mym.yarara_ccf(dir_root, files, rv_sys, fwhm, beta_gnd, 'G2', debug=self.debug, sub_dico=sub_dico, ccf_tag='', rv_mode=rv_mode)
            sinfo['FWHM']['G2'] = np.round(np.nanmedian(ccf_output['fwhm'].y),2)

            if np.round(np.nanmedian(ccf_output['fwhm'].y),2)<100:
                ccf_output1 = mym.yarara_ccf(dir_root, files, rv_sys, fwhm, beta_gnd, 'Kitty', ccf_oversampling=3, debug=self.debug, sub_dico=sub_dico, ccf_tag='',rv_shift=ccf_output['rv'].y)
                sinfo['FWHM']['KITTY'] = np.round(np.nanmedian(ccf_output1['fwhm'].y),2)
                del ccf_output1
                ccf_output2 = mym.yarara_ccf(dir_root, files, rv_sys, fwhm, beta_gnd, 'Garfield', ccf_oversampling=3, debug=self.debug, sub_dico=sub_dico, ccf_tag='',rv_shift=ccf_output['rv'].y)
                sinfo['FWHM']['GARFIELD'] = np.round(np.nanmedian(ccf_output2['fwhm'].y),2)
                del ccf_output2
            
            if (np.std(ccf_output['rv'].y)>1000)&(np.median(ccf_output['fwhm'].y)<30):
                sinfo = myf.update_info_lvl2(sinfo,'SB2','SNAKY',1)
                print(Fore.YELLOW+' [EMERGENCY STOP] Spectroscopy binary detected'+Fore.RESET)
                print('\n')
                #force_pre, force_summary, force_rvsys, force_ccf, force_master, force_atmos, force_resolution, force_vsini,force_abs_continuum, force_activity ,force_mhk, force_spectroscopy, force_magcycle, force_cleaning = [False]*14   
            pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))

    #@profile
    def compute_master(self):
        dir_root = self.sy_dir_root
        summary = mym.import_summary(dir_root)
        files = summary['filename']
        try:
            ccf_output = mym.import_ccf(dir_root,'G2')
            files2 = ccf_output['filename']
            mask = myf.in1d(files,files2)
            rv = ccf_output['rv'].y
            files = files2
        except:
            rv_sys = mym.import_star_info(dir_root)['Rv_sys']['SNAKY']
            rv = np.ones(len(files))*rv_sys
            mask = np.ones(len(files)).astype('bool')
        
        files = (self.sy_sts_wave,self.sy_sts_flux[mask],self.sy_sts_flux[mask]*0,files)
        master = mym.master_spectrum(files,rv,0)
        material = {'wave':master.x,'reference_spectrum':master.y}
        pickle.dump(material,open(dir_root+'WORKSPACE/Analyse_material.p','wb'))
        #sinfo = yarara_check_rv_sys_wrapper(dir_root,master,0) #check if on 0

    def compute_atmos(self):
        dir_root = self.sy_dir_root
        star = self.sy_starname
        sinfo = mym.import_star_info(dir_root)
        master = mym.import_master(dir_root)
        try:
            fwhm = sinfo['FWHM']['G2']
        except:
            fwhm = sinfo['FWHM']['fixed']
        
        try:
            ccf_output = mym.import_ccf(dir_root,'G2')
            rv = ccf_output['rv'].y
            rv_sys_correction = 0*np.nanmedian(rv)/1000 ## ???? do I need this if I shift the master already?
            del ccf_output
        except:
            rv_sys_correction = 0
        
        rv_sys = sinfo['Rv_sys']['SNAKY'] - rv_sys_correction

        CT,EW,FLAG = mym.yarara_iron_lines(dir_root, master, fwhm, rv_sys=rv_sys)
        for kw in CT.keys():
            sinfo['Contrast'][kw] = CT[kw]
        for kw in EW.keys():
            sinfo['EW'][kw] = EW[kw]
        pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))

        atmos = mym.yarara_atmos_xgb_spectroscopy(dir_root, sinfo, resolution=80000, phot=True, flag=(FLAG!=0))
        teff,feh,logg,M,R,BV,vmicro,vmacro = atmos

        suffixe = 'ATLAS_T%.0f_g%.1f'%(np.round(teff,-2),np.round(logg,1))
        myv.vprint(' [INFO] Atmospheric model set to : %s'%(suffixe))

        sinfo = myf.update_info_lvl2(sinfo,'Mstar','SNAKY',M)   
        sinfo = myf.update_info_lvl2(sinfo,'Rstar','SNAKY',R)
        sinfo = myf.update_info_lvl2(sinfo,'Teff','SNAKY',teff)
        sinfo = myf.update_info_lvl2(sinfo,'FeH','SNAKY',feh)
        sinfo = myf.update_info_lvl2(sinfo,'Log_g','SNAKY',logg)
        sinfo = myf.update_info_lvl2(sinfo,'BV','SNAKY',BV)
        sinfo = myf.update_info_lvl2(sinfo,'Vmicro','SNAKY',vmicro)
        sinfo = myf.update_info_lvl2(sinfo,'Vmacro','SNAKY',vmacro)
        sinfo = myf.update_info_lvl2(sinfo,'stellar_template','SNAKY',suffixe)

        teff = self.sy_user_object['Teff']

        if (FLAG==0)&(EW['LiI']==EW['LiI']):
            age = mym.yarara_lithium_age(dir_root, teff=teff)
            if age is not None:
                sinfo = myf.update_info_lvl2(sinfo,'Age','Jeffr+23',np.round(age,3))

        pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))

    #@profile
    def compute_resolution(self):
        dir_root = self.sy_dir_root
        star = self.sy_starname
        ins = self.sy_instrument

        sinfo = mym.import_star_info(dir_root)
        summary = mym.import_summary(dir_root)
        files = summary['filename']

        try:
            files2 = mym.import_ccf(dir_root,'G2')['filename']
            mask = myf.in1d(files,files2)
        except:
            mask = np.ones(len(summary)).astype('bool')
        berv = np.array(summary.loc[mask,'berv'])

        #pickle.dump({
        #    'dir_root':dir_root,
        #    'selfsy_sts_wave':self.sy_sts_wave,
        #    'selfsy_sts_flux':self.sy_sts_flux,
        #    'files':files,
        #    'mask':mask,
        #    'berv':berv
        #    },open('/Users/cretignier/Desktop/Snaky/TEST/compute_resolution/export.p','wb'))

        try:
            if self.sy_source_files[0]!='YARARA':
                fwhm_ins, berv_output = mym.yarara_instrumental_resolution(dir_root, (self.sy_sts_wave,self.sy_sts_flux[mask],files[mask]), np.zeros(len(berv)), berv.copy())
            else:
                fwhm_ins = np.nan*berv
                berv_output = berv

            summary = mym.import_summary(dir_root) # to reload updated table
            if np.sum(berv!=berv_output)!=0:
                summary.loc[mask,'berv_computed'] = berv_output
            output = np.array([files[mask],fwhm_ins]).T
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
        except AttributeError:
            ins_res = np.nan

        sinfo = myf.update_info_lvl2(sinfo,'FWHM','O2',ins_res)
        pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))
        summary.to_csv(dir_root+'WORKSPACE/Analyse_summary.csv')

    def compute_vsini(self, Prot=None, Rs=None):
        dir_root = self.sy_dir_root
        star = self.sy_starname
        sub_dico = self.sy_sub_dico
        ins = self.sy_instrument.split('_')[0]
        ref_value = self.sy_user_object['Vsini']

        sinfo = mym.import_star_info(dir_root)
        try:
            ins_res = sinfo['FWHM']['O2']
        except: 
            ins_res = np.nan

        if 'G2' in sinfo['FWHM'].keys():
            fwhm = sinfo['FWHM']['G2']

            if fwhm>30:
                rv_sys = sinfo['Rv_sys']['SNAKY']
                atmos = mym.yarara_atmos_fast_rotator(dir_root, rv_sys, fwhm*0.75, model='SNAKY')
                teff,feh,logg,M,R,BV,vmicro,vmacro,vsini = atmos
                suffixe = 'ATLAS_T%.0f_g%.1f'%(np.round(teff,-2),np.round(logg,1))
                myv.vprint(' [INFO] Atmospheric model set to : %s'%(suffixe))
                sinfo = myf.update_info_lvl2(sinfo,'Mstar','SNAKY',M)   
                sinfo = myf.update_info_lvl2(sinfo,'Rstar','SNAKY',R)
                sinfo = myf.update_info_lvl2(sinfo,'Teff','SNAKY',teff)
                sinfo = myf.update_info_lvl2(sinfo,'FeH','SNAKY',feh)
                sinfo = myf.update_info_lvl2(sinfo,'Log_g','SNAKY',logg)
                sinfo = myf.update_info_lvl2(sinfo,'BV','SNAKY',BV)
                sinfo = myf.update_info_lvl2(sinfo,'Vmicro','SNAKY',vmicro)
                sinfo = myf.update_info_lvl2(sinfo,'Vmacro','SNAKY',vmacro)
                sinfo = myf.update_info_lvl2(sinfo,'stellar_template','SNAKY',suffixe)
                mym.yarara_vsini(dir_root, Prot=Prot, Rs=Rs)
            else:
                if (ins_res==ins_res)|(ins in myv.instrument_res_kms.keys()):
                    try:
                        vsini = mym.yarara_vcat(dir_root, sub_dico=sub_dico, debug=self.debug, std_bias_kms=0.1, ref_value=ref_value) 
                    except FileNotFoundError:
                        pass
                    vsini = np.round(np.nanmean(vsini),2)
                    mym.yarara_vsini(dir_root, Prot=Prot, Rs=Rs)
                else:
                    vsini = np.nan

            sinfo = myf.update_info_lvl2(sinfo,'Vsini','SNAKY',vsini)
            pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))
        else:
            print(Fore.YELLOW+' [ERROR] G2 FWHM missing from the Star_info table'+Fore.RESET)

    #@profile
    def compute_abs_continuum(self):
        dir_root = self.sy_dir_root
        material = mym.import_material(dir_root)
        sinfo = mym.import_star_info(dir_root)

        model = sinfo['stellar_template']['SNAKY']
        if self.sy_user_object['Teff'] is not None:
            teff = self.sy_user_object['Teff']
            model = re.sub(r'T\d+', f'T{teff}', model)
        
        if self.sy_user_object['FeH'] is not None:
            feh = self.sy_user_object['FeH']
        else:
            feh = sinfo['FeH']['SNAKY']
        
        if self.sy_user_object['Rv_sys'] is not None:
            rv_sys = self.sy_user_object['Rv_sys']
        else:
            rv_sys = sinfo['Rv_sys']['SNAKY']

        if self.sy_user_object['Vsini'] is not None:
            vsini = self.sy_user_object['Vsini']
        else:
            try:
                vsini = sinfo['Vsini']['SNAKY']
                if vsini!=vsini:
                    vsini = 0.0
            except:
                vsini = 0.0

        template_flux, correction = mym.yarara_correct_continuum_absorption(dir_root, rv_sys, feh, model, vsini)
        material['stellar_template'] = template_flux
        material['correction_factor'] = correction
        pickle.dump(material,open(dir_root+'WORKSPACE/Analyse_material.p','wb'))

        del template_flux
        del correction
        del material

    #@profile
    def compute_activity(self):
        dir_root = self.sy_dir_root
        star = self.sy_starname
        summary = mym.import_summary(dir_root)
        material = mym.import_material(dir_root)
        sinfo = mym.import_star_info(dir_root)
        rv_sys = sinfo['Rv_sys']['SNAKY']
        fwhm = sinfo['FWHM']['fixed']
        
        ccf_output = mym.import_ccf(dir_root,'G2')
        files = ccf_output['filename']

        mask = myf.in1d(summary['filename'],files)
        files = (self.sy_sts_wave,self.sy_sts_flux[mask],files)
        rv = ccf_output['rv'].y

        del ccf_output

        #pickle.dump({
        #    'rv_sys':rv_sys,
        #    'files':files,
        #    'rv':rv,
        #    'material':material,
        #    'fwhm':fwhm
        #    },open('/Users/cretignier/Desktop/Snaky/TEST/compute_activity/export.p','wb'))

        tab_proxies, CT, mask_activity = mym.yarara_activity_index(files, rv_sys, rv, material=material, fwhm=fwhm)
        material['activity_proxies'] = mask_activity
        pickle.dump(material,open(dir_root+'WORKSPACE/Analyse_material.p','wb'))

        del material

        for kw in CT.keys():
            sinfo['Contrast'][kw] = np.round(CT[kw],5)
        pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))
        for kw in tab_proxies:
            if (kw!='filename')&(kw in summary.keys()):
                del summary[kw]
        summary = pd.merge(summary,tab_proxies,on='filename',how='left')
        summary.to_csv(dir_root+'WORKSPACE/Analyse_summary.csv')

    def compute_mhk(self):
        dir_root = self.sy_dir_root
        star = self.sy_starname
        summary = mym.import_summary(dir_root)
        sinfo = mym.import_star_info(dir_root)
        material = mym.import_material(dir_root)
        wave_min = np.min(material['wave'][material['reference_spectrum']>0])
        
        rv_sys = sinfo['Rv_sys']['SNAKY']

        mask_h = (abs(self.sy_sts_wave/100-myf.doppler_r(myv.Ca2H[0],rv_sys*1000)[0])<2)
        mask_k = (abs(self.sy_sts_wave/100-myf.doppler_r(myv.Ca2H[0],rv_sys*1000)[0])<2)

        mask_lines = (mask_h|mask_k)
        valid_percent = np.sum(self.sy_sts_flux[:,mask_lines]>0,axis=1)*100/np.sum(mask_lines)
        valid_spectra = (valid_percent>50)

        if (wave_min<4000)&(np.sum(valid_spectra)!=0):
            rhk_ref = self.sy_user_object['RHK']

            ccf_output = mym.import_ccf(dir_root,'G2')
            rv = ccf_output['rv'].y
            rv_sys_correction = np.nanmedian(rv)/1000

            rv_sys = rv_sys - rv_sys_correction*0 # ???? rv should already 
            if self.sy_user_object['Teff'] is not None:
                teff = self.sy_user_object['Teff']
            else:
                teff = sinfo['Teff']['SNAKY']

            if self.sy_user_object['Vsini'] is not None:
                vsini = self.sy_user_object['Vsini']
            else:
                try:
                    vsini = sinfo['Vsini']['SNAKY']
                    if vsini!=vsini:
                        vsini = 0.0
                except:
                    vsini = 0.0

            valid_spectra = valid_spectra[np.in1d(summary['filename'],ccf_output['filename'])]

            files = ccf_output['filename'][valid_spectra]
            rv = rv[valid_spectra]

            mask = myf.in1d(summary['filename'],files)
            files = (self.sy_sts_wave,self.sy_sts_flux[mask],files)

            proxy = np.array(summary.loc[mask,'CaII'])        
            dico, rhk, mhk = mym.yarara_activity_mhk(dir_root, files, rv_sys, rv, teff, material, proxy, vsini=vsini)
            
            if rhk==rhk:
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
                prot1 = np.min([np.max([prot1,1]),100])
                prot2 = np.min([np.max([prot2,1]),100])
                prot_vsini = np.min([prot_vsini,100])
                sinfo = myf.update_info_lvl2(sinfo,'Prot','Mamaj+08', prot1)
                sinfo = myf.update_info_lvl2(sinfo,'Prot','Noyes+84', prot2)
                sinfo = myf.update_info_lvl2(sinfo,'Prot','VSINI', prot_vsini)
                pickle.dump(sinfo,open(dir_root+'STAR_INFO/Stellar_info_%s.p'%(star),'wb'))

                mym.plot_mhk(dir_root,rhk_ref=rhk_ref)
                mym.create_finch_db(dir_root,sub_dico=self.sy_sub_dico)

    def compute_spectroscopy(self):
        dir_root = self.sy_dir_root
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

    def compute_mag_cycle(self, rm_source=['DACE','Yu+23']):
        dir_root = self.sy_dir_root
        star = self.sy_starname
        ins = self.sy_instrument
        try:
            sinfo = mym.import_star_info(dir_root)
            finch_output = mym.yarara_finch(dir_root, rm_source=rm_source, offset_instrument='no', ext='_fix_model')
            finch_output = mym.yarara_finch(dir_root, rm_source=rm_source, offset_instrument='yes', automatic_fit=True, ext='_free_model', predict_samples=[2026,2036])
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
        except:
            pass

    def compare_snaky_atmos(self):

        dir_root = self.sy_dir_root
        star = self.sy_starname
        ins = self.sy_instrument
        stellar_atmos_user = self.sy_user_object
        
        reference = {'ins':stellar_atmos_user['reference']}
        for kw1,kw2 in zip(['ms','rs','teff','logg','feh','rhk','vsini'],['Ms','Rs','Teff','Log_g','FeH','RHK','Vsini']):
            if kw2 in stellar_atmos_user.keys():
                if stellar_atmos_user[kw2] is not None:
                    reference[kw1] = stellar_atmos_user[kw2]
        
        parent_dir = '/'.join(dir_root.split('/')[:-2])

        count = -1
        files = glob.glob(parent_dir+'/*/WORKSPACE/Analyse_samples*')

        extract = []

        variables = ['ms','rs','teff','logg','feh','vsini','mhk','rhk']
        save = {kw:[] for kw in variables}

        plt.figure(figsize=(18,7))
        plt.subplots_adjust(left=0.06,right=0.96,hspace=0.60,top=0.95,bottom=0.15,wspace=0.30)
        for f in files:
            ins = f.split('/WORKSPACE')[0].split('/')[-1]
            code = ins[0]+ins.split('_')[0][-2:]+'_'+ins.split('_')[1]
            count += 1
            table = pd.read_csv(f)
            extract.append([ins]+list(np.array(table.mean())))
            borders = {'ms':[0,3,3],'rs':[0,3,3],'teff':[3000,8000,0],'logg':[3.5,5.0,3],'feh':[-1.5,0.5,3],'vsini':[0,10,3],'mhk':[-50,200,2],'rhk':[-6,-4,3],'prot':[0,100,1],'sini':[0,1,3]} #min max and digits
            
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

            if kw in reference.keys():
                plt.axhline(y=reference[kw], ls='-.', color='k', alpha=0.7, lw=1)
            plt.savefig(parent_dir+'/ALLINS_MERGED/Atmos_all_instrument.png')

        extract = pd.DataFrame(extract,columns=['ins']+list(table.columns))
        med_values = extract.drop(columns='ins').median(numeric_only=True)
        mean_row = pd.DataFrame([['ALLINS_MERGED'] + med_values.tolist()],columns=extract.columns)
        extract = pd.concat([extract, mean_row], ignore_index=True)
        for kw in list(table.columns):
            if kw in borders.keys():
                extract[kw] = np.round(np.array(extract[kw]),borders[kw][-1]) 

        for k in extract.keys():
            if k not in reference.keys():
                reference[k] = np.nan

        extract = pd.concat([extract, pd.DataFrame([reference])], ignore_index=True)
        extract.to_csv(parent_dir+'/ALLINS_MERGED/Atmos_all_instruments.csv')

    def cleaning(self):
        mym.clean_light_dir(self.sy_dir_root)

    def reset(self, suppression='minimal'):
        if self.warning_printed==1:        
            os.system('rm -f '+self.sy_dir_root+'/IMAGES/*')
            os.system('rm -f '+self.sy_dir_root+'/WORKSPACE/Analyse_*')
            os.system('rm -f '+self.sy_dir_root+'/WARNING/*.png')
            os.system('rm -f '+self.sy_dir_root+'/STAR_INFO/*')
            os.system('rm -f '+self.sy_dir_root+'/REDUCTION_INFO/*.txt')
            os.system('rm -f '+self.sy_dir_root+'/CCF_MASK/*.fits')
            if suppression=='all':
                os.system('rm -f '+self.sy_dir_root+'/WORKSPACE/RASSINE*')
                os.system('rm -f '+self.sy_dir_root+'/DACE_TABLE/*.csv')
                os.system('rm -f '+self.sy_dir_root+'/REDUCTION_INFO/*')
            self.warning_printed = 0
            print(' [INFO] Reduction reset, you can now relaunch the reduction.')
        else:
            print(Fore.YELLOW+'\n [WARNING] Resetting the reduction will erase all the products in:\n') 
            liste = ['IMAGES/*','WORKSPACE/Analyse*','WARNING/*','STAR_INFO/*','REDUCTION_INFO/*','CCF_MASK/*.fits']
            if suppression=='all':
                liste.append('WORKSPACE/RASSINE*')
                liste.append('DACE_TABLE/*.csv')
            for j in liste:
                print(' • '+j) 
            print('\n [WARNING] If you want to reset, please run .reset() again.'+Fore.RESET)
            self.warning_printed += 1

    def monitor_ram(self,stage=None):
        current, peak = tracemalloc.get_traced_memory()
        process = psutil.Process(os.getpid())
        rss = process.memory_info().rss / 1e9

        if stage is None:
            if len(self.memory_history)!=0:
                stage = self.memory_history[-1][1]+0.01
            else:
                stage = 0

        self.memory_history.append([
            stage,
            round(current/1e9,4),
            round(peak/1e9,4),
            round(rss,4)
        ])

        print(Fore.CYAN +
            f"\n [INFO] RAM [Gb] python current/peak: "
            f"{current/1e9:.4f} / {peak/1e9:.4f} | "
            f"total process: {rss:.4f}" +
            Fore.RESET)

    def write_progress(self, stage, step, savefile=None):

        if self.sy_end==14: 
            plt.close('all') 

        #monitoring memory
        self.monitor_ram(stage=stage)
        myf.print_ram(step=step+'='+str(stage))

        now = time.time()
        self.time_step[step] = now 

        table_time = pd.DataFrame(self.time_step.values(),index=self.time_step.keys(),columns=['time_abs'])
        dt = np.hstack([0,np.diff(table_time['time_abs'])])
        table_time['time_step_min'] = dt
        table_time['frac_time']=100*table_time['time_step_min']/np.sum(table_time['time_step_min'])
        table_time['time_step_min'] /= 60 #convert in minutes

        table_time['stage'] = np.array(self.memory_history)[:,0]
        table_time['RAM_gb'] = np.array(self.memory_history)[:,1]
        table_time['RAM_peak_gb'] = np.array(self.memory_history)[:,2]
        table_time['RAM_all_gb'] = np.array(self.memory_history)[:,3]

        table_time[['time_step_min','frac_time','RAM_gb','RAM_peak_gb','RAM_all_gb']] = np.round(table_time[['time_step_min','frac_time','RAM_gb','RAM_peak_gb','RAM_all_gb']],2)

        os.makedirs('/'.join(savefile.split('/')[:-1]), exist_ok=True)

        if savefile is not None:
            table_time.to_csv(savefile)

        if step=='end':
            print('\n',table_time[['time_step_min','frac_time','stage','RAM_peak_gb','RAM_all_gb']],'\n')
            time_start = table_time.loc['begin']['time_abs']
            time_end = table_time.loc['end']['time_abs']
            table_time['time_abs'] = table_time['time_abs'] - time_start
            duration = np.round((time_end-time_start)/60,2)
            tag_duration = str(int(duration//1))+'m'+str(int((duration%1)*60))+'s'
            print(Fore.CYAN+"\n [INFO] Processing achieved in "+tag_duration+Fore.RESET)
            print(Fore.CYAN+" [INFO] Processing time was estimated initially: "+self.sy_time_required_est+" \n"+Fore.RESET)
            
            plt.figure(figsize=(14,8))
            plt.subplot(2,1,1) ; plt.ylabel('Computation time [min]')
            plt.plot(np.arange(len(table_time)),table_time['time_step_min'].values,marker='o',color='k',alpha=0.9)
            plt.xticks(np.arange(len(table_time)))
            for x,y,t in np.array(table_time[['stage','time_step_min','frac_time']][1:]):
                plt.text(x+1.75,y,'%.0f%%'%(t),ha='left',va='bottom')
            plt.tick_params(direction='inout',top=True,right=True,labelbottom=False)
            plt.grid()

            RAM_max = np.max(np.array(table_time[['RAM_peak_gb','RAM_all_gb']]))
            RAM_max1 = np.max(np.array(table_time['RAM_peak_gb']))
            RAM_max2 = np.max(np.array(table_time['RAM_all_gb']))
            Ntot = len(self.sy_files)
            plt.title('Total time = %s minutes | RAM maximum = %.1f Gb | N files = %.0f'%(tag_duration, RAM_max, Ntot))

            plt.subplot(2,1,2) ; plt.ylabel('RAM [GB]')
            plt.plot(np.arange(len(table_time)),table_time['RAM_peak_gb'].values,marker='o',color='k')
            plt.plot(np.arange(len(table_time)),table_time['RAM_all_gb'].values,marker='o',color='gray')
            plt.xticks(np.arange(len(table_time)),labels=list(table_time.index),rotation=60,ha='right')
            plt.tick_params(direction='inout',top=True,right=True)
            plt.grid()
            plt.subplots_adjust(bottom=0.15,top=0.95,hspace=0.10)
            plt.savefig(savefile.replace('.csv','_N%s_TIME%s_GBP%.1f_GBT%.1f.png'%(str(Ntot).zfill(4),tag_duration,RAM_max1,RAM_max2)))

    #@profile
    def reduce(self,
            begin=1,
            end=14,
            automatic_db = False,
            atmos_db = False,
            rv_mode = 'RV', #either RV or EPRV 
            copy_rassine_files = True,
            ):
        
        """
        Processing Sequence
        -------------------

        1.  preprocessing
            Read the input spectrum (FITS format) and initialize the reduction with RASSINE

        2.  set_summary
            Extract and store relevant header metadata.

        3.  compute_rvsys
            Estimate the systemic radial velocity (RV).

        4.  compute_ccf
            Compute radial velocities using the Cross-Correlation Function (CCF).

        5.  compute_master
            Build the master spectrum from individual exposures.

        6.  compute_resolution
            Estimate the instrumental resolution (using step 02 calibration).

        7.  compute_atmos
            Derive stellar atmospheric parameters (Teff, logg, [Fe/H]).

        8.  compute_vsini
            Estimate the projected rotational velocity (v sin i).

        9.  compute_abs_continuum
            Apply absolute continuum correction (blue region normalization).

        10. compute_activity
            Compute chromospheric activity indicators.

        11. compute_mhk
            Compute the MHK magnetic activity index.

        12. compute_spectroscopy
            Generate the final master spectrum (SRF product).

        13. compute_magcycle
            Perform FINCH magnetic cycle analysis.

        14. compute_cleaning
            Remove intermediate products and finalize outputs.
        """

        if end<begin:
            end=begin

        self.sy_begin = begin
        self.sy_end = end

        star = self.sy_starname
        ins = self.sy_instrument
        dir_root = self.sy_dir_root

        debug = self.debug

        timestamp_reduction = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        filename_time = dir_root + 'REDUCTION_INFO/Time_info_reduction_snaky_%s_B%sE%s_%s.csv'%(__version__,str(begin).zfill(2),str(end).zfill(2),timestamp_reduction)

        steps = np.arange(begin,end+1,1).astype('int')

        tracemalloc.start()
        top_stats = []
        self.memory_history = [[-99.9,0,0,0]]
        begin = time.time()
        self.time_step = {'init':begin}

        self.write_progress(-1, 'start', savefile=filename_time)

        myf.print_box('\n---- Launching reduction %s with instrument %s  ----\n'%(star,ins))
        time_start = time.time()
        
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

        if debug==True:
            force_cleaning = False

        if automatic_db:
            print(' [INFO] Automatic sequence build...')
            force_pre = bool(1-mym.check_force_pre(dir_root,step_nb='(1)'))&force_pre
            force_summary = bool(1-mym.check_force_summary(dir_root,step_nb='(2)'))&force_summary
            force_rvsys = bool(1-mym.check_force_rvsys(dir_root,step_nb='(3)'))&force_rvsys
            force_ccf = bool(1-mym.check_force_ccf(dir_root,step_nb='(4)'))&force_ccf
            force_master = bool(1-mym.check_force_master(dir_root,step_nb='(5)'))&force_master
            force_atmos = bool(1-mym.check_force_atmos(dir_root,step_nb='(6)'))&force_atmos
            force_resolution = bool(1-mym.check_force_resolution(dir_root,step_nb='(7)'))&force_resolution
            force_vsini = bool(1-mym.check_force_vsini(dir_root,step_nb='(8)'))&force_vsini
            force_abs_continuum = bool(1-mym.check_force_abs_continuum(dir_root,step_nb='(9)'))&force_abs_continuum
            force_activity = bool(1-mym.check_force_activity(dir_root,step_nb='(10)'))&force_activity
            force_mhk = bool(1-mym.check_force_mhk(dir_root,step_nb='(11)'))&force_mhk
            force_spectroscopy = bool(1-mym.check_force_spectroscopy(dir_root,step_nb='(12)'))&force_spectroscopy
            force_magcycle = bool(1-mym.check_force_magcycle(dir_root,step_nb='(13)'))&force_magcycle
            print(' [INFO] Automatic sequence done!\n')
            print(' [INFO] Reduction launched, wait...\n')

        if atmos_db:
            self.get_atmos_db()
        
        ra = self.sy_user_object['Ra']
        dec = self.sy_user_object['Dec']
        Prot = self.sy_user_object['Prot']
        Rs = self.sy_user_object['Rs']

        self.write_progress(0, 'begin', savefile=filename_time)
        
        if force_pre: #1
            self.init_workspace(ra=ra, dec=dec, copy_rassine_files=copy_rassine_files, rv_mode=rv_mode)
            try:
                self.preprocess()
            except KeyError:
                raise SnakyError('[ERROR] Preprocessing failed. Some ESO files miss the expected keywords.')
            self.write_progress(1, 'pre', savefile=filename_time)
        qc1 = mym.check_force_pre(dir_root,step_nb='(1)')

        if force_summary: #2
            self.set_summary()

        try:
            mym.check_and_update_path(dir_root)
            summary = mym.import_summary(dir_root)
            files = np.array(summary['filename'])
        except:
            pass


        self.load_data()

        if force_summary: #2
            self.check_spectra()
            self.write_progress(2, 'summary', savefile=filename_time)
        qc2 = mym.check_force_summary(dir_root,step_nb='(2)')

        if force_rvsys: #3
            self.compute_rv_sys()
            self.write_progress(3, 'rv_sys', savefile=filename_time)
        qc3 = mym.check_force_rvsys(dir_root,step_nb='(3)')

        try:
            teff = mym.import_star_info(dir_root)['Teff']['FluxD']
            fwhm = mym.import_star_info(dir_root)['FWHM']['fixed']
            if teff>8000:
                force_activity, force_mhk, force_magcycle = [False]*3
                print(Fore.YELLOW+' [TRIGGER] Teff > 8000, ACT + MHK + MAG skipped'+Fore.RESET)
            elif teff>7500:
                force_activity, force_mhk, force_magcycle = [False]*3
                print(Fore.YELLOW+' [TRIGGER] Teff > 7500, ACT + MHK + MAG skipped'+Fore.RESET)
            if teff<4000:
                force_resolution, force_activity, force_mhk, force_magcycle = [False]*4
                print(Fore.YELLOW+' [TRIGGER] Teff < 4000, RES + ACT + MHK + MAG skipped'+Fore.RESET)
        except:
            pass

        if force_ccf: #4
            self.compute_ccf()
            self.write_progress(4, 'ccf', savefile=filename_time)
        qc4 = mym.check_force_ccf(dir_root,step_nb='(4)')

        if force_master: #5
            self.compute_master()
            self.write_progress(5, 'master', savefile=filename_time)
        qc5 = mym.check_force_master(dir_root,step_nb='(5)')

        try:
            material = mym.import_master(dir_root)
            del material
        except:
            force_atmos, force_resolution, force_vsini, force_abs_continuum, force_activity, force_mhk, force_spectroscopy, force_magcycle, force_cleaning = [False]*9

        if force_atmos: #6
            self.compute_atmos()
            self.write_progress(6, 'atmos', savefile=filename_time)
        qc6 = mym.check_force_atmos(dir_root,step_nb='(6)')

        if self.sy_sub_dico != 'matching_diff':
            force_resolution = False

        if force_resolution: #7
            self.compute_resolution()
            self.write_progress(7, 'resolution', savefile=filename_time)
        qc7 = mym.check_force_resolution(dir_root,step_nb='(7)')

        if force_vsini: #8
            if qc4==1:
                self.compute_vsini(Prot=Prot, Rs=Rs)
                self.write_progress(8, 'vsini', savefile=filename_time)
        qc8 = mym.check_force_vsini(dir_root,step_nb='(8)')

        if force_abs_continuum: #9
            if qc5==1:
                self.compute_abs_continuum()
                self.write_progress(9, 'abs_continuum', savefile=filename_time)
        qc9 = mym.check_force_abs_continuum(dir_root,step_nb='(9)')

        self.load_data(use_material=True)

        if force_activity: #10
            self.compute_activity()
            self.write_progress(10, 'activity',savefile=filename_time)
        qc10 = mym.check_force_activity(dir_root,step_nb='(10)')

        if force_mhk: #11
            self.compute_mhk()
            self.write_progress(11, 'mhk', savefile=filename_time)
        qc11 = mym.check_force_mhk(dir_root,step_nb='(11)')

        if force_spectroscopy: #12
            self.compute_spectroscopy()
            self.write_progress(12, 'spectroscopy', savefile=filename_time)
        qc12 = mym.check_force_spectroscopy(dir_root,step_nb='(12)')

        try: #Until a proper test condition is implemented
            if force_magcycle: #13
                self.compute_mag_cycle()
                self.write_progress(13, 'mag_cycle', savefile=filename_time)
            qc13 = mym.check_force_magcycle(dir_root,step_nb='(13)')
        except:
            pass

        try:
            if end>6:
                self.compare_snaky_atmos()
        except:
            pass

        self.write_progress(14, 'end', savefile=filename_time)

        print(Fore.CYAN+"\n [INFO] dir_root = '"+dir_root+"' \n"+Fore.RESET)

        if force_cleaning: #14
            self.cleaning()


def benchmark1(output_dir):
    # Benchmark Dataset1 (HARPS Epsilon Eridani)
    files = glob.glob(myv.TEST_DATASET1)

    job = start()
    job.set_output_dir(output_dir)
    job.set_dataset('HD123456','HARPS03_3.5',files) 
    job.warning_printed = 1
    job.reset(suppression='all')

    job.reduce(begin=1, end=14)

def benchmark2(output_dir):
    # Benchmark Dataset2
    files = glob.glob(myv.TEST_DATASET2)
    
    job = start()
    job.set_output_dir(output_dir)
    job.set_dataset('HD128621','HARPS15_3.3.6',files) 
    job.warning_printed = 1
    job.reset(suppression='all')

    job.set_star(ra=219.90, dec=-60.84, prot=36) # ra and dec in degrees (prot optional)
    job.reduce(begin=1, end=14,  copy_rassine_files=True) 

def benchmark3(output_dir):
    # Benchmark Dataset3
    files = glob.glob(myv.TEST_DATASET3)
    
    job = start()
    job.set_output_dir(output_dir)
    job.set_dataset('HD128621','HARPS03_3.3.6',files) 
    job.warning_printed = 1
    job.reset(suppression='all')

    job.set_star(ra=219.90, dec=-60.84, prot=36) # ra and dec in degrees (prot optional)
    job.reduce(begin=1, end=14,  copy_rassine_files=True) 

