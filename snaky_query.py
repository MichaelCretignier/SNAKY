import code
import getopt
import sys
import os
from pathlib import Path

this_file = Path(__file__).resolve()
current_dir = this_file.parent

#cwd = os.getcwd()
#root = '/'.join(cwd.split('/')[:-1])

import numpy as np
import pandas as pd
import glob as glob
from colorama import Fore
import matplotlib.pylab as plt
from datetime import datetime

import src_snaky.snaky_utilities as snaky_util
import src_snaky.run as snaky
import src_snaky.snaky_public_query as public_query
import src_snaky.run_db_analysis as snaky_db

from src_snaky.run import SnakyError

field_of_view = 2 #arcmin
N_spectra = 5 #number of spectra per instrument to download and process
begin = 1
end = 14
automatic_db = True
multiprocess_split = 1
multiprocess_index = 1
instrument = None
force_query = False
stars_to_process = None
star_to_process = None
output_root = None
verbose = True
debug = False
job_id = np.random.choice(np.arange(1000000),1)[0]
reference = 'user'
use_reference = True
meta = True
db_check = True

if len(sys.argv)>1:
    optlist,args =  getopt.getopt(sys.argv[1:],'p:f:n:P:b:e:a:i:F:s:S:o:v:H:J:r:m:D:R:')
    for j in optlist:
        if j[0] == '-P':
            multiprocess_split = int(j[1])
        if j[0] == '-p':
            multiprocess_index = int(j[1])
        if j[0] == '-f':
            field_of_view = float(j[1])
        if j[0] == '-n':
            N_spectra = int(j[1])
        if j[0] == '-b':
            begin = int(j[1])
        if j[0] == '-e':
            end = int(j[1])
        if j[0] == '-a':
            automatic_db = bool(int(j[1]))
        if j[0] == '-i':
            instrument = j[1].split(',')
        if j[0] == '-F':
            force_query = bool(int(j[1]))
        if j[0] == '-o':
            output_root = Path(j[1]).expanduser().resolve()
        if j[0] == '-s':
            stars_to_process = j[1].split(',')
        if j[0] == '-S':
            star_to_process = j[1]
        if j[0] == '-v':
            verbose = bool(int(j[1]))
        if j[0] == '-H':
            debug = bool(int(j[1]))
        if j[0] == '-J':
            job_id = int(j[1])
        if j[0] == '-r':
            reference = str(j[1])
        if j[0] == '-m':
            meta = bool(int(j[1]))
        if j[0] == '-D':
            db_check = bool(int(j[1]))
        if j[0] == '-R':
            use_reference = bool(int(j[1]))

# CONFIG PART

if (len(stars_to_process)==1)&(stars_to_process[0][-3:]=='csv'):
    file_db = stars_to_process[0]
    csv_file = Path(file_db).name
    db = pd.read_csv(stars_to_process[0],index_col=0) #optional keywords teff,logg,feh,logRHK,ms,rs,prot
    db = snaky_util.format(db) 
    db.loc[db['paper'].isna(),'paper'] = reference
    if output_root is None:
        output_root = Path(file_db).parent
else:
    db = pd.DataFrame(np.array([stars_to_process]).T,columns=['starname'])
    db = snaky_util.format(db) 
    db.loc[db['paper'].isna(),'paper'] = reference

db = snaky_util.fill_coordinates(db)
db.dropna(subset=['RA','DEC'],inplace=True)

if output_root is None:
    output_root = current_dir
    output_dir = current_dir / "snaky_data" / "SPECTRA_DB"
    output_dir_snaky = current_dir / "snaky_data"
else:
    output_dir = output_root / "SPECTRA_DB"
    output_dir_snaky = output_root / "SNAKY_DB_SPECTRA"

BLUE = "\033[94m"
RESET = "\033[0m"

print(f" [INFO] Default output dir for downloaded data is:",Fore.CYAN+f"{output_dir}"+Fore.RESET)
print(f" [INFO] Default output dir for processed SNAKY data is:",Fore.CYAN+f"{output_dir_snaky}\n"+Fore.RESET)

db2 = db.copy()
#db2 = db.drop_duplicates(subset=['starname']) # no more used in case multi source in same csv file
if star_to_process is not None:
    db2 = db2.loc[db2['starname']==star_to_process]

db2 = db2.loc[db2['starname']==(db2['starname'])]

indices = np.array_split(np.arange(len(db2)), multiprocess_split)
db3 = [db2.iloc[idx] for idx in indices]
db3 = db3[multiprocess_index-1] #multiprocessing splitting

# SPECTRA DOWNLOADING

idxs = np.ravel([db3.index.values])

# public_query.query_eso('51Peg', ra=None, dec=None, output_dir=output_dir, selection='os',search_by='coordinates', N_spectra = 1, fov=2, download=True)
print('\n [INFO] Querying ESO archive for stars in the database and downloading spectra...\n')
for i in idxs:
    s,ra,dec = db2.loc[i,['starname','RA','DEC']]
    if (not (output_dir / s / "HARPS").exists())|(force_query):
        print(f' [INFO] {s} downloading...')
        public_query.query_eso(
            s, 
            ra = ra, 
            dec = dec, 
            output_dir = output_dir,
            search_by = 'coordinates',
            selection = 'random',
            fov = field_of_view,
            N_spectra = N_spectra,
            download = True)
    else:
        print(f' [INFO] {s} spectra already downloaded, skipping...')

# public_query.query_tng('51Peg', ra=None, dec=None, output_dir=output_dir, N_spectra = 1, fov=2 download=True)
print('\n [INFO] Querying TNG archive for stars in the database and downloading spectra...\n')
servor_down_counter = 0
for i in idxs:
    s,ra,dec = db2.loc[i,['starname','RA','DEC']]
    if (not (output_dir / s / "HARPN").exists())|(force_query):
        print(f' [INFO] {s} downloading...')
        state = public_query.query_tng(
            s, 
            ra = ra, 
            dec = dec, 
            output_dir = output_dir,
            fov = field_of_view,
            selection = 'random',
            N_spectra = N_spectra,
            download = True)

        if state==0:
            servor_down_counter = 0
        else:
            servor_down_counter += 1

        if servor_down_counter==5:
            print(Fore.YELLOW + ' [WARNING] Failed to download from TNG 5 times in row, stopping...' + Fore.RESET)
            break
    else:
        print(f' [INFO] {s} spectra already downloaded, skipping...')

print('\n [INFO] Querying IAC archive for stars in the database and downloading spectra...\n')
for i in idxs:
    s,ra,dec = db2.loc[i,['starname','RA','DEC']]
    if (not (output_dir / s / "HERMES").exists())|(not (output_dir / s / "FIES").exists()):
        print(f' [INFO] {s} downloading...')
        public_query.query_iac(
            s, 
            ra = ra, 
            dec = dec, 
            output_dir = output_dir,
            fov = field_of_view,
            selection = 'random',
            N_spectra = N_spectra,
            download = True)
    else:
        print(f' [INFO] {s} spectra already downloaded, skipping...')

# public_query.query_sophie('51Peg', ra=None, dec=None, output_dir=output_dir, N_spectra = 1, fov=2, download=True)
print('\n [INFO] Querying SOPHIE archive for stars in the database and downloading spectra...\n')
for i in idxs:
    s,ra,dec = db2.loc[i,['starname','RA','DEC']]
    if not (output_dir / s / "SOPHIE").exists():
        print(f' [INFO] {s} downloading...')
        public_query.query_sophie(
            s, 
            ra = ra, 
            dec = dec, 
            output_dir = output_dir,
            fov = field_of_view,
            selection = 'random',
            N_spectra = N_spectra,
            download = True)
    else:
        print(f' [INFO] {s} spectra already downloaded, skipping...')


# SNAKY PROCESSING

snaky_ins = {
    'HARPS':'HARPS_3.8',
    'ESPRESSO':'ESPRESSO_3.3.6',
    'HARPN':'HARPN_3.0.1',
    'UVES':'UVES_1.0',
    'FIES':'FIES_1.0',
    'FEROS':'FEROS_1.0',
    'HERMES':'HERMES_1.0',
    'SOPHIE':'SOPHIE_1.0',
    'SOPHIE-HE':'SOPHIE-HE_1.0',
    'NEID':'NEID_1.0',
    'NEID-HE':'NEID-HE_1.0',
    'EXPRES':'EXPRES_1.0',
    }

print('\n [INFO] SNAKY processing...')

processed_db = pd.DataFrame([['','','',np.nan,np.nan,np.nan,'']],columns=['starname','ins','snaky_version','job_id','processed','time','csv_file'])
table_db = processed_db.copy()
if (stars_to_process[0][-3:]=='csv'):
    if not os.path.exists(output_dir.parent / 'SNAKY_processed_db.csv'):
        processed_db.to_csv(output_dir.parent / 'SNAKY_processed_db.csv')
    else:
        table_db = pd.read_csv(output_dir.parent / 'SNAKY_processed_db.csv',index_col=0)

def update_db(processed_db):
    if os.path.exists(output_dir.parent / 'SNAKY_processed_db.csv'):
        table_db = pd.read_csv(output_dir.parent / 'SNAKY_processed_db.csv',index_col=0)
        table_db = pd.concat([table_db, processed_db], ignore_index=True)
        table_db = table_db.sort_values(by=['snaky_version','starname','ins','time'],ascending=[False, True, True, False]).dropna(subset=['time']).reset_index(drop=True)
        table_db = table_db[['starname','ins','snaky_version','job_id','processed','time','csv_file']]
        table_db = (table_db.drop_duplicates(subset=['starname', 'ins', 'snaky_version'],keep='first').reset_index(drop=True))
        table_db.to_csv(output_dir.parent / 'SNAKY_processed_db.csv')
        processed_db = pd.DataFrame([['','','',np.nan,np.nan,np.nan,'']],
                                    columns=[
                                        'starname',
                                        'ins',
                                        'snaky_version',
                                        'job_id',
                                        'processed',
                                        'time',
                                        'csv_file'])
        print(Fore.CYAN+' [INFO] DB updated'+Fore.RESET)
        return processed_db, table_db

skip_reduction = 0
counter_processing = 0
for idx in idxs:
    s,ra,dec,teff,rhk,logg,feh,ms,rs,vsini,prot,age,reference = np.array(db3.loc[idx,['starname','RA','DEC','teff','logRHK','logg','feh','ms','rs','vsini','prot','age','paper']])
    teff_std,rhk_std,logg_std,feh_std,ms_std,rs_std,vsini_std,prot_std,age_std = np.array(db3.loc[idx,['teff_std','logRHK_std','logg_std','feh_std','ms_std','rs_std','vsini_std','prot_std','age_std']])

    params = [teff,rhk,logg,feh,ms,rs,vsini,prot,age]
    params_std = [teff_std,rhk_std,logg_std,feh_std,ms_std,rs_std,vsini_std,prot_std,age_std]
    teff,rhk,logg,feh,ms,rs,vsini,prot,age = [[p1,p2] for p1,p2 in zip(params,params_std)] # story the uncertainty

    if instrument is None:
        ins = sorted({p.name for p in (output_dir / s).iterdir()})
    else:
        ins = instrument

    ins = np.sort(ins)
    for i in ins:
        if i in snaky_ins.keys():
            files = list((output_dir / s / i).glob("*.fits"))
            files = [str(f) for f in files]
            i2 = snaky_ins[i]
            products = list((output_dir_snaky / s / "data" / "s1d" / i2 / "WORKSPACE").glob("RASSINE_*.p*"))
            products = [str(f) for f in products]
            if len(products)!=0:
                files = list(products)
    
            if i=='HARPN':
                source = ['IA2']*len(files)
            elif (i=='HERMES'):
                source = ['IAC']*len(files)
            elif (i=='SOPHIE')|(i=='NEID')|(i=='EXPRES'):
                source = ['DRS']*len(files)
            else:
                source = ['ESO']*len(files)

            if (star_to_process is None)&(db_check):
                condition1 = (table_db['starname']==s)
                condition2 = (table_db['ins']==i)
                condition3 = (table_db['processed']==1)
                condition4 = (table_db['snaky_version']==snaky.__version__)
                skip_reduction = np.sum(condition1&condition2&condition3&condition4)

            if len(files)>0:
                source = snaky_util.format_source(source,files)
                job = snaky.start(verbose=verbose, debug=debug, job_id=job_id)
                job.set_output_dir(str(output_dir_snaky))
                job.set_star(starname=s, ra=ra, dec=dec, teff=teff, rhk=rhk, ms=ms, rs=rs, logg=logg, feh=feh, vsini=vsini, prot=prot, age=age, reference=reference, use_reference=use_reference)
                QC = 1

                #star_info = glob.glob(str(output_dir_snaky / s / "data" / "s1d" / i2 / "STAR_INFO" / "Stellar_info_*.p"))
                
                if False:#len(star_info)>0:
                    star_info = pd.read_pickle(star_info[0])
                    if 'SNAKY' in star_info['Vsini'].keys():
                        if star_info['Vsini']['SNAKY']<25:
                            continue

                if skip_reduction>0:
                    print(Fore.GREEN+' [INFO] SNAKY already processed with %s for %s according to DB file'%(i, s)+Fore.RESET)
                    continue

                counter_processing += 1
                try:
                    job.set_dataset(s, i2, files, source=source)
                    try:
                        job.reduce(begin=begin, end=end, automatic_db=automatic_db)
                    except (ValueError,KeyError,FileNotFoundError,EOFError):
                        pass
                        #job.reduce(begin=1, end=14, automatic_db=automatic_db)
                except SnakyError:
                    print(Fore.RED+' [ERROR] The process interrupted somewhere with %s for %s'%(i, s)+Fore.RESET)
                    QC = 0    
                if not debug:
                    plt.close('all')
            else:
                print(Fore.YELLOW+' [WARNING] No files found to be SNAKY processed with %s for %s'%(i, s)+Fore.RESET)
                QC = np.nan

            now = current_time = datetime.now().isoformat()

            if (stars_to_process[0][-3:]=='csv'):
                processed_db = pd.concat([processed_db, pd.DataFrame([{
                    'starname': s, 
                    'ins': i, 
                    'snaky_version': snaky.__version__, 
                    'job_id': job_id, 
                    'processed': QC, 
                    'time': now, 
                    'csv_file':csv_file}])], ignore_index=True)

                if (counter_processing%10==9)&(QC==QC):
                    processed_db, table_db = update_db(processed_db)
dustbin = update_db(processed_db)

if (stars_to_process[0][-3:]=='csv')&(meta)&(star_to_process is None):
    plt.close('all')
    init_table = pd.read_csv(stars_to_process[0],index_col=0) ; init_table['REF_ID'] = init_table.index
    db_analysed = snaky_db.analyse_snaky_results(db, str(output_dir_snaky.parent), stars_to_process[0], reference, residual=False)
    db_final = pd.merge(db_analysed, init_table, on='REF_ID', how='left',suffixes=('_snaky','_init'))

    plt.close('all')
    snaky_db.compare_atmos_databases(db, str(output_dir_snaky.parent), csv_file=stars_to_process[0])
    #snaky_db.compare_atmos_databases(db, str(output_dir_snaky.parent), csv_file=None)
    #snaky_db.update_atmos_databases(str(output_dir_snaky.parent), xbg_model='_phot_1.8.0_sklearn_1.7.2')
    
    # import pygwalker as pyg
    # pyg.walk(db_final)
