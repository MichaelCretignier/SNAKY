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
import src_snaky.snaky_utilities as snaky_util
import src_snaky.run as snaky
import src_snaky.snaky_public_query as public_query

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

if len(sys.argv)>1:
    optlist,args =  getopt.getopt(sys.argv[1:],'p:f:n:P:b:e:a:i:F:s:S:o:v:H:')
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
        elif j[0] == '-H':
            debug = bool(int(j[1]))

# CONFIG PART

if (len(stars_to_process)==1)&(stars_to_process[0][-3:]=='csv'):
    file_db = stars_to_process[0]
    db = pd.read_csv(stars_to_process[0],index_col=0) #optional keywords teff,logg,feh,logRHK,ms,rs,prot
    db = snaky_util.format(db) 
    if output_root is None:
        output_root = Path(file_db).parent
else:
    db = pd.DataFrame(np.array([stars_to_process]).T,columns=['starname'])
    db = snaky_util.format(db) 

db = snaky_util.fill_coordinates(db)

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

db2 = db.drop_duplicates(subset=['starname'])
if star_to_process is not None:
    db2 = db2.loc[db2['starname']==star_to_process]

print(db2)

indices = np.array_split(np.arange(len(db2)), multiprocess_split)
db3 = [db2.iloc[idx] for idx in indices]
db3 = db3[multiprocess_index-1] #multiprocessing splitting

# SPECTRA DOWNLOADING

idxs = np.ravel([db3.index.values])

# public_query.query_eso('51Peg', ra=None, dec=None, output_dir=output_dir, selection='os',search_by='coordinates', N_spectra = 1, fov=2, download=True)
print('\n [INFO] Querying ESO archive for stars in the database and downloading spectra...\n')
for i in idxs:
    s,ra,dec = db2.loc[i,['starname','RA','DEC']]
    if (not (output_dir / s).exists())|(force_query):
        print(f' [INFO] {s} downloading...')
        public_query.query_eso(
            s, 
            ra = ra, 
            dec = dec, 
            output_dir = output_dir,
            search_by = 'coordinates',
            selection = 'closest',
            fov = field_of_view,
            N_spectra = N_spectra,
            download = True)
    else:
        print(f' [INFO] {s} spectra already downloaded, skipping...')

# public_query.query_tng('51Peg', ra=None, dec=None, output_dir=output_dir, N_spectra = 1, fov=2 download=True)
print('\n [INFO] Querying TNG archive for stars in the database and downloading spectra...\n')
for i in idxs:
    s,ra,dec = db2.loc[i,['starname','RA','DEC']]
    if (not (output_dir / s / "HARPN").exists())|(force_query):
        print(f' [INFO] {s} downloading...')
        public_query.query_tng(
            s, 
            ra = ra, 
            dec = dec, 
            output_dir = output_dir,
            fov = field_of_view,
            selection = 'closest',
            N_spectra = N_spectra,
            download = True)
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
            selection = 'closest',
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
    'NEID':'NEID_1.0',
    }

print('\n [INFO] SNAKY processing...')

for idx in idxs:
    s,ra,dec,teff,rhk,logg,feh,ms,rs,vsini,prot,age = db3.loc[idx,['starname','RA','DEC','teff','logRHK','logg','feh','ms','rs','vsini','prot','age']]

    if instrument is None:
        ins = sorted({p.name for p in (output_dir / s).iterdir()})
    else:
        ins = instrument

    for i in np.sort(ins):
        if i in snaky_ins.keys():
            files = list((output_dir / s / i).glob("*.fits"))
            files = [str(f) for f in files]
            i2 = snaky_ins[i]
            products = list((output_dir_snaky / s / "data" / "s1d" / i2 / "WORKSPACE").glob("RASSINE_*.p"))
            products = [str(f) for f in products]
            if len(products)!=0:
                files = list(products)
    
            if i=='HARPN':
                source = ['IA2']*len(files)
            elif (i=='HERMES'):
                source = ['IAC']*len(files)
            elif (i=='SOPHIE')|(i=='NEID'):
                source = None
            else:
                source = ['ESO']*len(files)

            #TBD helper that will identify GR8 tag for the source

            if len(files)>0:
                job = snaky.start(verbose=verbose, debug=debug)
                try:
                    job.set_output_dir(str(output_dir_snaky))
                    job.set_dataset(s, i2, files, source=source)
                    job.set_star(ra=ra, dec=dec, teff=teff, rhk=rhk, ms=ms, rs=rs, logg=logg, feh=feh, vsini=vsini, prot=prot, age=age)
                    job.reduce(begin=begin, end=end, automatic_db=automatic_db)
                except SnakyError:
                    print(Fore.RED+' [ERROR] The process interrupted somewhere.'+Fore.RESET)
                plt.close('all')
            else:
                print(Fore.YELLOW+' [WARNING] No files found to be SNAKY processed'+Fore.RESET)
