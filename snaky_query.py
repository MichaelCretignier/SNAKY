import getopt
import sys
import os
from pathlib import Path

cwd = os.getcwd()
root = '/'.join(cwd.split('/')[:-1])

if root+'/Python/SNAKY' not in sys.path:
    sys.path.append(root+'/Python/SNAKY') #Until SNAKY is pip installable

import pyvo
from pyvo.dal.adhoc import DatalinkResults
import requests
import numpy as np

from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
import pandas as pd
import glob as glob
from typing import Optional
from colorama import Fore
import matplotlib.pylab as plt
import src_snaky.snaky_utilities as snaky_util
import src_snaky.run as snaky
from src_snaky.run import SnakyError

# python snaky_query.py -s HD217014,HD127334

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

if len(sys.argv)>1:
    optlist,args =  getopt.getopt(sys.argv[1:],'p:f:n:P:b:e:a:i:f:s:S:o:')
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
        if j[0] == '-f':
            force_query = bool(int(j[1]))
        if j[0] == '-o':
            output_root = j[1]
            if output_root[-1]!='/':
                output_root = output_root+'/'
        if j[0] == '-s':
            stars_to_process = j[1].split(',')
        if j[0] == '-S':
            star_to_process = j[1]

# CONFIG PART

if (len(stars_to_process)==1)&(stars_to_process[0][-3:]=='csv'):
    file_db = stars_to_process[0]
    db = pd.read_csv(stars_to_process[0],index_col=0) #optional keywords teff,logg,feh,logRHK,ms,rs,prot
    db = snaky_util.format(db) 
    if output_root is None:
        output_root = '/'.join(file_db.split('/')[:-1])+'/'
else:
    db = pd.DataFrame(np.array([stars_to_process]).T,columns=['starname'])
    db = snaky_util.format(db) 

db = snaky_util.fill_coordinates(db)

if output_root is None:
    output_root = root
    output_dir = root+'/Python/SNAKY/Snaky_data/SPECTRA_DB/'
    output_dir_snaky = root+'/Python/SNAKY/Snaky_data/'
else:
    output_dir = output_root+'SPECTRA_DB/'
    output_dir_snaky = output_root+'SNAKY_DB_SPECTRA/'

db2 = db.drop_duplicates(subset=['starname'])
if star_to_process is not None:
    db2 = db2.loc[db2['starname']==star_to_process]

print(db2)

db3 = np.array_split(db2,multiprocess_split)
db3 = db3[multiprocess_index-1] #multiprocessing splitting

# QUERY FUNCTIONS

instruments_excluded = ['','CRIRES','EFOSC','SOFI','PIONIER','MUSE','XSHOOTER','VIRCAM','GRAVITY','ALMA','SPHERE','NIRPS','OMEGACAM','GIRAFFE','HAWKI','KMOS','FORS2']

def query_eso(
        starname: str,
        output_dir: str = '/Users/cretignier/Desktop/test/',
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        search_by: str = 'coordinates', 
        fov: float = 2, 
        N_spectra: int = 5,
        selection: str = 'random',
        download: bool = True
        ) -> None:

    if (ra is None)|(dec is None):
        c = SkyCoord.from_name(starname)
        ra = c.ra.deg
        dec = c.dec.deg

    tap = pyvo.dal.TAPService("https://archive.eso.org/tap_obs")

    fov = np.round(fov*60,0)

    if search_by == 'name':
        query = f"""
        SELECT
            target_name,
            instrument_name,
            s_ra,
            s_dec,
            t_exptime,
            snr,
            t_min,
            access_url,
            dp_id
        FROM ivoa.ObsCore
        WHERE target_name='{starname}'
        """
    else:
        query = f"""
        SELECT
            target_name,
            instrument_name,
            s_ra,
            s_dec,
            t_exptime,
            snr,
            t_min,
            access_url,
            dp_id
        FROM ivoa.ObsCore
        WHERE CONTAINS(
            POINT('ICRS', s_ra, s_dec),
            CIRCLE('ICRS', {ra}, {dec}, {fov}/3600.)
        ) = 1
        """

    results = tap.search(query)
    results_df = pd.DataFrame(results)


    #print(results_df)

    os.makedirs(output_dir + f'{starname}/',exist_ok=True)

    if len(results_df)!=0:

        results_df.loc[results['target_name']==starname,'target_name'] = '*'
        results_df['diff'] = np.sqrt((ra-results_df['s_ra'])**2+(dec-results_df['s_dec'])**2)*3600
        
        results_df = results_df.sort_values(by=['instrument_name','target_name','diff']).reset_index(drop=True)
        
        instruments_found = results_df['instrument_name'].value_counts()
        
        print('\nFov query [arcsec]:', fov)
        print(starname,instruments_found)

        mask = np.in1d(results_df['instrument_name'],instruments_excluded)
        results_df = results_df[~mask]

    if len(results_df)!=0:
        instruments_found = results_df['instrument_name'].value_counts()
        print('\n',starname,instruments_found)
        if download:
            for ins in list(instruments_found.keys()):
                os.makedirs(output_dir + f'{starname}/{ins}/',exist_ok=True)
                subset = results_df.loc[results_df['instrument_name']==ins].reset_index(drop=True)
                if selection=='random':
                    subset = subset.sample(n=np.min([N_spectra, len(subset)]),random_state=42)
                else:
                    subset = subset[0:np.min([N_spectra, len(subset)])]

                print(subset[['access_url','instrument_name','target_name','diff']])

                for row in subset['access_url'].values:
                    dl = DatalinkResults.from_result_url(row)
                    science = next(dl.bysemantics("#this"))
                    r = requests.get(science.access_url, allow_redirects=True)
                    #r.raise_for_status()

                    filename = science['eso_origfile']
                    output_name = output_dir+f"{starname}/{ins}/{filename}"
                    if not os.path.exists(output_name):
                        with open(output_name, "wb") as f:
                            f.write(r.content)

def query_tng(
        starname: str,
        output_dir: str = '/Users/cretignier/Desktop/test/',
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        fov: float = 2,
        N_spectra: int = 5,
        selection: str = 'random', 
        download: bool = True
    ) -> None:

    tap = pyvo.dal.TAPService("http://archives.ia2.inaf.it/vo/tap/tng")

    if (ra is None)|(dec is None):
        c = SkyCoord.from_name(starname)
        ra = c.ra.deg
        dec = c.dec.deg

    ra = ra*np.pi/180
    dec = dec*np.pi/180

    fov_query = np.round(fov*60,0)
    ra_min = ra - (fov_query/3600.)*np.pi/180
    ra_max = ra + (fov_query/3600.)*np.pi/180
    dec_min = dec - (fov_query/3600.)*np.pi/180
    dec_max = dec + (fov_query/3600.)*np.pi/180

    query = f"""
    SELECT TOP 20 OFFSET 0
        tng.EXP_ID AS file_name,
        tng.policy AS policy,
        tng.DATE_OBS AS DATE_OBS,
        tng.OBS_MODE AS OBS_MODE,
        tng.INSTRUMENT AS INSTRUMENT,
        tng.PROGRAM AS PROGRAM,
        tng.OBJECT AS OBJECT,
        tng.file_url AS file_url,
        tng.PROGRAM AS program,
        tng.RA AS ra,
        tng.DEC AS dec
    FROM tng.TNG_TAP tng
    JOIN tng.harpn harpn
        ON tng.EXP_ID = harpn.EXP_ID
    WHERE
        tng.INSTRUMENT LIKE 'HARPN'
        AND tng.OBS_MODE LIKE 'SCIENCE'
        AND tng.RA_RAD BETWEEN {ra_min} AND {ra_max}
        AND tng.DEC_RAD BETWEEN {dec_min} AND {dec_max}
        AND tng.PROGRAM <> 'SOLAR'
        AND harpn.FILE_TYPE_REDUCED = 's1d_fluxcal'
        AND harpn.FILE_TYPE = 'reduced'
        """
    
    job = tap.submit_job(query)
    job.run()
    job.wait(timeout=60)
    results = job.fetch_result()

    results_df = pd.DataFrame(results)

    print('Fov query [arcsec]:', fov_query)
    print('HARPN     %.0f'%(len(results_df)))

    if len(results_df)!=0:
        results_df.loc[results['object']==starname,'object'] = '*'

        coords = SkyCoord(
            ra=results_df["ra"].values,
            dec=results_df["dec"].values,
            unit=(u.hourangle, u.deg)
        )

        results_df["ra"] = coords.ra.radian
        results_df["dec"] = coords.dec.radian
        results_df['diff'] = np.sqrt((ra-results_df['ra'])**2+(dec-results_df['dec'])**2)*180/np.pi*3600.
        
        results_df = results_df.sort_values(by=['instrument','object','diff']).reset_index(drop=True)
        
    ins = 'HARPN'
    os.makedirs(output_dir + f'{starname}/',exist_ok=True)
    os.makedirs(output_dir + f'{starname}/{ins}/',exist_ok=True)

    if len(results_df)!=0:        
        if download:
            if selection=='random':
                subset = results_df.sample(n=np.min([N_spectra, len(results_df)]),random_state=4)
            else:
                subset = results_df[0:np.min([N_spectra, len(results_df)])]

            print(subset[['file_url','instrument','object','diff']])

            for row in subset['file_url'].values:
                r = requests.get(row)
                #r.raise_for_status()

                filename = row.split('/')[-1]
                output_name = output_dir+f"{starname}/{ins}/{filename}"
                if not os.path.exists(output_name):
                    with open(output_name, "wb") as f:
                        f.write(r.content)

                os.system('gunzip ' + output_name)

# SPECTRA DOWNLOADING
#query_eso('51Peg', ra=None, dec=None, output_dir=output_dir, selection='os',search_by='coordinates', N_spectra = N_spectra, fov=field_of_view, download=True)
#query_tng('51Peg', ra=None, dec=None, output_dir=output_dir, N_spectra = N_spectra, fov=field_of_view, download=True)

print('\n [INFO] Querying ESO archive for stars in the database and downloading spectra...\n')
for i in db2.index:
    s,ra,dec = db2.loc[i,['starname','RA','DEC']]
    if (not os.path.exists(output_dir+s))|(force_query):
        print(f' [INFO] {s} downloading...')
        query_eso(
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

print('\n [INFO] Querying TNG archive for stars in the database and downloading spectra...\n')
for i in db2.index:
    s,ra,dec = db2.loc[i,['starname','RA','DEC']]
    if (not os.path.exists(output_dir+s+'/HARPN'))|(force_query):
        print(f' [INFO] {s} downloading...')
        query_tng(
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

# SNAKY PROCESSING

snaky_ins = {
    'FEROS':'FEROS_1.0',
    'HARPS':'HARPS_3.8',
    'UVES':'UVES_1.0',
    'FIES':'FIES_1.0',
    'ESPRESSO':'ESPRESSO19_3.3.6',
    'HARPN':'HARPN_3.0.1'}

print(' [INFO] SNAKY processing...')

idxs = np.ravel([db3.index.values])
for idx in idxs:
    s,ra,dec,teff,rhk,logg,feh,ms,rs,vsini,prot = db3.loc[idx,['starname','RA','DEC','teff','logRHK','logg','feh','ms','rs','vsini','prot']]

    if instrument is None:
        ins = snaky.glob.glob(output_dir+s+'/*')
        ins = np.unique([i.split('/')[-1] for i in ins])
    else:
        ins = instrument

    for i in np.sort(ins):
        if i in snaky_ins.keys():
            files = snaky.glob.glob(output_dir+s+'/'+i+'/*.fits')
            i2 = snaky_ins[i]
            products = snaky.glob.glob(output_dir_snaky+s+'/data/s1d/'+i2+'/WORKSPACE/RASSINE_*.p')

            if len(products)!=0:
                files = list(products)
    
            if i=='HARPN':
                source = 'IA2'
            else:
                source = 'ESO'

            if len(files)>0:
                job = snaky.start()
                try:
                    job.set_output_dir(output_dir_snaky)
                    job.set_dataset(s, i2, files, source=source)
                    job.set_star(ra=ra, dec=dec, teff=teff, rhk=rhk, ms=ms, rs=rs, logg=logg, feh=feh, vsini=vsini, prot=prot)
                    job.reduce(begin=begin, end=end, automatic_db=automatic_db)
                except SnakyError:
                    print(Fore.RED+' [ERROR] The process interrupted somewhere.'+Fore.RESET)
                plt.close('all')
            else:
                print(Fore.YELLOW+' [WARNING] No files found to be SNAKY processed'+Fore.RESET)
            


        


