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

multiprocess_split = 3
multiprocess_index = 0
if len(sys.argv)>1:
    optlist,args =  getopt.getopt(sys.argv[1:],'p:f:N:')
    for j in optlist:
        if j[0] == '-p':
            multiprocess_index = j[1]
        if j[0] == '-f':
            field_of_view = j[1]
        if j[0] == '-N':
            N_spectra = int(j[1])

if multiprocess_index>(multiprocess_split-1):
    multiprocess_index = multiprocess_split-1

# configpart

output_dir = '/Users/cretignier/Documents/Atmos_Vsini/SPECTRA_DB/' #downloaded spectra output dir
output_dir_snaky = '/Users/cretignier/Documents/Atmos_Vsini/SNAKY_DB_SPECTRA/' #snaky output working space
field_of_view = 2 #arcmin
N_spectra = 5 #number of spectra per instrument to download and process

db = pd.read_csv('/Users/cretignier/Documents/Atmos_Vsini/db_merge.csv',index_col=0)
db2 = db.drop_duplicates(subset=['starname'])[['starname','RA','DEC']]

# list of [[starname1,ra1,dec1],[starname2,ra2,dec2]]
#db2 = [['HD26965',63.818000,-7.652870],['HD4628',None,None],['HD197481',None,None]]

db3 = np.array_split(db2,multiprocess_split)
db3 = db3[multiprocess_index] #multiprocessing splitting

# functions

instruments_excluded = ['CRIRES','PIONIER','MUSE','XSHOOTER','VIRCAM','GRAVITY','ALMA','SPHERE','NIRPS','OMEGACAM','GIRAFFE','HAWKI','KMOS','FORS2']

def query_eso(
        starname: str,
        output_dir: str = '/Users/cretignier/Desktop/test/',
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        search_by: str = 'coordinates', 
        fov: float = 2, 
        N_spectra: int = 5,
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
        instruments_found = results_df['instrument_name'].value_counts()
        
        print(starname,instruments_found)

        if download:
            for ins in list(instruments_found.keys()):
                if ins not in instruments_excluded:
                    os.makedirs(output_dir + f'{starname}/{ins}/',exist_ok=True)
                    subset = results_df.loc[results_df['instrument_name']==ins]
                    subset = subset.sample(n=np.min([N_spectra, len(subset)]),random_state=42)
                    for row in subset['access_url'].values:
                        dl = DatalinkResults.from_result_url(row)
                        science = next(dl.bysemantics("#this"))
                        r = requests.get(science.access_url, allow_redirects=True)
                        #r.raise_for_status()

                        filename = science['eso_origfile']

                        with open(output_dir+f"{starname}/{ins}/{filename}", "wb") as f:
                            f.write(r.content)


def query_tng(
        starname: str,
        output_dir: str = '/Users/cretignier/Desktop/test/',
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        fov: float = 2,
        N_spectra: int = 5,
        download: bool = True
    ) -> None:

    tap = pyvo.dal.TAPService("http://archives.ia2.inaf.it/vo/tap/tng")

    if (ra is None)|(dec is None):
        c = SkyCoord.from_name(starname)
        ra = c.ra.deg
        dec = c.dec.deg

    ra = ra*np.pi/180
    dec = dec*np.pi/180

    fov = np.round(fov*60,0)
    ra_min = ra - (fov/3600.)*np.pi/180
    ra_max = ra + (fov/3600.)*np.pi/180
    dec_min = dec - (fov/3600.)*np.pi/180
    dec_max = dec + (fov/3600.)*np.pi/180

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
    os.makedirs(output_dir + f'{starname}/',exist_ok=True)

    ins = 'HARPN'

    os.makedirs(output_dir + f'{starname}/{ins}/',exist_ok=True)

    if len(results_df)!=0:        
        if download:
            subset = results_df.sample(n=np.min([N_spectra, len(results_df)]),random_state=4)
            for row in subset['file_url'].values:
                r = requests.get(row)
                #r.raise_for_status()

                filename = row.split('/')[-1]

                with open(output_dir+f"{starname}/{ins}/{filename}", "wb") as f:
                    f.write(r.content)

                os.system('gunzip ' + output_dir+f"{starname}/{ins}/{filename}")

# SPECTRA DOWNLOADING

# output_dir = '...'
# query_eso('51Peg', ra=None, dec=None, output_dir=output_dir, search_by='coordinates', fov=2, download=True)
# query_tng('51Peg', ra=None, dec=None, output_dir=output_dir, fov=2, download=True)

print('\n [INFO] Querying ESO archive for stars in the database and downloading spectra...\n')
for s,ra,dec in np.array(db2):
    if not os.path.exists(output_dir+s):
        print(f' [INFO] {s} downloading...')
        query_eso(
            s, 
            ra = ra, 
            dec = dec, 
            output_dir = output_dir,
            search_by = 'coordinates',
            fov = field_of_view,
            N_spectra = N_spectra,
            download = True)
    else:
        print(f' [INFO] {s} spectra already downloaded, skipping...')

print('\n [INFO] Querying TNG archive for stars in the database and downloading spectra...\n')
for s,ra,dec in np.array(db2):
    if not os.path.exists(output_dir+s+'/HARPN'):
        print(f' [INFO] {s} downloading...')
        query_tng(
            s, 
            ra = ra, 
            dec = dec, 
            output_dir = output_dir,
            fov = field_of_view,
            N_spectra = N_spectra,
            download = True)
    else:
        print(f' [INFO] {s} spectra already downloaded, skipping...')

# SNAKY PROCESSING

import src_snaky.run as snaky
from src_snaky.run import SnakyError

snaky_ins = {
    'FEROS':'FEROS_1.0',
    'HARPS':'HARPS_3.8',
    'UVES':'UVES_1.0',
    'FIES':'FIES_1.0',
    'ESPRESSO':'ESPRESSO19_3.3.6',
    'HARPN':'HARPN_3.0.1'}

print(' [INFO] SNAKY processing...')
for s,ra,dec in np.array(db3):
    ins = snaky.glob.glob(output_dir+s+'/*')
    ins = [i.split('/')[-1] for i in ins]
    for i in ins:
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
                job.set_star(ra=ra, dec=dec)
                job.reduce(begin=1, end=14, automatic_db=True)
            except SnakyError:
                pass
        else:
            print(' [INFO] No files found to be SNAKY processed')
            


        


