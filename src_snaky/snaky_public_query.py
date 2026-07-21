import xml.etree.ElementTree as ET
from typing import Optional
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u

import pandas as pd
import numpy as np 
import requests
import subprocess
import re
import os
from urllib.parse import quote

import pyvo
from pyvo.dal.adhoc import DatalinkResults


instruments_excluded = ['','CRIRES','EFOSC','SOFI','PIONIER','MUSE','XSHOOTER','VIRCAM','GRAVITY','ALMA','SPHERE','NIRPS','OMEGACAM','GIRAFFE','HAWKI','KMOS','FORS2']

def sophie_rectangle(ra_deg, dec_deg, fov_deg):

    dra = (fov_deg/2)/np.cos(np.deg2rad(dec_deg))
    ddec = fov_deg/2

    c1 = SkyCoord((ra_deg-dra)*u.deg, (dec_deg-ddec)*u.deg)
    c2 = SkyCoord((ra_deg+dra)*u.deg, (dec_deg+ddec)*u.deg)

    ra1 = c1.ra.to_string(unit=u.hour, sep=" ", precision=0, pad=True)
    ra2 = c2.ra.to_string(unit=u.hour, sep=" ", precision=0, pad=True)

    dec1 = c1.dec.to_string(
        unit=u.deg,
        sep=" ",
        precision=0,
        alwayssign=True,
        pad=True,
    )

    dec2 = c2.dec.to_string(
        unit=u.deg,
        sep=" ",
        precision=0,
        alwayssign=True,
        pad=True,
    )

    rectangle = f"[J{ra1}{dec1}],[J{ra2}{dec2}]"

    return quote(rectangle, safe="[],")

def query_sophie(
        starname: str,
        output_dir: str = '/Users/cretignier/Desktop/test/',
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        search_by: str = 'coordinates', 
        fov: float = 2, 
        N_spectra: int = 5,
        selection: str = 'random',
        download: bool = True
):

    if (ra is None)|(dec is None):
        c = SkyCoord.from_name(starname)
        ra = c.ra.deg
        dec = c.dec.deg

    if search_by!='coordinates':
        url_link = f"http://atlas.obs-hp.fr/sophie/sophie.cgi?n=sophies&c=o&o={starname}&of=1,leda,simbad&sql=view_head%20IS%20NOT%20NULL&a=t&z=d|wg|e&ob=ra,seq&d=[%27wget%20%22http%3A//atlas.obs-hp.fr/sophie/sophie.cgi?c=i%26a=mime%3Aapplication/fits%26o=sophie%3A[s1d,%27||seq||%27]%22%20-O%20%27||seq||%27_s1d.fits%27]&nra=l,simbad,d"
    else:
        r2 = sophie_rectangle(ra, dec, fov/60)
        url_link = f"http://atlas.obs-hp.fr/sophie/sophie.cgi?n=sophies&c=o&of=1,leda,simbad&r={r2}&sql=view_head%20IS%20NOT%20NULL&a=t&z=d|wg|e&ob=ra,seq&d=[%27wget%20%22http%3A//atlas%2Eobs-hp%2Efr/sophie/sophie.cgi?c=i%26a=mime%3Aapplication/fits%26o=sophie%3A[s1d,%27||seq||%27]%22%20-O%20%27||seq||%27_s1d%2Efits%27]&nra=l,simbad,d"
    
    r = requests.get(url_link, allow_redirects=True)

    text = r.text

    commands = [
        line.strip()
        for line in text.splitlines()
        if line.startswith("wget ")
    ]

    if selection=='random':
        np.random.seed(42)
        commands = np.random.choice(commands,np.min([N_spectra,len(commands)]),replace=False)
    else:
        commands = commands[0:np.min([N_spectra,len(commands)])]
    

    if download:
        os.makedirs(output_dir + f'{starname}/SOPHIE/',exist_ok=True)
        for cmd in commands:

            m = re.search(r'wget "(.*?)" -O (.*)', cmd)

            url = m.group(1)
            filename = m.group(2)

            r = requests.get(url)
            r.raise_for_status()

            output_name = output_dir+f"{starname}/SOPHIE/{filename}"

            with open(output_name, "wb") as f:
                f.write(r.content)

def query_iac(
        starname: str,
        output_dir: str = '/Users/cretignier/Desktop/test/',
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        fov: float = 2, 
        N_spectra: int = 5,
        selection: str = 'random',
        download: bool = True
        ):    

    if (ra is None)|(dec is None):
        c = SkyCoord.from_name(starname)
        ra = c.ra.deg
        dec = c.dec.deg

    size = np.round(fov/60,3)
    url = (
        f"http://ocan.iac.es:8080/iacob/jsp/ssap.jsp"
        f"?REQUEST=queryData"
        f"&POS={ra},{dec}"
        f"&SIZE={size}"
    )

    try:
        xml = requests.get(url).text

        root = ET.fromstring(xml)

        output = []

        for tr in root.findall(".//TR"):

            td = tr.findall("TD")

            download_url = td[1].text.strip().strip('"')
            filename     = td[6].text
            target       = td[7].text
            sptype       = td[8].text
            snr          = float(td[10].text)
            instrument   = td[12].text
            resolution   = float(td[13].text)

            output.append([starname,target,download_url,instrument,snr,filename])
            
        results_df = pd.DataFrame(output,columns=['starname','object','access_url','instrument_name','snr','filename'])
    except:
        print(' [INFO] Download crashed...')
        results_df = []

    if len(results_df)!=0:
        results_df.loc[results_df['instrument_name']=='MERCATOR','instrument_name'] = 'HERMES'
        results_df.loc[results_df['instrument_name']=='NOT','instrument_name'] = 'FIES'
        instruments_found = results_df['instrument_name'].value_counts()
        print('\n',starname,instruments_found)
        if download:
            for ins in list(instruments_found.keys()):
                os.makedirs(output_dir + f'{starname}/{ins}/',exist_ok=True)
                subset = results_df.loc[results_df['instrument_name']==ins].reset_index(drop=True)
                subset = subset.sample(n=np.min([N_spectra, len(subset)]),random_state=42)

                print(subset[['access_url','instrument_name','object']])

                for row,filename in subset[['access_url','filename']].values:
                    r = requests.get(row, allow_redirects=True)
                    output_name = output_dir+f"{starname}/{ins}/{filename}"
                    if not os.path.exists(output_name):
                        with open(output_name, "wb") as f:
                            f.write(r.content)

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

