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
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import pyvo
from pyvo.dal.adhoc import DatalinkResults
import gzip, shutil
import time
from pathlib import Path

from . import snaky_functions as myf

instruments_excluded = ['','CRIRES','EFOSC','SOFI','PIONIER','MUSE','XSHOOTER','VIRCAM','GRAVITY','ALMA','SPHERE','NIRPS','OMEGACAM','GIRAFFE','HAWKI','KMOS','FORS2']

def build_session():
    s = requests.Session()
    retry = Retry(
        total=8,
        connect=8,
        read=8,
        status=8,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s

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
        output_dir: Path = Path('/Users/cretignier/Desktop/test/').expanduser().resolve(),
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
        sophie_dir = output_dir / starname / "SOPHIE"
        sophie_dir.mkdir(parents=True, exist_ok=True)
        for cmd in commands:

            m = re.search(r'wget "(.*?)" -O (.*)', cmd)

            url = m.group(1)
            filename = m.group(2)

            r = requests.get(url)
            r.raise_for_status()

            output_name = sophie_dir / filename
            with open(output_name, "wb") as f:
                f.write(r.content)

def query_iac(
        starname: str,
        output_dir: Path = Path('/Users/cretignier/Desktop/test/').expanduser().resolve(),
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
            for ins in instruments_found:

                ins_dir = output_dir / starname / ins
                ins_dir.mkdir(parents=True, exist_ok=True)

                subset = results_df.loc[
                    results_df["instrument_name"] == ins
                ].reset_index(drop=True)

                subset = subset.sample(
                    n=min(N_spectra, len(subset)),
                    random_state=42
                )

                print(subset[["access_url", "instrument_name", "object"]])

                for row, filename in subset[["access_url", "filename"]].values:

                    r = requests.get(row, allow_redirects=True)
                    r.raise_for_status()

                    output_name = ins_dir / filename

                    if not output_name.exists():
                        output_name.write_bytes(r.content)

def query_eso(
        starname: str,
        output_dir: Path = Path('/Users/cretignier/Desktop/test/').expanduser().resolve(),
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
        SELECT target_name, instrument_name, s_ra, s_dec, t_exptime, snr, t_min, access_url, dp_id
        FROM ivoa.ObsCore
        WHERE target_name='{starname}'
        """
    else:
        query = f"""
        SELECT target_name, instrument_name, s_ra, s_dec, t_exptime, snr, t_min, access_url, dp_id
        FROM ivoa.ObsCore
        WHERE CONTAINS(POINT('ICRS', s_ra, s_dec),CIRCLE('ICRS', {ra}, {dec}, {fov}/3600.)) = 1
        """

    results = tap.search(query)
    results_df = pd.DataFrame(results)

    (output_dir / f'{starname}').mkdir(parents=True, exist_ok=True)

    if len(results_df)!=0:

        results_df['dyear'] = (myf.today() - results_df['t_min'])//365
        
        results_df['dra']= (ra-results_df['s_ra'])*3600
        results_df['ddec']= (dec-results_df['s_dec'])*3600
        
        results_df.loc[results_df['target_name']==starname,'target_name'] = '*'
        results_df['diff'] = np.sqrt((results_df['dra'])**2+(results_df['ddec'])**2)

        if np.sum(results_df['target_name']=='*')==0:
            results_df = results_df.sort_values(by=['instrument_name','diff']).reset_index(drop=True)
        else:
            results_df = results_df.sort_values(by=['instrument_name','target_name','diff']).reset_index(drop=True)
        instruments_found = results_df['instrument_name'].value_counts()

        print('\nFov query [arcsec]:', fov)
        print(starname,instruments_found)

        mask = np.in1d(results_df['instrument_name'],instruments_excluded)
        results_df = results_df[~mask]


    #print(results_df[['instrument_name','diff','dra','ddec','dyear','target_name']])
    #import matplotlib.pyplot as plt
    #plt.scatter(results_df['dra'],results_df['ddec'],c=results_df['dyear'],s=50,cmap='plasma')
    #plt.axvline(x=0,color='k',ls='--')
    #plt.axhline(y=0,color='k',ls='--')
    #plt.plot(np.sin(np.linspace(0,2*np.pi,100))*fov,np.cos(np.linspace(0,2*np.pi,100))*fov,color='k',ls='--')
    #plt.colorbar()
    #plt.show()

    if len(results_df)!=0:
        instruments_found = results_df['instrument_name'].value_counts()
        print('\n',starname,instruments_found)
        if download:
            for ins in list(instruments_found.keys()):
                (output_dir / f'{starname}/{ins}').mkdir(parents=True, exist_ok=True)
                subset = results_df.loc[results_df['instrument_name']==ins].reset_index(drop=True)
                if np.sum(results_df['target_name']=='*')!=0: #if the target is found with the same user specified name
                    subset = subset.loc[subset['target_name']=='*'].reset_index(drop=True)

                if selection=='random':
                    subset = subset.sample(n=np.min([N_spectra, len(subset)]),random_state=42)
                else:
                    subset = subset[0:np.min([N_spectra, len(subset)])]

                print(subset[['access_url','instrument_name','target_name','diff']])

                session = build_session()

                for row in subset['access_url'].values:
                    time.sleep(1.0)
                    try:
                        url = str(row)
                        if url.startswith("http://"):
                            url = "https://" + url[len("http://"):]

                        dl = DatalinkResults.from_result_url(url, session=session)
                        science = next(dl.bysemantics("#this"), None)
                        if science is None or not science.access_url:
                            print(f"[skip] no science product for {url}")
                            continue

                        sci_url = str(science.access_url)
                        if sci_url.startswith("http://"):
                            sci_url = "https://" + sci_url[len("http://"):]

                        r = session.get(sci_url, allow_redirects=True, timeout=(10, 120))
                        r.raise_for_status()

                        filename = science.get('eso_origfile', os.path.basename(sci_url))
                        output_name = output_dir / f"{starname}/{ins}/{filename}"

                        if not output_name.exists():
                            with open(output_name, "wb") as f:
                                f.write(r.content)

                    except Exception as e:
                        print(f"[skip] failed for {row}: {e}")
                        continue

def query_tng(
        starname: str,
        output_dir: Path = Path('/Users/cretignier/Desktop/test/').expanduser().resolve(),
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

    ra_rad = ra*np.pi/180
    dec_rad = dec*np.pi/180

    fov_query = np.round(fov*60,0)
    ra_min = ra_rad - (fov_query/3600.)*np.pi/180
    ra_max = ra_rad + (fov_query/3600.)*np.pi/180
    dec_min = dec_rad - (fov_query/3600.)*np.pi/180
    dec_max = dec_rad + (fov_query/3600.)*np.pi/180

    query = f"""
    SELECT TOP 30 OFFSET 0
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
        AND tng.policy LIKE 'FREE'
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
        results_df['diff'] = np.sqrt((ra_rad-results_df['ra'])**2+(dec_rad-results_df['dec'])**2)*180/np.pi*3600.
        
        results_df = results_df.sort_values(by=['instrument','object','diff']).reset_index(drop=True)
        
    ins = 'HARPN'
    (output_dir / starname / ins).mkdir(parents=True, exist_ok=True)

    if len(results_df)!=0:        
        if download:
            if selection=='random':
                subset = results_df.sample(n=np.min([N_spectra, len(results_df)]),random_state=4)
            else:
                subset = results_df[0:np.min([N_spectra, len(results_df)])]

            print(subset[['file_url','instrument','object','diff']])

            session = build_session()

            for row in subset['file_url'].values:
                time.sleep(1.0)
                try:
                    url = str(row)
                    r = session.get(url, timeout=(10, 30), allow_redirects=True)
                    #r.raise_for_status()

                    filename = url.split('/')[-1]
                    output_name = output_dir / starname / ins / filename

                    if not output_name.exists():
                        output_name.write_bytes(r.content)

                    if output_name.suffix == ".gz":
                        unzipped = output_name.with_suffix("")

                        if not unzipped.exists():
                            with gzip.open(output_name, "rb") as fin, unzipped.open("wb") as fout:
                                shutil.copyfileobj(fin, fout)

                            # Delete the compressed file
                            output_name.unlink()
                
                except Exception as e:
                    print(f"[skip] failed download for {row}: {e}")
                    continue

