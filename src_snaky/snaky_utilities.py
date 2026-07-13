import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord, Angle

keywords_allowed = {
    'starname':['primary','hd','hip'],
    'RA':['ra'],
    'DEC':['dec'],
    'ms':['mass','msol','mstar'],
    'rs':['radius','rsol','rstar'],
    'prot':['lsper'],
    'teff':['teffstar'],
    'logg':['loggstar'],
    'feh':['fe/h'],
    'logRHK':['rhk','logrhk'],
    'vsini':['vsinistar']
    }

def format(df):

    df.columns = [c.lower().replace('_','') for c in df.columns]

    for kw in keywords_allowed.keys():
        if kw not in df.keys():
            for kw_alt in keywords_allowed[kw]:
                try:
                    df[kw] = df[kw_alt].astype(object)
                    df.loc[df[kw]=='-',kw] = None
                    df.loc[pd.isna(df[kw]),kw] = None
                    break
                except:
                    pass

    for kw in keywords_allowed.keys():
        if kw not in df.keys():
            df[kw] = None

    df = df[list(keywords_allowed.keys())]

    return df

def fill_coordinates(db):
    for i in db.index.values:
        entry = db.loc[i]
        if (entry['RA'] is None)|(entry['DEC'] is None)|(entry['RA']!=entry['RA'])|(entry['DEC']!=entry['DEC']):
            c = SkyCoord.from_name(entry.starname)
            ra = c.ra.deg
            dec = c.dec.deg
            db.loc[i,'RA'] = ra
            db.loc[i,'DEC'] = dec
    return db

