import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord, Angle
import glob as glob
import matplotlib.pylab as plt

keywords_allowed = {
    'starname':['hd','hip','primary'],
    'RA':['ra'],
    'DEC':['dec'],
    'ms':['mass','msol','mstar'],
    'rs':['radius','rsol','rstar'],
    'prot':['lsper'],
    'teff':['teffstar'],
    'logg':['loggstar'],
    'feh':['fe/h'],
    'logRHK':['rhk','logrhk'],
    'vsini':['vsinistar'],
    'age':['age','age_gyr']
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

def mad(array,axis=0,sigma_conv=True):
    """"""
    if axis == 0:
        step = abs(array-np.nanmedian(array,axis=axis))
    else:
        step = abs(array-np.nanmedian(array,axis=axis)[:,np.newaxis])
    return np.nanmedian(step,axis=axis)*[1,1.48][int(sigma_conv)]

def boxplot_comparison(files, reference={}):

    count = -1

    extract = []

    variables = ['teff','logg','feh','ms','rs','vsini','mhk','rhk','age']
    save = {kw:[] for kw in variables}

    plt.figure(figsize=(18,12))
    plt.subplots_adjust(left=0.06,right=0.96,hspace=0.60,top=0.95,bottom=0.15,wspace=0.30)
    for f in files:
        ins = f.split('/WORKSPACE')[0].split('/')[-1]
        code = ins[0]+ins.split('_')[0][-2:]+'_'+ins.split('_')[1]
        count += 1
        table = pd.read_csv(f)
        extract.append([ins]+list(np.array(table.mean())))
        
        for j,kw in enumerate(variables):
            if kw in table.keys():
                plt.subplot(3,3,j+1)
                plt.boxplot(np.array(table[kw]),positions=[count],showfliers=False,labels=[code],widths=[0.5])
                plt.ylabel(kw,fontsize=14)
                plt.xticks(rotation=45,ha='right')
                save[kw].append(np.array(table[kw]))

    ylims = {}
    for j,kw in enumerate(variables):
        plt.subplot(3,3,j+1)
        med_val = np.median(np.ravel(save[kw]))
        sup_val = np.nanpercentile(np.ravel(save[kw]),75) - med_val
        inf_val = med_val - np.nanpercentile(np.ravel(save[kw]),25)
        ylim = [med_val-5*inf_val,med_val+5*sup_val]
        plt.boxplot(np.ravel(save[kw]),positions=[count+2],showfliers=False,labels=['ALL'],widths=[0.5],patch_artist=True,boxprops=dict(facecolor='lightsteelblue',edgecolor='black',linewidth=1.))
        plt.ylim(ylim)

        if kw=='teff':
            plt.title('%s = %.0f +/- %.0f'%(kw,np.median(save[kw]),mad(np.ravel(save[kw]))))
        else:
            plt.title('%s = %.2f +/- %.2f'%(kw,np.median(save[kw]),mad(np.ravel(save[kw]))))
        plt.xticks(rotation=90,ha='center')

        if kw in reference.keys():
            plt.axhline(y=reference[kw], ls='-.', color='k', alpha=0.7, lw=1)

        ylims[kw] = ylim

    return table, extract, ylims


def pdf_comparison(files, xlims={}, reference={}):

    count = -1

    variables = ['teff','logg','feh','ms','rs','vsini','mhk','rhk','age']
    save = {kw:[] for kw in variables}

    plt.figure(figsize=(18,12))
    plt.subplots_adjust(left=0.06,right=0.96,hspace=0.60,top=0.95,bottom=0.15,wspace=0.30)

    instrument = [ins.split('_')[0][0:5] for ins in [f.split('/WORKSPACE')[0].split('/')[-1] for f in files]]

    count = -1
    for i in np.unique(instrument):
        count+=1
        loc = np.where(np.array(instrument)==i)[0]
        files_i = [files[j] for j in loc]

        df = []
        for f in files_i:
            table = pd.read_csv(f)
            df.append(table)

        df = pd.concat(df,axis=0,ignore_index=True)

        for j,kw in enumerate(variables):
            ins = f.split('/WORKSPACE')[0].split('/')[-1]
            if kw in xlims.keys():
                bins = np.linspace(xlims[kw][0], xlims[kw][1], 50)
            else:
                bins = 50

            if kw in df.keys():
                vec = np.array(df[kw])
                vec = vec[vec==vec]
                if len(vec)>0:
                    plt.subplot(3,3,j+1)
                    a,b = np.histogram(vec, bins=bins, density=True)
                    plt.plot(b[:-1], a, color='C%.0f'%(count), alpha=0.5)
                    plt.xlabel(kw,fontsize=14)
                    plt.yticks(ha='right')
                    save[kw].append(vec)
    
    for j,kw in enumerate(variables):
        save[kw] = np.hstack(save[kw])
        plt.subplot(3,3,j+1)
        med_val = np.median(np.ravel(save[kw]))
        sup_val = np.nanpercentile(np.ravel(save[kw]),75) - med_val
        inf_val = med_val - np.nanpercentile(np.ravel(save[kw]),25)
        xlim = [med_val-5*inf_val,med_val+5*sup_val]

        a,b = np.histogram(np.ravel(save[kw]), bins=np.linspace(xlim[0], xlim[1], 50), density=True)
        plt.plot(b[:-1], a, alpha=1.0, label='ALL', color='k',lw=2.5)
        plt.ylabel('PDF',fontsize=14)
        plt.xlim(xlim)
        plt.ylim(0,None)

        if kw=='teff':
            plt.title('%s = %.0f +/- %.0f'%(kw,np.median(save[kw]),mad(np.ravel(save[kw]))))
        else:
            plt.title('%s = %.2f +/- %.2f'%(kw,np.median(save[kw]),mad(np.ravel(save[kw]))))

        if kw in reference.keys():
            plt.axvline(x=reference[kw], ls='-.', color='k', alpha=0.7, lw=1)
