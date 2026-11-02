import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord, Angle
import glob as glob
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
import re
import os

from . import snaky_functions as myf
from . import snaky_variables as myv

keywords_allowed = {
    'starname':['primary','hd','namehd','hip','namehip'],
    'RA':['ra','ra_j2000','ra2000'],
    'DEC':['dec','dec_j2000','dec2000'],
    'ms':['ms','mass','msol','mstar','m'],
    'rs':['rs','radius','rsol','rstar','r'],
    'prot':['prot','lsper'],
    'teff':['teff','teffstar'],
    'logg':['logg','loggstar'],
    'feh':['feh','fe/h'],
    'logRHK':['rhk','logrhk'],
    'vsini':['vsini','vsinistar'],
    'age':['age','age_gyr'],
    'paper':['paper','source','reference'],
    }

keywords_std = ['ms','rs','prot','teff','logg','feh','logRHK','vsini','age']

def filter_job_id(dir_output,job_id):
    files = glob.glob(dir_output+'/data/s1d/*/REDUCTION_INFO/Time_info_reduction_ID%s_snaky_*.csv'%str(job_id).zfill(6))

def ra_to_deg(ra,ra_ref=None):
    if ra==ra:
        try:
            ra = float(ra)
            h  = int(ra // 10000)
            m  = int((ra % 10000) // 100)
            s  = ra % 100
            return (h + m/60 + s/3600) * 15
        except:
            if ra_ref is not None:
                return ra_ref
    else:
        return np.nan

def dec_to_deg(dec,dec_ref=None):
    if dec==dec:
        try:
            dec = float(dec)
            sign = -1 if dec < 0 else 1
            dec = abs(dec)
            d  = int(dec // 10000)
            m  = int((dec % 10000) // 100)
            s  = dec % 100
            return sign * (d + m/60 + s/3600)
        except:
            if dec_ref is not None:
                return dec_ref
    else:
        return np.nan

def format_coordinates_instrument(table,ins,ra=None,dec=None):

    for ins in np.unique(table['ins_code']):
        summary = table[table['ins_code']==ins].copy()
        instrument = ins
        files = summary['files'].values
        if ins=='EXPRE':
            summary['RA'] = np.round(np.array([ra_to_deg(ra2.replace(':','')) for ra2 in np.array(summary['RA'])]),6)
            summary['DEC'] = np.round(np.array([dec_to_deg(dec2.replace(':','')) for dec2 in np.array(summary['DEC'])]),6)
            summary['berv'] = np.nan
            summary['snr'] = np.nan
            summary['rjd'] = summary['rjd'].astype('float') + 2400000
        if ins=='GR8':
            summary['rjd'] = np.nan
            summary['berv'] = np.nan
        if ins=='FEROS':
            summary['rjd'] = summary['rjd'].astype('float') + 2400000
        if ins=='UVES':
            summary['rjd'] = summary['rjd'].astype('float') + 2400000
        if ins=='ESO':
            summary['rjd'] = summary['rjd'].astype('float') + 2400000
            summary['berv'] = np.nan
        if ins=='PEPSI':
            for i in summary.index:
                RA = summary.loc[i,'RA'].replace(':','')
                summary.loc[i,'RA'] = RA
                DEC = summary.loc[i,'DEC'].replace(':','')
                summary.loc[i,'DEC'] = DEC
            summary['RA'] = np.round(np.array([ra_to_deg(ra2) for ra2 in np.array(summary['RA'])]),6)
            summary['DEC'] = np.round(np.array([dec_to_deg(dec2) for dec2 in np.array(summary['DEC'])]),6)
        if (ins=='SOPHI'):
            if (ra is None)|(dec is None):
                for i in summary.index:
                    RA = summary.loc[i,'RA']
                    length = len(str(RA).split('.')[0])
                    RA = '0'*(6-length)+str(RA)
                    summary.loc[i,'RA'] = RA
                summary['RA'] = np.round(np.array([ra_to_deg(ra2) for ra2 in np.array(summary['RA'])]),6)
                summary['DEC'] = np.round(np.array([dec_to_deg(dec2) for dec2 in np.array(summary['DEC'])]),6)
            else:
                summary['RA'] = ra
                summary['DEC'] = dec
        if (ins=='NEID'):
            for i in summary.index:
                RA = summary.loc[i,'RA'].replace(':','')
                summary.loc[i,'RA'] = RA
                DEC = summary.loc[i,'DEC'].replace(':','')
                summary.loc[i,'DEC'] = DEC
            summary['RA'] = np.round(np.array([ra_to_deg(ra2) for ra2 in np.array(summary['RA'])]),6)
            summary['DEC'] = np.round(np.array([dec_to_deg(dec2) for dec2 in np.array(summary['DEC'])]),6)
        if (ins=='HARPN'):
            summary['rjd'] = summary['rjd'].astype('float') + 2400000
            for i in summary.index:
                RA = summary.loc[i,'RA'].replace('h','').replace('m','')
                summary.loc[i,'RA'] = RA
                DEC = summary.loc[i,'DEC'].replace(':','')
                summary.loc[i,'DEC'] = DEC
            summary['RA'] = np.round(np.array([ra_to_deg(ra2) for ra2 in np.array(summary['RA'])]),6)
            summary['DEC'] = np.round(np.array([dec_to_deg(dec2) for dec2 in np.array(summary['DEC'])]),6)
        if (ins=='ESPRE')|(ins=='HARPS')|(ins=='harps'): 
            summary['RA'] = np.round(np.array([ra_to_deg(ra2) for ra2 in np.array(summary['RA'])]),6)
            summary['DEC'] = np.round(np.array([dec_to_deg(dec2) for dec2 in np.array(summary['DEC'])]),6)
        if ins=='CORAL':
            summary['RA'] = np.round(summary['RA'].astype('float'),6)
            summary['DEC'] = np.round(summary['DEC'].astype('float'),6)
        if ins=='RASSINE':
            missing_time = np.array(summary['rjd']!=summary['rjd'])
            if np.sum(missing_time)!=0:
                myv.vprint(' [INFO] Nb of missing time = %.0f'%(np.sum(missing_time)))

            if np.sum(missing_time)!=0: #search ut in filename
                if instrument[0:4]=='NEID':
                    uttime = np.array([f[-17:-2] for f in files])
                    uttime = np.array([f"{s[:4]}-{s[4:6]}-{s[6:8]}T{s[9:11]}:{s[11:13]}:{s[13:15]}.000" for s in uttime])
                else:
                    uttime = np.array(myf.detect_uttime_str(files))
                year = np.array([ut[0:4] for ut in uttime])
                year_num = np.array([float(y) if y.isdigit() else np.nan for y in year])
                valid = (year_num>1950)&(year_num<2050)
                if np.sum(valid&missing_time)>0:
                    rjd = myf.conv_time(uttime[valid&missing_time])[0]
                    summary.loc[valid&missing_time,'rjd'] = rjd
                missing_time = (summary['rjd']!=summary['rjd'])
                myv.vprint(' [INFO] Nb of missing time after UT search = %.0f'%(np.sum(missing_time)))
            
            summary['rjd'] = summary['rjd'].astype('float') + 2400000
            summary['RA'] = ra
            summary['DEC'] = dec  
        if ins =='YARARA':
            summary['RA'] = ra
            summary['DEC'] = dec

        table.loc[summary.index, summary.columns] = summary
    return table

def format_source(source,files):
    if len(files)!=0:
        filenames = [f.split('/')[-1] for f in files]
        if source is None:
            source = np.array(['DRS']*len(files))
        else:
            source = np.array(source)
            source[myf.string_contained_in(filenames, 'GR8')[0]] = 'GR8'
            source[myf.string_contained_in(filenames, 'gr8')[0]] = 'GR8'
        return list(source)
    else:
        return None

def format_source_uncertainties(table):



    static_table = pd.read_csv(myv.MATERIAL_DIR+'/static_table.csv')

def format_user_object(parameter):
    if parameter is None:
        parameter = [None,None]
    elif type(parameter) is not list:
        parameter = [parameter,None]

    return parameter

def format(df):

    for c in df.columns:

        if c.split('_')[-1]=='uerr':
            df[c.replace('_uerr','_std')] = (df[c]+df[c.replace('_uerr','_lerr')])/2

        if c.split('_')[-1]=='err':
            df[c.replace('_err','_std')] = df[c].astype(object)

    df.columns = [c.lower().replace('_','') for c in df.columns]
    liste = []

    final_liste = list(keywords_allowed.keys()) + [k+'_std' for k in keywords_std]

    for kw in keywords_allowed.keys():
        if kw not in df.keys():
            for kw_alt in keywords_allowed[kw]:
                try:
                    df[kw] = df[kw_alt].astype(object)
                    df.loc[df[kw]=='-',kw] = None
                    df.loc[pd.isna(df[kw]),kw] = None
                    break #in case multi keyword stop in the first in the list
                except:
                    pass

    for kw in keywords_std:
        if kw+'_std' not in df.keys():
            for kw_alt in keywords_allowed[kw]:
                try:
                    df[kw+'_std'] = df[kw_alt+'std'].astype(object)
                    df.loc[df[kw+'_std']=='-',kw+'_std'] = None
                    df.loc[pd.isna(df[kw+'_std']),kw+'_std'] = None
                    break
                except:
                    pass

    for kw in final_liste:
        if kw not in df.keys():
            df[kw] = None

    df = df[list(final_liste)]
    df = df.dropna(subset=['starname'])

    return df

def fill_coordinates(db):
    for i in db.index.values:
        entry = db.loc[i]
        if (entry['RA'] is None)|(entry['DEC'] is None)|(entry['RA']!=entry['RA'])|(entry['DEC']!=entry['DEC']):
            try:
                c = SkyCoord.from_name(entry.starname)
                ra = c.ra.deg
                dec = c.dec.deg
                db.loc[i,'RA'] = ra
                db.loc[i,'DEC'] = dec
            except:
                pass
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
    all_variables = ['teff','logg','feh','ms','rs','vsini','mhk','rhk','age','sini','prot']
    save = {kw:[] for kw in variables}

    kept = []
    for f in files:
        table = pd.read_csv(f)
        stds = table.std()
        if stds['logg']<0.25:
            kept.append(1)
        else:
            kept.append(0)
    kept = np.array(kept)
    if sum(kept) == 0:
        kept = np.ones(len(kept))
    kept = kept.astype(bool)

    files = np.array(files)[kept]

    plt.figure(figsize=(18,12))
    plt.subplots_adjust(left=0.06,right=0.96,hspace=0.60,top=0.95,bottom=0.15,wspace=0.30)
    for f in files:
        ins = f.split('/WORKSPACE')[0].split('/')[-1]
        code = ins[0]+ins.split('_')[0][-2:]+'_'+ins.split('_')[1]
        count += 1
        table = pd.read_csv(f)
        for kw in all_variables:
            if kw not in table.keys():
                table[kw] = np.nan
        extract.append([ins]+list(np.array(table[all_variables].mean()))+list(np.array(table[all_variables].std())))
        for j,kw in enumerate(variables):
            if kw in variables:
                plt.subplot(3,3,j+1)
                plt.boxplot(np.array(table[kw]),positions=[count],showfliers=False,labels=[code],widths=[0.5])
                plt.ylabel(kw,fontsize=14)
                plt.xticks(rotation=45,ha='right')
                save[kw].append(np.array(table[kw]))

    ylims = {}
    for j,kw in enumerate(variables):
        vector = np.hstack(save[kw])
        vector = vector[vector==vector]
        if len(vector)>0:
            plt.subplot(3,3,j+1)
            med_val = np.nanmedian(vector)
            sup_val = np.nanpercentile(vector,75) - med_val
            inf_val = med_val - np.nanpercentile(vector,25)
            ylim = [med_val-5*inf_val,med_val+5*sup_val]
            plt.boxplot(vector,positions=[count+2],showfliers=False,labels=['ALL'],widths=[0.5],patch_artist=True,boxprops=dict(facecolor='lightsteelblue',edgecolor='black',linewidth=1.))
            plt.ylim(ylim)

            if kw=='teff':
                plt.title('%s = %.0f +/- %.0f'%(kw,np.nanmedian(vector),mad(vector)))
            else:
                plt.title('%s = %.2f +/- %.2f'%(kw,np.nanmedian(vector),mad(vector)))
            plt.xticks(rotation=90,ha='center')

            if kw in reference.keys():
                plt.axhline(y=reference[kw], ls='-.', color='k', alpha=0.7, lw=1)

            ylims[kw] = ylim

    return table, extract, ylims, all_variables, kept


def pdf_comparison(files, xlims={}, reference={}):

    count = -1

    variables = ['teff','logg','feh','ms','rs','vsini','mhk','rhk','age']
    save = {kw:[] for kw in variables}
    pdf = {kw:{} for kw in variables}
    pdf_grid = {kw:[] for kw in variables}

    plt.figure(figsize=(18,12))
    plt.subplots_adjust(left=0.06,right=0.96,hspace=0.60,top=0.95,bottom=0.15,wspace=0.30)

    instrument = [re.sub(r"\d+$", "", ins.split('_')[0]) for ins in [f.split('/WORKSPACE')[0].split('/')[-1] for f in files]] #to merge HARPS03 HARPS15

    count = -1
    ax = plt.axes([0.1,0,0.8,0.10])
    plt.axis('off')
    for i in np.unique(instrument):
        count+=1
        ax.plot([],[],color='C%.0f'%(count),label=i)
    ax.legend(ncol=10,loc=1,fontsize=12)

    all_ins = ['REFERENCE']+['ALLINS_MIXTURE']+['ALLINS_BAYESIAN']+list(np.unique(instrument))
    columns = ['ins'] + [item for kw in variables for item in ['%s'%kw,'%s_std'%kw]]
    table_out = np.ones((len(all_ins),len(columns)))*np.nan
    table_out = pd.DataFrame(table_out,columns=columns)
    table_out['ins'] = all_ins

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
                    cquantile2 = np.percentile(vec,[16,50,84])
                    table_out.loc[table_out['ins']==i,kw] = cquantile2[1]
                    table_out.loc[table_out['ins']==i,kw+'_std'] = 0.5*(cquantile2[2]-cquantile2[0])
                    plt.subplot(3,3,j+1)
                    a,b = np.histogram(vec, bins=bins, density=True)
                    a = a/np.sum(a)
                    plt.plot(b[:-1], a, color='C%.0f'%(count), alpha=0.5)
                    plt.xlabel(kw,fontsize=14)
                    plt.yticks(ha='right')
                    save[kw].append(vec)
                    pdf[kw][ins] = a
                    pdf_grid[kw] = b[:-1]

    for j,kw in enumerate(variables):
        pdf[kw] = pd.DataFrame(pdf[kw])
        pdf[kw]['grid'] = pdf_grid[kw]

    for j,kw in enumerate(variables):

        pdf_tot = pdf[kw].copy()
        del pdf_tot['grid']
        
        ptot = np.product(pdf_tot,axis=1)
        ptot /= np.sum(ptot)
        cdf = np.cumsum(ptot)

        plt.subplot(3,3,j+1)

        if len(pdf_grid[kw])>1:

            cdf = interp1d(pdf_grid[kw], cdf, kind='cubic', bounds_error=False, fill_value='extrapolate')(np.linspace(pdf_grid[kw][0], pdf_grid[kw][-1], 1000))
            cquantile = np.interp([0.16,0.50,0.84], cdf, np.linspace(pdf_grid[kw][0], pdf_grid[kw][-1], 1000))

            save[kw] = np.hstack(save[kw])

            cquantile2 = np.percentile(save[kw],[16,50,84])

            med_val = np.median(np.ravel(save[kw]))
            sup_val = np.nanpercentile(np.ravel(save[kw]),75) - med_val
            inf_val = med_val - np.nanpercentile(np.ravel(save[kw]),25)
            xlim = [med_val-5*inf_val,med_val+5*sup_val]

            a,b = np.histogram(np.ravel(save[kw]), bins=np.linspace(xlim[0], xlim[1], 50), density=True)
            x = 0.5 * (b[:-1] + b[1:])
            y = a / np.sum(a)

            pdf[kw]['mixture'] = y
            pdf[kw]['bayesian'] = ptot

            plt.plot(x,y, alpha=1.0, label='ALL', color='k',lw=2.5)
            plt.plot(pdf_grid[kw], 3*np.max(y)-ptot/np.max(ptot)*np.max(y), alpha=1.0, label='BAYE', color='k',lw=2,ls='-')
            plt.fill_between(pdf_grid[kw], 3*np.max(y)-ptot/np.max(ptot)*np.max(y), 3*np.max(y), color='k', alpha=0.10)
            plt.tick_params(top=True,direction='inout')

            for i1,i2 in zip([16,2.5],[84,97.5]):
                p1, p2 = np.percentile(np.ravel(save[kw]), [i1, i2])
                mask = (x >= p1) & (x <= p2)
                plt.fill_between(x, 0, y, where=mask, color='k', alpha=0.10)

            val = cquantile[1]
            val2 = cquantile2[1]

            sval = 0.5*(cquantile[2]-cquantile[0])
            sval2 = 0.5*(cquantile2[2]-cquantile2[0])
        else:
            val = np.nan
            val2 = np.nan
            sval = np.nan
            sval2 = np.nan
            xlim = [0,1]

        plt.ylabel('PDF',fontsize=14)
        if kw=='age':
            if xlim[0]<0:
                xlim[0] = 0
            if xlim[1]>13:
                xlim[1] = 13
        plt.xlim(xlim)
        plt.ylim(0,3*np.max(y))


        table_out.loc[table_out['ins']=='ALLINS_MIXTURE',kw] = val2
        table_out.loc[table_out['ins']=='ALLINS_MIXTURE',kw+'_std'] = sval2

        table_out.loc[table_out['ins']=='ALLINS_BAYESIAN',kw] = val
        table_out.loc[table_out['ins']=='ALLINS_BAYESIAN',kw+'_std'] = sval


        if kw=='teff':
            title = '%s = %.0f +/- %.0f (BAYE)\n'%(kw,val,sval)
            title = title+'%s = %.0f +/- %.0f (MIXT)'%(kw,val2,sval2)
        else:
            title = '%s = %.2f +/- %.2f (BAYE)\n'%(kw,val,sval)
            title = title+'%s = %.2f +/- %.2f (MIXT)'%(kw,val2,sval2)

        plt.title(title)

        if kw in reference.keys():
            table_out.loc[table_out['ins']=='REFERENCE',kw] = reference[kw]
            table_out.loc[table_out['ins']=='REFERENCE',kw+'_std'] = 0
            plt.axvline(x=reference[kw], ls='-.', color='k', alpha=0.7, lw=1)
        pdf[kw] = np.round(pdf[kw],6)

    table_out = np.round(table_out,3)
    table_out['teff'] = np.round(table_out['teff'],0)
    table_out['teff_std'] = np.round(table_out['teff_std'],0)
    
    return pdf, table_out   

def get_atmos_table(dir_root):
    if not os.path.exists(dir_root+'/Atmos_literature.csv'):
        kw = ['teff','logg','feh','ms','rs','vsini','mhk','rhk','age','sini','prot']
        values = np.array([np.nan]+[np.nan]*2*len(kw))[:,np.newaxis].T
        db = pd.DataFrame(values,columns=['ins']+['%s'%kw for kw in kw]+['%s_std'%kw for kw in kw])
        db.to_csv(dir_root+'/Atmos_literature.csv')
    else:
        db = pd.read_csv(dir_root+'/Atmos_literature.csv',index_col=0)
    return db


def repair_csv(db_file = '/Volumes/MyPirisi/Atmos_parameters/SNAKY_processed_db.csv'):
    try:
        file = pd.read_csv(db_file)
    except:
        file = pd.read_fwf(db_file)
        lines = [f[0].split(',') for f in file.values]
        counts = [len(l) for l in lines]
        excluded = np.where(np.array(counts)!=8)[0]
        new_file = []
        for n,f in enumerate(lines):
            if n not in excluded:
                new_file.append(f)
        file = pd.DataFrame(new_file,columns=['index','starname','ins','snaky_version','job_id','processed','time','csv_file'])
        del file['index']
        file.to_csv(db_file)
