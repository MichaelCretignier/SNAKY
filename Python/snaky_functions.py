"""
@author: Cretignier Michael 
@university University of Geneva
"""

import os
import pickle
import sys
import time
import warnings

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from astropy import units as u
from astropy.coordinates import EarthLocation
import astropy.time as Time

from scipy import ndimage, signal
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from tqdm import tqdm

import snaky_variables as myv

try:
    np.warnings.filterwarnings('ignore', category=RuntimeWarning)
except:
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

pickle_protocol_version = 5

#astronomical units

udeg = 1*u.deg 
uarcmin = 1*u.arcmin 

#astronomical constant 

Mass_sun = 1.99e30
Mass_earth = 5.97e24
Mass_jupiter = 1.89e27

radius_sun = 696343*1000
radius_earth = 6352*100
G_cst = 6.67e-11
au_m = 149597871*1000

cwd = os.getcwd()
root = '/'.join(cwd.split('/')[:-1])

# statistical

def find_turbulence(teff, logg):
    """From Bruntt+10 """
    
    if logg>4.0:
        DT = teff-5700

        vmac = 2.26 + 2.90e-3*DT + 5.86e-7*DT**2  #Eq 9  (5000-6500K + log<4.0) 
        vmic = 1.01 + 4.56e-4*DT + 2.75e-7*DT**2  #Eq 10 (5000-6500K + log<4.0)

        if teff<4750:
            vmac = 0.5
            vmin = 1.0
        if teff>6500:
            vmin = 1.5
            vmac = 4.5
    else:
        vmic = 1.0
        vmac = 1.0

    return vmic,vmac

def find_stellar_mass_radius_MS(Teff, logg):
    samples_T = np.random.randn(10000)*75+Teff
    samples_g = np.random.randn(10000)*0.07+logg

    samples_m = (samples_T/5772)**(4/3)*(10**(samples_g-4.437))**(-1/3)
    mass = np.median(samples_m)
    mass_std = mad(samples_m)

    samples_m = samples_m[samples_m>0]

    samples_R = 0.5*(4.437+np.log10(samples_m)-samples_g) #Smette 2005
    radius = np.median(10**samples_R)
    radius_std = mad(10**samples_R)

    return mass, mass_std, radius, radius_std, samples_m, 10**samples_R

def find_stellar_mass_radius(Teff, sp_type='G2V'):
    """Habets 1981 calibration curve"""
    lim=0
    for k in sp_type[::-1]:
        try:
            int(k)
            break
        except:
            lim+=1
    
    class_lum = sp_type[len(sp_type)-lim:]
    if class_lum=='':
        class_lum='V'    
    if class_lum!='V':
        class_lum='IV'
    calib = pd.read_pickle(root+'/Python/Material_snaky/logT_logM_logR.p')[class_lum]

    m = interp1d(10**calib['log(T)'],10**calib['log(M/Ms)'], kind='linear', bounds_error=False, fill_value='extrapolate')(np.array([Teff]))
    r = interp1d(10**calib['log(T)'],10**calib['log(R/Rs)'], kind='linear', bounds_error=False, fill_value='extrapolate')(np.array([Teff]))

    log_g = 2+np.log10(6.67e-11*(m*1.98e30)/(r*696342000)**2)
    return m[0], r[0], log_g[0]


def get_info_lvl2(file,kw1,kw2):
    try:
        value = file[kw1][kw2]
    except:
        value = np.nan
    return value

def current_time():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def pickle_dump(obj,obj_file,protocol=None):
    if protocol is None:
        protocol = pickle_protocol_version
    pickle.dump(obj, obj_file, protocol=protocol)

def parabole(x, a, b, c):
    return a+b*x+c*x**2

def mad(array,axis=0,sigma_conv=True):
    """"""
    if axis == 0:
        step = abs(array-np.nanmedian(array,axis=axis))
    else:
        step = abs(array-np.nanmedian(array,axis=axis)[:,np.newaxis])
    return np.nanmedian(step,axis=axis)*[1,1.48][int(sigma_conv)]

def touch_dir(path):
    if not os.path.exists(path):
        os.system('mkdir -p '+path)

def touch_pickle(filename):
    if not os.path.exists(filename):
        pickle_dump({},open(filename,'wb'))
        return {}
    else:
        return pd.read_pickle(filename)

def touch_npy(filename):
    if not os.path.exists(filename):
        arr = np.array([]) 
        np.save(filename, arr)
        return arr
    else:
        return np.load(filename)

def local_max(spectre,vicinity):
    vec_base = spectre[vicinity:-vicinity]
    maxima = np.ones(len(vec_base))
    for k in range(1,vicinity):
        maxima *= 0.5*(1+np.sign(vec_base - spectre[vicinity-k:-vicinity-k]))*0.5*(1+np.sign(vec_base - spectre[vicinity+k:-vicinity+k]))
    
    index = np.where(maxima==1)[0]+vicinity
    if len(index)==0:
        index = np.array([0,len(spectre)-1])
    flux = spectre[index]       
    return np.array([index,flux])

def smooth(y, box_pts, shape='rectangular'): #rectangular kernel for the smoothing
    box2_pts = int(2*box_pts-1)
    if type(shape)==int:
        y_smooth = np.ravel(pd.DataFrame(y).rolling(box_pts,min_periods=1,center=True).quantile(shape/100))
    
    elif shape=='savgol':
        if box2_pts>=5:
            y_smooth = savgol_filter(y, box2_pts, 3)
        else:
            y_smooth = y
    else:
        if shape=='rectangular':
            box = np.ones(box2_pts)/box2_pts
        if shape == 'gaussian':
            vec = np.arange(-25,26)
            box = norm.pdf(vec,scale=(box2_pts-0.99)/2.35)/np.sum(norm.pdf(vec,scale = (box2_pts-0.99)/2.35))
        y_smooth = np.convolve(y, box, mode='same')
        y_smooth[0:int((len(box)-1)/2)] = y[0:int((len(box)-1)/2)]
        y_smooth[-int((len(box)-1)/2):] = y[-int((len(box)-1)/2):]
    return y_smooth

def conv_smw_mhk(smw,teff):
    teff[teff>6100] = 6100
    teff[teff<3000] = 3000
    slope = 100*hinge_soft(teff/5778, -0.41346, 0.25300, 28.2320, 0.78761, 20.9049)
    intercept = 100*hinge_soft(teff/5778, 35.520, 6.71928, -6.4585, -0.00794, 2.8009)
    return smw*slope + intercept

def conv_rhk_prot(log_rhk, bv):
    """From Noyes 1984 and Mamajek 2008"""
    x = 1 - bv
    y = log_rhk+5
    log_t = int(x>0)*(1.362 - 0.166*x + 0.025*x**2 - 5.323*x**3) + int(x<0)*(1.362 - 0.14*x)
    log_p = log_t + 0.324 - 0.4*y - 0.283*y**2 - 1.325*y**3
    
    prot_n84 = 10**log_p
    sig_prot_n84 = np.log(10)*0.08*prot_n84

    prot_m08 = (0.808-2.966*(log_rhk+4.52))*10**log_t
    sig_prot_m08 = 4.4*bv*1.7-1.7
    
    if (prot_m08 > 0.) & (bv >= 0.50):
        age_m08 = 1e-3*(prot_m08/0.407/(bv-0.495)**0.325)**(1./0.566)
        sig_age_m08 = 0.05*np.log(10)*age_m08
    else:
        age_m08 = 0.0
        sig_age_m08 = 0.0
    
    return prot_n84, sig_prot_n84, prot_m08, sig_prot_m08, age_m08, sig_age_m08


def find_nearest(array,value,dist_abs=True,closest='abs'):
    if type(array)!=np.ndarray:
        array = np.array(array)
    if type(value)!=np.ndarray:
        value = np.array([value])
    
    array[np.isnan(array)] = 1e16

    dist = array-value[:,np.newaxis]
    if closest=='low':
        dist[dist>0] = -np.inf
    elif closest=='high':
        dist[dist<0] = np.inf
    idx = np.argmin(np.abs(dist),axis=1)
    
    distance = abs(array[idx]-value) 
    if dist_abs==False:
        distance = array[idx]-value
    return idx, array[idx], distance

def find_nearest_ram_friendly(array1,array2,delta,extra_search=1000):

    closest_indice = []
    for i in array2: #to spare RAM memory because of find_nearest algo
        estimated = int((i-np.min(array1))/delta)
        cut_left = np.max([0,estimated-extra_search])
        cut_right = np.min([estimated+extra_search,len(array1)])
        closest_indice.append(cut_left+find_nearest(array1[cut_left:cut_right],i)[0][0])
    closest_indice = np.array(closest_indice)

    return closest_indice


def find_nearest_ndim(array, value, znorm=True):
    if type(array)!=np.ndarray:
        array = np.array(array)
    if type(value)!=np.ndarray:
        value = np.array([value])
    
    array[np.isnan(array)] = 1e16
    
    array1 = array.copy()
    array2 = value.copy()
    
    if znorm:
        med_vec = np.median(array1,axis=1)[:,np.newaxis]
        mad_vec = mad(array1,axis=1)[:,np.newaxis]
    else:
        med_vec = np.zeros(len(array1)).astype(type(array1[0]))[:,np.newaxis]
        mad_vec = np.ones(len(array1)).astype(type(array1[0]))[:,np.newaxis]
        
    array1 = array1 - med_vec
    array2 = array2 - med_vec
    
    array1/= mad_vec
    array2/= mad_vec
    
    
    dist = np.sum([abs(array1[j]-array2[j][:,np.newaxis]) for j in np.arange(len(array1))],axis=0)
    idx = np.argmin(dist,axis=1)    
        
    distance = np.array([np.sum(abs(array1[:,i1]-array2[:,i2])) for i1,i2 in zip(idx,np.arange(len(array2.T)))])
    
    return idx, array1[:,idx], distance

def my_colormesh(x,y,z,cmap='seismic',vmin=None,vmax=None,zoom=1,shading='auto', return_output=False, order=3, smooth_box=1, alpha=1, grid=False):
    
    dx = x[-1] - x[-2] 
    dy = y[-1] - y[-2] 
    x,y = np.meshgrid(x,y)

    x = np.hstack([x,x[:,-1][:,np.newaxis]+dx])
    x = np.vstack([x,x[-1,:]])

    y = np.hstack([y,y[:,-1][:,np.newaxis]])
    y = np.vstack([y,y[-1,:]+dy])
    
    z = np.hstack([z,z[:,-1][:,np.newaxis]])
    z = np.vstack([z,z[-1,:]])
    
    z = smooth2d(z,smooth_box,borders=False)
    
    if zoom!=1:
        Z = ndimage.zoom(z, zoom, order=order)
    else:
        Z = z
    X = ndimage.zoom(x, zoom, order=order)
    Y = ndimage.zoom(y, zoom, order=order)
    
    if return_output:
        return X,Y,Z
    else:
        plt.pcolormesh(X,Y,Z,shading=shading,cmap=cmap,vmin=vmin,vmax=vmax,alpha=alpha) 
        if grid:
            for yi in np.unique(y)[1:]-dy/2:
                plt.axhline(y=yi,color='k',alpha=0.2,lw=1)
            for xi in np.unique(x)[1:]-dx/2:
                plt.axvline(x=xi,color='k',alpha=0.2,lw=1)

def clustering(array, tresh, num):
    difference = abs(np.diff(array))
    cluster = (difference<tresh)
    if len(cluster)>0:
        indice = np.arange(len(cluster))[cluster]
        
        j = 0
        border_left = [indice[0]]
        border_right = []
        while j < len(indice)-1:
            if indice[j]==indice[j+1]-1:
                j+=1
            else:
                border_right.append(indice[j])
                border_left.append(indice[j+1])
                j+=1
        border_right.append(indice[-1])        
        border = np.array([border_left,border_right]).T
        border = np.hstack([border,(1+border[:,1]-border[:,0])[:,np.newaxis]])
        
        kept = []
        for j in range(len(border)):
            if border[j,-1]>=num:
                kept.append(array[border[j,0]:border[j,1]+2])
        return kept, border
    else:
        print('no cluster found with such treshhold')

def flat_clustering(length,cluster_output,extended=0,elevation=1):
    vec_init = np.arange(length)
    if type(elevation)==int:
        elevation = np.ones(len(cluster_output))*elevation

    total = length*len(cluster_output)
    total_cut = np.max([1,int(total/(300000*100))]) #a classical STS YARARA

    if True: #RAM user friendly
        vecs = np.array_split(vec_init,total_cut)
    else: #old recipe
        vecs = [vec_init]
    
    flat = []
    for vec in vecs:
        larger = (vec >= (cluster_output[:,0][:,np.newaxis]-extended)).astype('int')*elevation[:,np.newaxis]
        smaller = (vec <= (cluster_output[:,1][:,np.newaxis]+1+extended)).astype('int')*elevation[:,np.newaxis]
        flat.append(np.sqrt(np.sum(larger*smaller,axis=0)))
    flat = np.hstack(flat)
    return flat
    

def identify_nearest(array1,array2):
    """identify the closest elements in array2 of array1"""
    array1 = np.sort(array1)
    array2 = np.sort(array2)
    
    identification = []
    
    begin=0
    for value in tqdm(array1):
        begin2 = find_nearest(array2[begin:],value)[0]
        identification.append(begin2+begin)
        begin=int(begin2)
    return np.ravel(identification)
    
def match_unique_closest(array1, array2):
    """return a table [idx1,idx2,num1,num2,distance] matching the closest element from an array to the other, each pair is unique. Remark : algorithm very slow by conception if the arrays are too large."""
    if type(array1)!=np.ndarray:
        array1 = np.array(array1)
    if type(array2)!=np.ndarray:
        array2 = np.array(array2)    
    if not (np.product(~np.isnan(array1))*np.product(~np.isnan(array2))):
        print('there is a nan value in your list, remove it first to be sure of the algorithme reliability')
    index1 = np.arange(len(array1))[~np.isnan(array1)] ; index2 = np.arange(len(array2))[~np.isnan(array2)]  
    array1 = array1[~np.isnan(array1)] ;  array2 = array2[~np.isnan(array2)]
    liste1 = np.arange(len(array1))[:,np.newaxis]*np.hstack([np.ones(len(array1))[:,np.newaxis],np.zeros(len(array1))[:,np.newaxis]])
    liste2 = np.arange(len(array2))[:,np.newaxis]*np.hstack([np.ones(len(array2))[:,np.newaxis],np.zeros(len(array2))[:,np.newaxis]])
    liste1 = liste1.astype('int') ; liste2 = liste2.astype('int')
    
    if len(array1)>1:
        dmin = np.diff(np.sort(array1)).min()
    else:
        dmin=0
    if len(array2)>1:
        dmin2 = np.diff(np.sort(array2)).min()
    else:
        dmin2=0
    array1_r = array1 + 0.001*dmin*np.random.randn(len(array1))
    array2_r = array2 + 0.001*dmin2*np.random.randn(len(array2))

    m = array2_r-array1_r[:,np.newaxis]
    m_line = np.ones(len(array2_r))*np.arange(len(array1_r))[:,np.newaxis]
    m_col = np.arange(len(array2_r))*np.ones(len(array1_r))[:,np.newaxis]
    
    
    save = []
    
    for j in range(np.min([len(array1_r),len(array2_r)])):
        line,col = np.where(m==np.nanmin(abs(m)))
        if len(line):
            line = line[0]
            col = col[0]
            save.append([m_line[line][col], m_col[line][col], array1_r[line],  array2_r[col], m[line][col]])
            
            m = np.delete(m,line,axis=0)
            m = np.delete(m,col,axis=1)
    
            m_col = np.delete(m_col,line,axis=0)
            m_col = np.delete(m_col,col,axis=1)    
    
            m_line = np.delete(m_line,line,axis=0)
            m_line = np.delete(m_line,col,axis=1)    
        else:
            break
    
    save = np.array(save)
    
    return save     
    
def match_nearest(array1, array2,fast=True,max_dist=None,random=True):
    """return a table [idx1,idx2,num1,num2,distance] matching the closest element from two arrays. Remark : algorithm very slow by conception if the arrays are too large."""
    if type(array1)!=np.ndarray:
        array1 = np.array(array1)
    if type(array2)!=np.ndarray:
        array2 = np.array(array2)    
    if not (np.product(~np.isnan(array1))*np.product(~np.isnan(array2))):
        print('there is a nan value in your list, remove it first to be sure of the algorithme reliability')
    index1 = np.arange(len(array1))[~np.isnan(array1)] ; index2 = np.arange(len(array2))[~np.isnan(array2)]  
    array1 = array1[~np.isnan(array1)] ;  array2 = array2[~np.isnan(array2)]
    liste1 = np.arange(len(array1))[:,np.newaxis]*np.hstack([np.ones(len(array1))[:,np.newaxis],np.zeros(len(array1))[:,np.newaxis]])
    liste2 = np.arange(len(array2))[:,np.newaxis]*np.hstack([np.ones(len(array2))[:,np.newaxis],np.zeros(len(array2))[:,np.newaxis]])
    liste1 = liste1.astype('int') ; liste2 = liste2.astype('int')
    
    if fast:
        #ensure that the probability for two close value to be the same is null
        if len(array1)>1:
            dmin = np.diff(np.sort(array1)).min()
        else:
            dmin=0
        if len(array2)>1:
            dmin2 = np.diff(np.sort(array2)).min()
        else:
            dmin2=0
        array1_r = array1 + int(random)*0.001*dmin*np.random.randn(len(array1))
        array2_r = array2 + int(random)*0.001*dmin2*np.random.randn(len(array2))
        #match nearest
        m = abs(array2_r-array1_r[:,np.newaxis])
        arg1 = np.argmin(m,axis=0)
        arg2 = np.argmin(m,axis=1)
        mask = (np.arange(len(arg1)) == arg2[arg1])
        liste_idx1 = arg1[mask]
        liste_idx2 = arg2[arg1[mask]]
        array1_k = array1[liste_idx1]
        array2_k = array2[liste_idx2]

        liste_idx1 = index1[liste_idx1]
        liste_idx2 = index2[liste_idx2] 
        
        mat = np.hstack([liste_idx1[:,np.newaxis],liste_idx2[:,np.newaxis],
                          array1_k[:,np.newaxis],array2_k[:,np.newaxis],(array1_k-array2_k)[:,np.newaxis]]) 
        
        if max_dist is not None:
           mat = mat[(abs(mat[:,-1])<max_dist)]
        
        return mat
             
    else:
        for num,j in enumerate(array1):
            liste1[num,1] = int(find_nearest(array2,j)[0])
        for num,j in enumerate(array2):
            liste2[num,1] = int(find_nearest(array1,j)[0])
            
        save = liste2[:,0].copy()
        liste2[:,0] = liste2[:,1].copy()
        liste2[:,1] = save.copy() 
        
        liste1 = np.vstack([liste1,liste2])
        liste = []
        for j in np.unique(liste1,axis=0):
            if np.sum(np.product(liste1 == j.astype(tuple),axis=1))==2:
                liste.append(j)
        liste = np.array(liste)
        distance = []
        for j in liste[:,0]:
            distance.append(find_nearest(array2,array1[j],dist_abs=False)[2])
        
        liste_idx1 = index1[liste[:,0]]
        liste_idx2 = index2[liste[:,1]] 
        
        mat = np.hstack([liste_idx1[:,np.newaxis],liste_idx2[:,np.newaxis],array1[liste[:,0],np.newaxis],array2[liste[:,1],np.newaxis],np.array(distance)[:,np.newaxis]])
        
        if max_dist is not None:
            mat = mat[(abs(mat[:,-1])<max_dist)]
        
        return mat
  

def planck_function(wave,Teff):
    y = 2*myv.h_planck*myv.c_lum/wave**5*(np.exp(myv.h_planck*myv.c_lum/(myv.k_boltz*Teff*wave*1e-10))-1)**-1
    return y

def black_body_ratio(T0,teff,wave):
    if type(teff)==np.ndarray:
        planck_factor = (np.exp(myv.h_planck*myv.c_lum/(myv.k_boltz*T0*wave*1e-10))-1)/(np.exp(myv.h_planck*myv.c_lum/(myv.k_boltz*teff[:,np.newaxis]*wave*1e-10))-1)
    else:
        planck_factor = (np.exp(myv.h_planck*myv.c_lum/(myv.k_boltz*T0*wave*1e-10))-1)/(np.exp(myv.h_planck*myv.c_lum/(myv.k_boltz*teff*wave*1e-10))-1)
    return planck_factor


def ccf(wave, spec1, spec2, extended=1500, rv_range=45, oversampling=10, spec1_std=None):
    "CCF for a equidistant grid in log wavelength spec1 = spectrum, spec2 =  binary mask"   
    dwave = np.median(np.diff(wave))
    
    if spec1_std is None:
        spec1_std = np.zeros(np.shape(spec1))
    
    if len(np.shape(spec1))==1:
        spec1 = spec1[:,np.newaxis].T
    if len(np.shape(spec1_std))==1:
        spec1_std = spec1_std[:,np.newaxis].T
    #spec1 = np.hstack([np.ones(extended),spec1,np.ones(extended)])
    
    spec1 = np.hstack([np.ones((len(spec1),extended)),spec1,np.ones((len(spec1),extended))])
    spec2 = np.hstack([np.zeros(extended),spec2,np.zeros(extended)])
    spec1_std = np.hstack([np.zeros((len(spec1_std),extended)), spec1_std, np.zeros((len(spec1_std),extended))])
    wave = np.hstack([np.arange(-extended*dwave+wave.min(),wave.min(),dwave),wave,np.arange(wave.max()+dwave,(extended+1)*dwave+wave.max(),dwave)])
    shift = np.linspace(0,dwave,oversampling+1)[:-1]
    shift_save = []
    sum_spec = np.nansum(spec2)
    convolution = []
    convolution_std = []
    
    rv_max = int(np.log10((rv_range/299.792e3)+1)/dwave)
    for j in tqdm(shift):
        new_spec = interp1d(wave+j,spec2,kind='cubic', bounds_error=False, fill_value='extrapolate')(wave)
        for k in np.arange(-rv_max,rv_max+1,1):
            new_spec2 = np.hstack([new_spec[-k:],new_spec[:-k]])
            convolution.append(np.nansum(new_spec2*spec1,axis=1)/sum_spec)
            convolution_std.append(np.sqrt(np.abs(np.nansum(new_spec2*spec1_std**2,axis=1)))/sum_spec)
            shift_save.append(j+k*dwave)
    shift_save = np.array(shift_save)
    sorting = np.argsort(shift_save)
    return (299.792e6*10**shift_save[sorting])-299.792e6, np.array(convolution)[sorting], np.array(convolution_std)[sorting]
    
def GND(x,cen,amp,offset,wid,beta):
    """Based on Heitzmann+21"""
    return amp*np.exp(-0.5*np.abs((x-cen)/wid)**(beta))+offset

def gaussian(x, cen, amp, offset, wid):
    """width the classical sigma (FWHM=2.355*sigma)"""
    return amp * np.exp(-(x-cen)**2 / (2*wid**2))+offset    


def substract_model(x, y, *par):
    if np.shape(par[0])==(): 
        a, b = par[0], par[1]
    else:
        a, b = par[0]
    model = a*x+b
    return y-model
#

def string_contained_in(array,string,inv=False,exclusion=[]):
    array = np.array(array)
    split = np.array([len(i.split(string))-1 for i in array])
    mask = split.astype('bool')
    if inv:
        mask = ~mask

    for exclu in exclusion:
        split = np.array([len(i.split(exclu))-1 for i in array])
        mask = mask&(~split.astype('bool'))

    return mask, array[mask]
    

def rm_outliers(array, m=1.5, kind='sigma',axis=0, return_borders=False,Plot=False):
    if type(array)!=np.ndarray:
        array=np.array(array)
    
    if m!=0:
        array[array==np.inf] = np.nan
        #array[array!=array] = np.nan
        
        if kind == 'inter':
            interquartile = np.nanpercentile(array, 75, axis=axis) - np.nanpercentile(array, 25, axis=axis)
            inf = np.nanpercentile(array, 25, axis=axis)-m*interquartile
            sup = np.nanpercentile(array, 75, axis=axis)+m*interquartile    
            if axis==0:        
                mask = (array >= inf)&(array <= sup)
            else:
                mask = (array.T >= inf)&(array.T <= sup)
                mask = mask.T
        if kind == 'sigma':
            sup = np.nanmean(array, axis=axis) + m * np.nanstd(array, axis=axis)
            inf = np.nanmean(array, axis=axis) - m * np.nanstd(array, axis=axis)
            mask = abs(array-np.nanmean(array, axis=axis)) <= m * np.nanstd(array, axis=axis)
        if kind =='mad':
            median = np.nanmedian(array, axis=axis)
            mad = np.nanmedian(abs(array-median), axis=axis)
            sup = median+m * mad * 1.48
            inf = median-m * mad * 1.48
            if axis==0:        
                mask = (array >= inf)&(array <= sup)
            else:
                mask = (array.T >= inf)&(array.T <= sup)
                mask = mask.T            
    else:
        mask = np.ones(len(array)).astype('bool')
    
    if Plot:
        plt.plot(array)
        plt.plot(np.arange(len(array))[mask],array[mask])

    if return_borders:
        return mask,  array[mask], sup, inf        
    else:
        return mask,  array[mask]


def conv_void_air(wave):
    s2 = 1e4/wave
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s2) + 0.00015998 / (38.9 - s2)
    return wave/n

def conv_air_void(wave):
    s2 = 1e4/wave
    n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s2) + 0.0001599740894897 / (38.92568793293 - s2)
    return wave*n

def gaus(x,x0,sigma,norm=False):
    if not norm:
        return np.sqrt(2.0*np.pi*sigma**2)*np.exp(-(x-x0)**2/(2*sigma**2))
    if norm:
        return np.sqrt(2.0*np.pi*sigma**2)*np.exp(-(x-x0)**2/(2*sigma**2))/np.max(np.sqrt(2.0*np.pi*sigma**2)*np.exp(-(x-x0)**2/(2*sigma**2)))

def broadGaussFast(x, y, sigma, edgeHandling=None, maxsig=None):
    dxs = x[1:] - x[0:-1]
    if maxsig is None:
        lx = len(x)
    else:
        lx = int(((sigma * maxsig) / dxs[0]) * 2.0) + 1
    nx = (np.arange(lx, dtype=int) - sum(divmod(lx, 2)) + 1) * dxs[0]
    e = gaus(nx,0,sigma)
    e /= np.sum(e)
    if edgeHandling == "firstlast":
        nf = len(y)
        y = np.concatenate((np.ones(nf) * y[0], y, np.ones(nf) * y[-1]))
        result = np.convolve(y, e, mode="same")[nf:-nf]
    elif edgeHandling is None:
        result = np.convolve(y, e, mode="same")
    return result

def instrBroadGaussFast(wvl, flux, resolution, edgeHandling=None, fullout=False, maxsig=None):
    meanWvl = np.mean(wvl)
    fwhm = 1.0 / float(resolution) * meanWvl
    sigma = fwhm / (2.0 * np.sqrt(2. * np.log(2.)))

    result = broadGaussFast(
        wvl, flux, sigma, edgeHandling=edgeHandling, maxsig=maxsig)

    if not fullout:
        return result
    else:
        return (result, fwhm)    

def print_box(sentence):
    print('\n')
    print('L'*len(sentence))
    print(sentence)
    print('T'*len(sentence))
    print('\n')

def doppler_r(lamb,v):
    """Relativistic Doppler. Take (wavelength, velocity in [m/s]) and return lambda observed and lambda source"""
    c= 299.792e6
    button=False
    factor=np.sqrt((1+v/c)/(1-v/c))
    if type(factor)!=np.ndarray:
        button=True
        factor=np.array([factor])
    lambo=lamb*factor[:,np.newaxis]
    lambs=lamb*(factor**(-1))[:,np.newaxis]
    if button:
        return lambo[0],lambs[0]
    else:
        return lambo, lambs
#

def only_axis(color=None,lw=2,ax=None,side='all',ls='-'):
    plt.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)  

def conv_time(time):    
    time = np.array(time)
    if (type(time[0])==np.float64)|(type(time[0])==np.int64):
        fmt='mjd'
        if time[0]<2030:
            fmt='decimalyear'
        elif np.mean(time)<20000:
            time+=50000
        if fmt=='mjd':
            t0 = time
            t1 = np.array([Time.Time(i, format=fmt).decimalyear for i in time])
            t2 = np.array([Time.Time(i, format=fmt).isot for i in time])
        else:
            t0 = np.array([Time.Time(i, format=fmt).mjd for i in time])
            t1 = time
            t2 = np.array([Time.Time(i, format=fmt).isot for i in time])            
    elif type(time[0])==np.str_:
        fmt='isot'
        t0 = np.array([Time.Time(i, format=fmt).jd-2400000 for i in time]) 
        t1 = np.array([Time.Time(i, format=fmt).decimalyear for i in time])
        t2 = time  
    return t0,t1,t2
    
def observatory(instrument='HARPS'):
    if instrument=='HARPS':
        obs_loc = EarthLocation(lat=-29.260972*u.deg, lon=-70.731694*u.deg, height=2400) 
    elif instrument=='HARPS03':
        obs_loc = EarthLocation(lat=-29.260972*u.deg, lon=-70.731694*u.deg, height=2400) 
    elif instrument=='HARPS15':
        obs_loc = EarthLocation(lat=-29.260972*u.deg, lon=-70.731694*u.deg, height=2400) 
    elif instrument=='HARPN':
        obs_loc = EarthLocation(lat=28.754000*u.deg, lon=-17.889055*u.deg, height=2387.2) 
    elif instrument=='ESPRESSO':
        obs_loc = EarthLocation(lat=-24.627622*u.deg, lon=-70.405075*u.deg, height=2635)
    elif instrument=='ESPRESSO18':
        obs_loc = EarthLocation(lat=-24.627622*u.deg, lon=-70.405075*u.deg, height=2635) 
    elif instrument=='ESPRESSO19':
        obs_loc = EarthLocation(lat=-24.627622*u.deg, lon=-70.405075*u.deg, height=2635)
    elif instrument=='EXPRES':
        obs_loc = EarthLocation(lat=34.74444*u.deg, lon=-68.578056*u.deg, height=2360.0)
    elif instrument=='CARMENES':
        obs_loc = EarthLocation(lat=37.223611*u.deg, lon=2.546111*u.deg, height=2168.0)
    elif instrument=='Geneva':
        obs_loc = EarthLocation(lat=46.204391*u.deg, lon=6.143158*u.deg, height=300)
    elif instrument=='CORALIE14':
        obs_loc = EarthLocation(lat=-29.260972*u.deg, lon=-70.731694*u.deg, height=2400) 
    elif instrument=='CORALIE07':
        obs_loc = EarthLocation(lat=-29.260972*u.deg, lon=-70.731694*u.deg, height=2400) 
    elif instrument=='CORALIE98':
        obs_loc = EarthLocation(lat=-29.260972*u.deg, lon=-70.731694*u.deg, height=2400) 
    elif instrument=='SOPHIE':
        obs_loc = EarthLocation(lat=43.930833*u.deg, lon=5.713333*u.deg, height=650) 
    elif instrument=='PEPSI':
        obs_loc = EarthLocation(lat=32.701388*u.deg, lon=-109.889166*u.deg, height=3221) 
    elif instrument=='UVES':
        obs_loc = EarthLocation(lat=-24.627622*u.deg, lon=-70.405075*u.deg, height=2635)
    elif instrument=='ESPADONS':
        obs_loc = EarthLocation(lat=19.8256*u.deg, lon=-155.4681*u.deg, height=4204)
    elif instrument=='NEID':
        obs_loc = EarthLocation(lat=31.9584*u.deg, lon=-111.5987*u.deg, height=2096)
    elif instrument=='KPF':
        obs_loc = EarthLocation(lat=19.8261*u.deg, lon=-155.4700*u.deg, height=4145)
    
    return obs_loc