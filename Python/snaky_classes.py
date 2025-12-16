"""
@author: Cretignier Michael 
@university University of Geneva
"""

import os
import sys

cwd = os.getcwd()
root = '/'.join(cwd.split('/')[:-1])

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from lmfit import Model, Parameters
from tqdm import tqdm
import warnings
import matplotlib.colors as mplcolors
import matplotlib.cm as cmx

import snaky_functions as myf
import snaky_variables as myv

try:
    np.warnings.filterwarnings('ignore', category=RuntimeWarning)
except:
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

#######functions

class table(object):
    """this classe has been establish with pandas DataFrame"""
    
    def __init__(self, array):
        self.table = array
        self.dim = np.shape(array)
    
    def copy(self):
        new_table = table(np.array(self.table).copy())
        return new_table
    
    def transpose(self):
        new_tab = self.table.T
        self.table = new_tab

    def plot(self, x=None, cmap='brg', new=True, color=None, xmin=None, xmax=None, alpha=0.4, plot_median=False, vmin=None, vmax=None, fontsize=16, color_label='', inv=False):
        jet = plt.get_cmap(cmap)

        table = self.table

        if color is None:
            color = np.arange(len(table))
        else:
            color= np.array(color).astype('float')
            if (len(table)!=len(color))&(len(table.T)==len(color)):
                table = table.T
            
            if (len(table)!=len(color)):
                color = np.arange(len(table))
                print('[WARNING] The color vector has not the same size (%.0f) than the table : '%(len(color)),np.shape(table))

        index = color
        mask = np.isnan(index)
        table = table[~mask]
        index = index[~mask]

        if vmin is None:
            vmin = np.nanpercentile(index,16)
        if vmax is None:
            vmax = np.nanpercentile(index,84)

        cNorm  = mplcolors.Normalize(vmin=vmin, vmax=vmax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

        if x is None:
            x = np.arange(len(table.T))
        if xmin is None:
            xmin = np.min(x)
        if xmax is None:
            xmax = np.max(x)
        
        begin = int(myf.find_nearest(x,xmin)[0][0])
        end = int(myf.find_nearest(x,xmax)[0][0])

        if new:
            plt.figure()
        
        for i,j in enumerate(table):
            colorVal = scalarMap.to_rgba(index[i])   
            if not inv:         
                plt.plot(x[begin:end],j[begin:end], color=colorVal, alpha=alpha)
            else:
                plt.plot(j[begin:end],x[begin:end], color=colorVal, alpha=alpha)

        ax = plt.gca()
        x1 = ax.get_xlim()
        y1 = ax.get_ylim()
        plt.scatter(x1,y1,c=[vmin,vmax],cmap=cmap,s=0)
        ax = plt.colorbar(pad=0)
        ax.ax.set_ylabel(color_label,fontsize=fontsize)
        ax.ax.tick_params(labelsize=fontsize)
        
        if plot_median:
            median = np.median(np.array(table),axis=0)
            plt.plot(x[begin:end],median[begin:end],color='k')

    def fit_unique_base(self, base_vec, weight=None, ortho_to=None, offset=False, perm=1, iteration=0):
        """ weights define as 1/sigma**2 """
                
        nb_vec = len(base_vec)
        if ortho_to is not None:
            base_vec = np.vstack([base_vec,ortho_to])
        if offset:
            base_vec = np.vstack([base_vec,np.ones(len(base_vec[0]))])

        tab = self.table.T

        empirical = False
        if weight is None :
            empirical = True
            weight = np.ones(np.shape(tab))
            weight_x = np.ones(np.shape(tab)[0])
        else:
            if len(np.shape(weight))>1: #if a 2D map of noise is provided
                weight_x = np.mean(weight,axis=0)
            else:
                weight_x = weight

        self.debug = []
        for itrt in range(iteration+1):
            coeff = np.linalg.lstsq(base_vec.T*np.sqrt(weight_x)[:,np.newaxis], tab*np.sqrt(weight_x)[:,np.newaxis],rcond=None)[0].T
            coeff_std = np.zeros(np.shape(coeff))

            vec_fitted = np.dot(coeff,base_vec)
            vec_residues = self.table - vec_fitted
            self.debug.append(vec_fitted)
            if iteration:
                mask_res = ~myf.rm_outliers(vec_residues,m=2,kind='inter',axis=1)[0]
                w = np.where(mask_res)
                vec_residues[w[0],w[1]] = 0
                tab = (vec_residues+vec_fitted).T

        dim1 = np.shape(tab)[0]
        dim2 = np.shape(tab)[1]

        if empirical:
            weight = np.ones(np.shape(self.table))
            mad = myf.mad(np.ravel(vec_residues))
            weight /= mad**2
            #weight = 1

        if perm>1: #bootstrap uncertainties
            noise = np.random.randn(dim1,dim2*perm)/np.sqrt(np.hstack([weight.T]*perm))
            tab = np.hstack([tab]*perm)+noise
            
            coeff = np.linalg.lstsq(base_vec.T, tab,rcond=None)[0].T
            coeff_std=np.array([np.std(coeff[i::dim2],axis=0) for i in range(dim2)])
            coeff=np.array([np.median(coeff[i::dim2],axis=0) for i in range(dim2)])

        if ortho_to is not None:
            coeff = coeff[:,0:nb_vec]
            base_vec = base_vec[0:nb_vec]
            coeff_std = coeff_std[:,0:nb_vec]

        vec_fitted = np.dot(coeff,base_vec)

        self.coeff_fitted = coeff
        self.coeff_fitted_std = coeff_std
        self.vec_fitted = vec_fitted
        vec_residues = self.table - vec_fitted
        vec_residues[self.table==0] = 0
        self.vec_residues = vec_residues
        
class tableXY(object):
    def __init__(self, x, y, *yerr):
        self.stats = pd.DataFrame({},index=[0])
        self.y = np.array(y)  #vector of y

        if x is None:# for a fast table initialisation
            x = np.arange(len(y))
        self.x = np.array(x)  #vector of x
        
        try:
            np.sum(self.y) 
        except: #in case of None
            self.y = np.zeros(len(self.x))
            yerr = [np.ones(len(self.y))]
                
        if len(x)!=len(y):
            print('X et Y have no the same lenght (%.0f vs %.0f)'%(len(x),len(y)))

        if len(yerr)!=0:
            if len(yerr)==1:
                self.yerr = np.array(yerr[0])
                self.xerr =  np.zeros(len(self.x))
            elif len(yerr)==2:
                self.xerr = np.array(yerr[0])
                self.yerr = np.array(yerr[1])
        else :
            if sum(~np.isnan(self.y.astype('float'))):
                self.yerr = np.ones(len(self.x))*myf.mad(myf.rm_outliers(self.y.astype('float'),m=2,kind='sigma')[1])
                if not np.sum(abs(self.yerr)):
                    self.yerr = np.ones(len(self.x))
            else:
                self.yerr = np.ones(len(self.x))
            self.xerr =  np.zeros(len(self.x))
        
        self.yerr = np.abs(self.yerr)
        self.xerr = np.abs(self.xerr)
        self.mask_qc = np.ones(len(self.x)).astype('bool')
            
    def null(self):
        self.yerr = 0*self.yerr

    def copy(self):
        new_table = tableXY(self.x.copy(),self.y.copy(),self.xerr.copy(),self.yerr.copy())
        new_table.mask_qc = self.mask_qc
        return new_table
    
    def diff(self,replace=True):
        diff = np.diff(self.y)/np.diff(self.x)
        new = tableXY(self.x[0:-1]+np.diff(self.x)/2,diff)
        new.interpolate(new_grid=self.x,replace=True)
        
        self.deri = tableXY(self.x,new.y,self.xerr,self.yerr)
        
        if replace:
            self.y_backup = self.y
            self.y = new.y

    def find_max(self, vicinity = 3, sort=False):
        self.index_max, self.y_max = myf.local_max(self.y,vicinity = vicinity)
        self.index_max = self.index_max.astype('int') 
        self.x_max = self.x[self.index_max.astype('int')]
        if sort:
            ordering = np.argsort(self.y_max)
            self.y_max = self.y_max[ordering]
            self.x_max = self.x_max[ordering]
        self.max_extremum = tableXY(self.x_max,self.y_max)

    def find_min(self, vicinity = 3, sort=False):
        self.index_min, self.y_min = myf.local_max(-self.y,vicinity = vicinity)
        self.index_min = self.index_min.astype('int') 
        self.y_min *=-1
        self.x_min = self.x[self.index_min.astype('int')]
        if sort:
            ordering = np.argsort(self.y_min)
            self.y_min = self.y_min[ordering]
            self.x_min = self.x_min[ordering]
        self.min_extremum = tableXY(self.x_min,self.y_min)
    
    def smooth(self,box_pts = 5,shape='rectangular',replace=True):
        self.y_smoothed = myf.smooth(self.y,box_pts,shape=shape)
        
        self.smoothed = tableXY(self.x,self.y_smoothed,self.xerr,self.yerr)
        
        if replace:
            self.x_backup = self.x.copy()
            self.y_backup = self.y.copy()
            self.xerr_backup = self.xerr.copy()
            self.yerr_backup = self.yerr.copy()
            self.y = self.y_smoothed

    def rv_shift(self,rv,method='linear',xmin=None,xmax=None,replace=True,fill_value='extrapolate',x_grid=None):
        """rv in kms, x wavelength in [\\AA]"""
        vec = self.copy()

        if x_grid is None:
            change_grid = False
            x_grid = vec.x.copy()
            i1 = 0 ; i2 = len(self.x)
        else:
            change_grid = True
            replace = False
            i1 = 0 ; i2 = len(x_grid)
            xmin = None
            xmax = None

        vec.x = myf.doppler_r(vec.x,rv*1000)[0]
        vec.interpolate(new_grid=x_grid, method=method, fill_value=fill_value)
        
        if xmin is not None:
            i1 = myf.find_nearest(vec.x,xmin)[0][0]

        if xmax is not None:
            i2 = myf.find_nearest(vec.x,xmax)[0][0]      
        
        if replace:
            self.y[i1:i2] = vec.y[i1:i2]
        else:
            if change_grid :
                self.shifted = vec.copy()
            else:
                self.shifted = self.copy()
            self.shifted.y[i1:i2] = vec.y[i1:i2]
            
    def masked(self,mask,replace=True):
        if replace:
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.xerr = self.xerr[mask]
            self.yerr = self.yerr[mask]
            self.mask_qc = self.mask_qc[mask]
        else:
            new_table = tableXY(self.x[mask],self.y[mask],self.xerr[mask],self.yerr[mask])
            new_table.mask_qc = self.mask_qc[mask]
            return new_table

    def clip(self, min=[None,None], max=[None,None], replace=True, invers=False):
        """This function seems sometimes to not work without any reason WARNING"""
        min2 = np.array(min, dtype=object).copy() ; max2 = np.array(max, dtype=object).copy()
        masky = np.ones(len(self.y)).astype('bool')
        maskx = np.ones(len(self.x)).astype('bool')

        if (min2[1] != None)|(max2[1] != None):
            if min2[1] == None:
                min2[1] = np.nanmin(self.y)-1
            if max2[1] == None:
                max2[1] = np.nanmax(self.y)+1
            masky = (self.y<=max2[1])&(self.y>=min2[1])
        
        if (min2[0] != None)|(max2[0] != None):
            if min2[0] == None:
                min2[0] = np.nanmin(self.x)-1
            if max2[0] == None:
                max2[0] = np.nanmax(self.x)+1
            maskx = (self.x<=max2[0])&(self.x>=min2[0])

        mask = maskx&masky
        try:
            self.clip_mask = self.clip_mask&mask
        except:    
            self.clip_mask = mask

        if invers:
            mask = ~mask
        self.clipped = tableXY(self.x[mask],self.y[mask],self.xerr[mask],self.yerr[mask])
        self.clipped.mask_qc = self.mask_qc[mask]
        if replace==True:
            self.x = self.x[mask] ; self.y = self.y[mask] ; self.yerr = self.yerr[mask] ; self.xerr= self.xerr[mask] ; self.mask_qc = self.mask_qc[mask]
        else:
            self.clipx = self.x[mask] ; self.clipy=self.y[mask] ; self.clipyerr = self.yerr[mask] ; self.clipxerr=self.xerr[mask]

    def interpolate(self, new_grid = 'auto', method = 'cubic', replace = True, interpolate_x=True, fill_value='extrapolate', scale='lin'):
        
        if scale!='lin':
            self.inv()
        
        if type(new_grid)==str:
            new_grid = np.linspace(self.x.min(),self.x.max(),10*len(self.x))
        if type(new_grid)==int:
            new_grid = np.linspace(self.x.min(),self.x.max(),new_grid*len(self.x))
        
        warning = 0
        if len(self.x)==len(new_grid):
            if np.sum(new_grid!=self.x)==0:
                warning = 1
        
        if warning!=1:
            if replace:
                self.x_backup = self.x.copy()
                self.y_backup = self.y.copy()
                self.xerr_backup = self.xerr.copy()
                self.yerr_backup = self.yerr.copy()  
                self.y = interp1d(self.x, self.y, kind = method, bounds_error = False, fill_value = fill_value)(new_grid)
                if np.sum(abs(self.yerr)):
                    self.yerr = interp1d(self.x, self.yerr, kind = method, bounds_error = False, fill_value = fill_value)(new_grid)
                else:
                    self.yerr = np.zeros(len(new_grid))
                if (interpolate_x)&(bool(np.sum(abs(self.xerr)))):
                    self.xerr = interp1d(self.x, self.xerr, kind = method, bounds_error = False, fill_value = fill_value)(new_grid)  
                else:
                    self.xerr = np.zeros(len(new_grid))
                self.x = new_grid
                self.mask_qc = np.ones(len(new_grid)).astype('bool')
                
                if scale!='lin':
                    self.inv()
    
            else:
                self.y_interp = interp1d(self.x, self.y, kind = method, bounds_error = False, fill_value = fill_value)(new_grid)
                if np.sum(abs(self.yerr)):
                    self.yerr_interp = interp1d(self.x, self.yerr, kind = method, bounds_error = False, fill_value = fill_value)(new_grid)
                else:
                    self.yerr_interp = np.zeros(len(new_grid))
                if (interpolate_x)&(bool(np.sum(abs(self.xerr)))):
                    self.xerr_interp = interp1d(self.x, self.xerr, kind = method, bounds_error = False, fill_value = fill_value)(new_grid)        
                else:
                    self.xerr_interp = np.zeros(len(new_grid))
                self.x_interp = new_grid
                self.interpolated = tableXY(self.x_interp,self.y_interp,self.xerr_interp,self.yerr_interp)

                if scale!='lin':
                    self.interpolated.inv()
                    self.inv()

    def supress_mask(self,mask):
        self.x = self.x[mask]
        self.y = self.y[mask]
        self.xerr = self.xerr[mask]
        self.yerr = self.yerr[mask]  
        self.mask_qc = self.mask_qc[mask]  

    def supress_nan(self):
        mask = ~np.isnan(self.x)&~np.isnan(self.y)&~np.isnan(self.yerr)&~np.isnan(self.xerr)
        if sum(~mask)==len(mask):
            self.replace_nan()
        else:
            self.mask_not_nan = mask
            self.x = self.x[mask]
            self.y = self.y[mask]
            self.xerr = self.xerr[mask]
            self.yerr = self.yerr[mask]
            self.mask_qc = self.mask_qc[mask]

    def replace_nan(self,value=None):
        if value is None:
            self.y[np.isnan(self.y)] = np.random.randn(sum(np.isnan(self.y)))
            self.x[np.isnan(self.x)] = np.random.randn(sum(np.isnan(self.x)))
            self.yerr[np.isnan(self.yerr)] = np.random.randn(sum(np.isnan(self.yerr)))
            self.xerr[np.isnan(self.xerr)] = np.random.randn(sum(np.isnan(self.xerr)))
        else:
            self.y[np.isnan(self.y)] = value
            self.x[np.isnan(self.x)] = value

    def plot(self, Show=False, color='k', label='', ls='', lw=2, offset=0, mask=None, capsize=0, fmt='o', markersize=6, zorder=1, species=None, alpha=1, modulo=None, modulo_norm=False, cmap=None, new=False, phase_mod=0, shift_mod=0, periodic=False, frac=1, yerr=True, xerr=True, sp=None, highlight_seasons=False, cmin=None, cmax=None, transit_table=None):
        
        '''For the mask give either the first and last index in a list [a,b] or the mask boolean'''
        
        mask_qc = self.mask_qc

        if modulo==100000: #default value in YARARA
            modulo=None
        
        if transit_table is not None:
            modulo=1 #
        
        if (modulo is not None)&(cmap is None):
            try:
                cmap = {'k':'viridis','b':'Blues','r':'Reds','g':'Greens'}[color]
            except:
                cmap = 'viridis'
        if (len(self.x)>25000)&(ls=='')&(modulo is None):
            ls='-'
        
        if species is None:
            species = np.ones(len(self.x))

        if highlight_seasons:
            self.split_seasons(min_gap=highlight_seasons,Plot=False)
            species = self.seasons_species

        if len(np.unique(species))==1:
            colors_species = [color]
        else:
            colors_species = ['k']+['C%.0f'%(i) for i in range(1,1+len(np.unique(species)))]

        for num, selection in enumerate(np.unique(species)):
            
            color = colors_species[num]
            
            if len(np.unique(species))>1:
                label = selection

            if mask is None:
                mask2 = np.ones(len(self.x)).astype('bool')
            elif type(mask[0])==int:
                mask2 = np.zeros(len(self.x)).astype('bool')
                mask2[mask[0]:mask[1]]=True
            else:
                mask2 = mask

            loc = np.where(species[mask2]==selection)[0]
            
            sel = np.arange(len(loc))
            if frac!=1:
                sel = np.random.choice(np.arange(len(loc)),size=int(frac*len(loc)),replace=False)

            self.debug = mask_qc,mask2,loc,sel

            qc = mask_qc[mask2][loc][sel]

            if new:
                plt.figure()
                
            if sp is not None:
                plt.subplot(sp)
            
            if ls!='':
                plt.plot(self.x[mask2][loc][sel],self.y[mask2][loc][sel]+offset,ls=ls,lw=lw,zorder=zorder,label=label,color=color,alpha=alpha)
            else:
                plt.errorbar(self.x[mask2][loc][sel][qc], self.y[mask2][loc][sel][qc]+offset, xerr=self.xerr[mask2][loc][sel][qc]*int(xerr), yerr=self.yerr[mask2][loc][sel][qc]*int(yerr), fmt=fmt, color=color, alpha=alpha, capsize=capsize, label=label, markersize=markersize,zorder=zorder)
                plt.errorbar(self.x[mask2][loc][sel][~qc], self.y[mask2][loc][sel][~qc]+offset, xerr=self.xerr[mask2][loc][sel][~qc]*int(xerr), yerr=self.yerr[mask2][loc][sel][~qc]*int(yerr), fmt='x', color=color, alpha=alpha, capsize=capsize, markersize=markersize,zorder=zorder)
        if Show==True:
            if label!='':
                plt.legend()
            plt.show()


    def fit_line(self, perm=1000, Draw=False, color='k', info=False, fontsize=13, label=True, compute_r=True, offset=True, recenter=True, info_printed=['r','s','i','rms'],loc_legend=0,ls='-.',s_end_point=0):
        k = perm
        self.yerr[self.yerr==0] = [np.min(self.yerr),0.1][np.min(self.yerr)==0] #to avoid 0 value
        
        w = 1/self.yerr**2    
        if offset:    
            A = np.array([(self.x-np.mean(self.x)*int(recenter)),np.ones(len(self.x))]).T
        else:
            A = np.array([self.x]).T

        A = A *np.sqrt(w)[:,np.newaxis]
        B = np.array([self.y]*(k+1)).T
        noise = np.random.randn(np.shape(B)[0],np.shape(B)[1])/np.sqrt(w)[:,np.newaxis] ; noise[:,0] = 0
        B = B + noise
        Bmean = np.sum(B*w[:,np.newaxis],axis=0)/np.sum(w)*int(recenter)
        Brms = np.sqrt(np.sum(((B-Bmean)**2*w[:,np.newaxis]),axis=0)/np.sum(w))
        B = B*np.sqrt(w)[:,np.newaxis]
        Cmean = np.sum(self.x*w,axis=0)/np.sum(w)*int(recenter)
        Crms = np.sqrt(np.sum(((self.x-Cmean)**2*w),axis=0)/np.sum(w))

        self.s = np.linalg.lstsq(A,B,rcond=None)[0][0]
        if offset:
            self.i = np.linalg.lstsq(A,B,rcond=None)[0][1]      
        else:
            self.i = self.s*0

        self.lin_slope_w = np.mean(self.s)
        self.lin_errslope_w = np.std(self.s)
        
        self.lin_intercept_w = np.mean(self.i)
        self.lin_errintercept_w = np.std(self.i)

        self.stats['lin_slope_w'] = self.lin_slope_w
        self.stats['lin_slope_w_std'] = self.lin_errslope_w
        self.stats['lin_intercept_w'] = self.lin_intercept_w
        self.stats['lin_intercept_w_std'] = self.lin_errintercept_w     

        if compute_r:
            self.r = self.s*Crms/Brms        
            self.r_pearson_w = np.mean(self.r)
            self.r_errpearson_w = np.std(self.r)
        else:
            self.r_pearson_w = np.inf
            self.r_errpearson_w = np.inf
            
        self.stats['r_pearson_w'] = self.r_pearson_w
        self.stats['r_pearson_w_std'] = self.r_errpearson_w
        
        
        temp = tableXY(self.x, self.y-((self.x-np.mean(self.x)*int(recenter))*self.lin_slope_w+self.lin_intercept_w), self.yerr)
        self.vec_res = temp

    def fit_GND(self,guess=None,Plot=True,color='r',free_offset=True, beta_fixed=0, norm=True, mini=[None,None,1,0,1.8],maxi=[0,None,None,None,5]):
        """guess = [amp,cen,width,offset,beta]"""

        if guess is None:
            self.find_min()
            loc_min = np.argmin(self.y_min)
            guess_center = self.x_min[loc_min]
            guess_offset = np.nanpercentile(self.y,75)
            guess_amp = self.y_min[loc_min] - guess_offset
            guess_width = (np.max(self.x)-np.min(self.x))/10
            guess_beta = 2
            guess = [guess_amp,guess_center,guess_width,guess_offset,guess_beta]

        if beta_fixed!=0:
            guess[-1] = beta_fixed

        gmodel = Model(myf.GND)
        fit_params = Parameters()
        if norm:
            mini[0] = -2
            maxi[3] = 2

        fit_params.add('amp', value=guess[0], min=mini[0], max=maxi[0])
        fit_params.add('cen', value=guess[1], min=mini[1], max=maxi[1])
        fit_params.add('wid', value=guess[2], min=mini[2], max=maxi[2])
        fit_params.add('offset', value=guess[3], min=mini[3], max=maxi[3])
        fit_params.add('beta', value=guess[4], min=mini[4], max=maxi[4])

        if not free_offset:
            fit_params['offset'].vary = False

        if beta_fixed!=0:
            fit_params['beta'].vary = False

        result1 = gmodel.fit(self.y, fit_params, 1/self.yerr**2, x=self.x)
        self.lmfit = result1
        self.params = result1.params
        self.model_gnd = gmodel.eval(result1.params, x=self.x)
        self.res = self.y - self.model_gnd

        if Plot:
            newx = np.linspace(np.min(self.x),np.max(self.x),10*len(self.x))
            plt.plot(newx,gmodel.eval(result1.params, x=newx),color=color)

    def fit_poly(self, Draw = False, d = 2, color='r',cov=True):
        if np.sum(self.yerr)!=0:
            weights=self.yerr
        else :
            weights =np.ones(len(self.x))
        if cov:
            coeff, V = np.polyfit(self.x, self.y, d, w=1/weights,cov=cov)
            self.cov = V
            self.err = np.sqrt(np.diag(V))
        else:
            coeff= np.polyfit(self.x, self.y, d, w=1/weights,cov=cov)
        self.poly_coefficient = coeff
        self.vec_fitted = np.polyval(coeff, self.x)
        self.chi2 = np.sum((self.y-np.polyval(coeff,self.x))**2)/np.sum(self.yerr**2)
        self.bic = self.chi2+(d+1)*np.log(len(self.x))
        if Draw==True:
            new_x = np.linspace(self.x.min(),self.x.max(),10000)
            plt.plot(new_x, np.polyval(coeff, new_x), linestyle='-.', color=color, linewidth=1)

    def fit_gaussian(self,guess=None,Plot=True,color='r', norm=True, mask=None, mini=[None,None,1,0],maxi=[0,None,None,None],free_offset=True,free_center=True,free_width=True):
        """guess = [amp,cen,width,offset]"""
        if guess is None:
            if True:
                self.find_min()
                loc_min = np.argmin(self.y_min)
                guess_center = self.x_min[loc_min]
                guess_offset = np.nanpercentile(self.y,75)
                guess_amp = self.y_min[loc_min] - guess_offset
                guess_width = (np.max(self.x)-np.min(self.x))/10
                guess = [guess_amp,guess_center,guess_width,guess_offset]
                #print(' [INFO] Automatic CCF guess : ',guess)
            else:
                guess = [-0.5,0,3,1]

        gmodel = Model(myf.gaussian)
        fit_params = Parameters()
        if norm:
            mini[0] = -1
            maxi[3] = 2

        fit_params.add('amp', value=guess[0], min=mini[0], max=maxi[0])
        fit_params.add('cen', value=guess[1], min=mini[1], max=maxi[1])
        fit_params.add('wid', value=guess[2], min=mini[2], max=maxi[2])
        fit_params.add('offset', value=guess[3], min=mini[3], max=maxi[3])

        if not free_offset:
            fit_params['offset'].vary = False
        if not free_center:
            fit_params['cen'].vary = False
        if not free_width:
            fit_params['wid'].vary = False

        if mask is None:
            mask = np.ones(len(self.x)).astype('bool')
        result1 = gmodel.fit(self.y[mask], fit_params, 1/self.yerr[mask]**2, x=self.x[mask])
        self.lmfit = result1
        self.params = result1.params
        self.model_gaussian = gmodel.eval(result1.params, x=self.x)
        self.res = self.y - self.model_gaussian

        if Plot:
            newx = np.linspace(np.min(self.x),np.max(self.x),10*len(self.x))
            plt.plot(newx,gmodel.eval(result1.params, x=newx),color=color)


    def fit_rassine(self, par_R, par_Rmax, par_stretching, tag=''):
        if tag!='':
            tag = '_'+tag
        df = pd.DataFrame({'wave':self.x,'flux':self.y,'flux_err':self.yerr})
        df.to_csv(cwd+'/temp/spectrum_to_normalise%s.csv'%(tag))
        os.system('python Rassine.py -s %s -r %.1f -R %.1f -p %.2f -a 0 -F 6.00'%(cwd+'/temp/spectrum_to_normalise%s.csv'%(tag),par_R,par_Rmax,par_stretching))
        os.system('rm '+cwd+'/temp/spectrum_to_normalise%s.csv'%(tag))
        rassine_file = pd.read_pickle(cwd+'/temp/RASSINE_spectrum_to_normalise%s.p'%(tag))
        self.rassine_continuum = tableXY(rassine_file['wave'],rassine_file['output']['continuum_linear'])
        self.rassine_output = rassine_file['output']
        os.system('rm '+cwd+'/temp/RASSINE_spectrum_to_normalise%s.p'%(tag))


    def ccf(self, mask2, rv_sys=0, rv_range=15, weighted=True, ccf_oversampling=1, wave_min=None, wave_max=None, norm=True, Plot=True, pow_weight=2, fit_gaussian=True, return_mask=False):
        
        mask = mask2.copy()
        if len(np.shape(mask))<2:
            mask = np.hstack([mask[:,np.newaxis],np.ones(len(mask))[:,np.newaxis]])
        
        grid = self.x.copy()
        flux = self.y[:,np.newaxis].T.copy()
        flux_err = self.yerr[:,np.newaxis].T.copy()

        if rv_sys:
            mask[:,0] = myf.doppler_r(mask[:,0],rv_sys)[0]
        
        mask_shifted = myf.doppler_r(mask[:,0],(rv_range+5)*1000)

        mask = mask[(myf.doppler_r(mask[:,0],30000)[0]<grid.max())&(myf.doppler_r(mask[:,0],30000)[1]>grid.min()),:] #supres line farther than 30kms
        if wave_min is not None:
            mask = mask[mask[:,0]>wave_min,:] 
        if wave_max is not None:
            mask = mask[mask[:,0]<wave_max,:] 

        if not len(mask):
            return None
        else:
            mask_min = np.min(mask[:,0])
            mask_max = np.max(mask[:,0])
            
            grid_min = int(myf.find_nearest(grid,myf.doppler_r(mask_min,-100000)[0])[0])
            grid_max = int(myf.find_nearest(grid,myf.doppler_r(mask_max,100000)[0])[0])
            grid = grid[grid_min:grid_max]

            log_grid = np.linspace(np.log10(grid).min(),np.log10(grid).max(),len(grid))
            dgrid = log_grid[1] - log_grid[0]
            #dv = (10**(dgrid)-1)*299.792e6  
            
            used_region = ((10**log_grid)>=mask_shifted[1][:,np.newaxis])&((10**log_grid)<=mask_shifted[0][:,np.newaxis])
            used_region = (np.sum(used_region,axis=0)!=0).astype('bool')
            print('\n [INFO] Percentage of the spectrum used : %.1f [%%] \n'%(100*sum(used_region)/len(grid)))
            
            mask_wave = np.log10(mask[:,0])
            mask_contrast = mask[:,1]*weighted + (1-weighted)
                    
            log_grid_mask = np.arange(log_grid.min()-10*dgrid,log_grid.max()+10*dgrid+dgrid/10,dgrid/11)
            log_mask = np.zeros(len(log_grid_mask))
            
            match = myf.identify_nearest(mask_wave,log_grid_mask)
            for j in np.arange(-5,6,1):
                log_mask[match+j] = (mask_contrast)**pow_weight        

            all_flux = []
            all_flux.append(interp1d(np.log10(self.x), flux[0], kind='cubic', bounds_error=False, fill_value='extrapolate')(log_grid))
            flux = np.array(all_flux)

            all_flux_err = []
            all_flux_err.append(interp1d(np.log10(self.x), flux_err[0], kind='linear', bounds_error=False, fill_value='extrapolate')(log_grid))
            flux_err = np.array(all_flux_err)

            log_template = interp1d(log_grid_mask, log_mask, kind='linear', bounds_error=False, fill_value='extrapolate')(log_grid)
            
            vrad, ccf_power, ccf_power_std = myf.ccf(log_grid[used_region], flux[:,used_region], log_template[used_region], 
                                                    rv_range = rv_range, oversampling = ccf_oversampling, spec1_std = flux_err[:,used_region]) #to compute on all the ccf simultaneously

            self.ccf_profile = tableXY(vrad,np.ravel(ccf_power))
            if norm:
                self.ccf_profile.yerr/=np.max(self.ccf_profile.y)
                self.ccf_profile.y/=np.max(self.ccf_profile.y)
            
            ccf_profile = self.ccf_profile
            
            if Plot:
                plt.figure(figsize=(18,6))
                plt.axes([0.05,0.1,0.58,0.75])
                plt.plot(self.x,self.y,color='k')
                plt.xlim(np.min(mask[:,0])-5,np.max(mask[:,0])+5)
                if norm:
                    plt.axhline(y=1,ls=':',color='k')
                    plt.ylim(0,2.1)
                for j,w in zip(mask[:,0],mask[:,1]/np.max(mask[:,1])):
                    plt.axvline(x=j,color='b',alpha=w*0.5)
                plt.axes([0.68,0.1,0.3,0.75])
                plt.scatter(ccf_profile.x,ccf_profile.y,color='k',marker='o')
                plt.axvline(x=0,ls=':',color='k')
                if norm:
                    plt.axhline(y=1,ls=':',color='k')
                    plt.ylim(0,1.1)
            
            if fit_gaussian:
                maxi = np.percentile(ccf_profile.y,95)
                amp = maxi - np.percentile(ccf_profile.y,5) 
                xmin = np.argmin(ccf_profile.y)
                if (xmin==0)|((xmin+1)==len(ccf_profile.y)):
                    xmin = int(len(ccf_profile.y)/2)
                x1 = myf.find_nearest(ccf_profile.y[0:xmin],maxi-amp/2)[0][0]
                x2 = myf.find_nearest(ccf_profile.y[xmin:],maxi-amp/2)[0][0]
                width = ccf_profile.x[xmin:][x2]-ccf_profile.x[0:xmin][x1]
                center = ccf_profile.x[xmin]
                
                ccf_profile.fit_gaussian(guess=[-amp,center,width,maxi],Plot=Plot,norm=norm)
                try:
                    ccf_profile.fit_GND(guess=[-amp,center,width,maxi,2],color='g',beta_fixed=0,Plot=Plot,norm=norm)
                    self.ccf_params = ccf_profile.params
                    self.params_beta = ccf_profile.params['beta'].value
                    print(' [INFO] Using GND profile for the fit')
                except:
                    ccf_profile.fit_gaussian(guess=[-amp,center,width,maxi],Plot=Plot,norm=norm)
                    self.ccf_params = ccf_profile.params
                    self.params_beta = 2.0
                    print(' [INFO] Using Gaussian profile (GND=2) for the fit')
                self.warning_multipeak = 0
                try:
                    res = tableXY(ccf_profile.x,ccf_profile.res)
                    res.find_min(sort=True)
                    contrast2 = -res.y_min[0] 
                    contrast1 = -ccf_profile.params['amp'].value
                    if (contrast2>0.05)&(contrast2<contrast1)&(abs(res.x_min[0]/1000)<40):
                        res.plot(color='gray',ls='-',offset=1.025)
                        plt.axvline(x=res.x_min[0],color='r',ls='-.')
                        self.warning_multipeak = 1
                        print(' [WARNING] Multi peak detected!')
                except:
                    pass

                if Plot:
                    if ccf_profile.params['wid'].stderr is not None:
                        width_err = ccf_profile.params['wid'].stderr/1000*2.355
                    else:
                        width_err = 0.0
                    if ccf_profile.params['cen'].stderr is not None:
                        ct_err = ccf_profile.params['cen'].stderr
                    else:
                        ct_err = 0.0
                    if ccf_profile.params['amp'].stderr is not None:
                        amp_err = 100*(ccf_profile.params['amp'].stderr/ccf_profile.params['offset'].value)
                    else:
                        amp_err = 0.0

                    plt.axvline(x=ccf_profile.params['cen'].value,color='r',alpha=0.3)
                    plt.title('Beta = %.2f \n C = %.2f +/- %.2f [%%] \n FWHM = %.2f +/- %.2f [km/s] \n RV = %.2f +/- %.2f [m/s]'%(
                        self.params_beta,
                        100*(-ccf_profile.params['amp'].value/ccf_profile.params['offset'].value),
                        amp_err,
                        ccf_profile.params['wid'].value/1000*2.355,
                        width_err,
                        ccf_profile.params['cen'].value+rv_sys,
                        ct_err))
            
            if return_mask:
                return log_grid, log_template, used_region
