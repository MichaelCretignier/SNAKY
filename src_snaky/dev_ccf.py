import pandas as pd
import numpy as np


file = pd.read_pickle('/Users/cretignier/Desktop/Snaky/compress_snaky/74.p')

wave         = file['wave']
spec1        = file['spec1']
spec2        = file['spec2']
extended     = file['extended']
rv_range     = file['rv_range']
oversampling = file['oversampling']
velocity     = file['velocity']
conv         = file['conv']

for j in range(4):
    spec1 = np.vstack([spec1,spec1])

rv_range = 20
oversampling = 3

def ccf(wave, spec1, spec2, extended=1500, rv_range=45, oversampling=3, spec1_std=None, method='cubic'):
    "CCF for a equidistant grid in log wavelength spec1 = spectrum, spec2 =  binary mask"   

    dwave = np.median(np.diff(wave))

    if len(np.shape(spec1))==1:
        spec1 = spec1[:,np.newaxis].T
    #spec1 = np.hstack([np.ones(extended),spec1,np.ones(extended)])
    
    rv_max = int(np.log10((rv_range/299.792e3)+1)/dwave)
    rv_shift = np.arange(-rv_max,rv_max+1,1)

    spec1 = np.hstack([np.ones((len(spec1),extended)),spec1,np.ones((len(spec1),extended))])
    spec2 = np.hstack([np.zeros(extended),spec2,np.zeros(extended)])
    wave = np.hstack([np.arange(-extended*dwave+wave.min(),wave.min(),dwave),wave,np.arange(wave.max()+dwave,(extended+1)*dwave+wave.max(),dwave)])
    #shift = np.linspace(0,dwave,oversampling+1)[:-1] #oversampling is now done on the product
    shift = np.linspace(0,dwave,2)[:-1]
    shift_save = []
    sum_spec = np.nansum(spec2)

    new_spec = np.empty((len(shift), len(wave)), dtype=float)
    for i,j in enumerate(shift):
        spec2_s = interp1d(wave+j,spec2,kind='linear', bounds_error=False, fill_value='extrapolate')(wave)
        new_spec[i] = spec2_s

    spec1[spec1!=spec1] = 0
    new_spec[new_spec!=new_spec] = 0

    convolution = []
    for k in tqdm(rv_shift):
        new_spec2 = np.hstack([new_spec[:,-k:],new_spec[:,:-k]])
        result = spec1 @ new_spec2.T
        result /= sum_spec
        convolution.append(result)
    convolution = np.hstack(np.array(convolution)).T

    shift_save = (shift[None, :] + rv_shift[:, None] * dwave).ravel()
    velocity = (299.792e6*10**shift_save)-299.792e6
    
    conv_std = np.zeros(np.shape(convolution))

    vel_oversamp = np.arange(0,np.max(velocity)+0.001,np.diff(velocity)[0]/oversampling)
    vel_oversamp = np.array(list(-vel_oversamp[1:][::-1])+list(vel_oversamp))
    convolution = np.array([interp1d(velocity,conv,kind=method, bounds_error=False, fill_value='extrapolate')(vel_oversamp) for conv in convolution.T]).T
    velocity = vel_oversamp

    return velocity, convolution, conv_std

vel_ref, conv_ref, dust = ccf(wave, spec1, spec2, extended=1500, rv_range=20, oversampling=3, spec1_std=None)
plt.scatter(vel_ref,conv_ref[:,0])

from collections import namedtuple
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from tqdm import tqdm

CCFReturn = namedtuple("CCFReturn", ["velocity", "convolution", "convolution_errors"])

def ccf_damien(
    wavelengths: NDArray[np.float64],
    spectrums: NDArray[np.float64],
    mask: NDArray[np.bool_],
    extended: int = 1500,
    rv_range: int = 45,
    oversampling: int = 10,
    method='cubic',
) -> CCFReturn:
    "CCF for a equidistant grid in log wavelength spec1 = spectrum, spec2 =  binary mask"

    dwave = np.median(np.diff(wavelengths))

    if len(np.shape(spectrums)) == 1:
        spectrums = spectrums[:, np.newaxis].T
    # spec1 = np.hstack([np.ones(extended),spec1,np.ones(extended)])

    spectrums = pad(spectrums, extended, 1)
    mask = pad(mask, extended, 0)

    min: np.float64 = wavelengths.min().astype(wavelengths.dtype)
    max: np.float64 = wavelengths.max().astype(wavelengths.dtype)
    extended_left = np.linspace(
        min - extended * dwave, min - dwave, extended, dtype=wavelengths.dtype
    )
    extended_right = np.linspace(
        max + dwave, max + extended * dwave, extended, dtype=wavelengths.dtype
    )
    wavelengths = np.hstack(
        [
            extended_left,
            wavelengths,
            extended_right,
        ]
    )

    #shift = np.linspace(0, dwave, oversampling + 1)[:-1]
    shift = np.linspace(0,dwave,1+1)[:-1] #oversample the product now

    new_spec = np.empty((len(shift), len(wavelengths)), dtype=mask.dtype)
    for i, j in enumerate(shift):
        new_spec[i] = np.interp(wavelengths, wavelengths + j, mask)

    spectrums[spectrums != spectrums] = 0
    new_spec[new_spec != new_spec] = 0

    sum_spec = np.nansum(mask)

    rv_shift, convolution = process_convolution(
        spectrums, rv_range, center=dwave, new_spec=new_spec, true_value_amout=sum_spec
    )

    shift_save = (shift[None, :] + rv_shift[:, None] * dwave).ravel()
    velocity = (299.792e6 * 10**shift_save) - 299.792e6

    convolution = np.hstack(convolution).T

    vel_oversamp = np.arange(0,np.max(velocity)+0.001,np.diff(velocity)[0]/oversampling)
    vel_oversamp = np.array(list(-vel_oversamp[1:][::-1])+list(vel_oversamp))
    convolution = np.array([interp1d(velocity,conv,kind=method, bounds_error=False, fill_value='extrapolate')(vel_oversamp) for conv in convolution.T]).T
    velocity = vel_oversamp

    conv_std = np.zeros(np.shape(convolution))

    return CCFReturn(velocity, convolution, conv_std)

DType = TypeVar("DType", bound=np.generic)


def pad(arr: NDArray[DType], amount: int, pad_value: float = 0) -> NDArray[DType]:
    if arr.ndim == 1:
        result = np.full(
            arr.shape[0] + 2 * amount, fill_value=pad_value, dtype=arr.dtype
        )
        result[amount : amount + arr.shape[0]] = arr
    else:
        result = np.full(
            (arr.shape[0], arr.shape[1] + 2 * amount),
            fill_value=pad_value,
            dtype=arr.dtype,
        )
        result[:, amount : amount + arr.shape[1]] = arr
    return result

ConvolutionReturn = namedtuple("ConvolutionReturn", ("pixel_shifts", "convolution"))

def process_convolution(
    spectrums,
    rv_range: int,
    center: np.float64,
    new_spec: NDArray[np.bool_],
    true_value_amout: np.intp,
) -> ConvolutionReturn:
    speed_of_light = 299.792e3
    max_shift = int(np.log10((rv_range / speed_of_light) + 1) / center)
    pixel_shifts = np.arange(-max_shift, max_shift + 1, 1)

    all_shifts = np.array([np.roll(new_spec, k, axis=1) for k in tqdm(pixel_shifts)])
    convolution = (
        spectrums @ all_shifts.transpose(0, 2, 1) / true_value_amout
    ).squeeze()
    return ConvolutionReturn(pixel_shifts, convolution)

vel, conv, dust = ccf_damien(wave, spec1, spec2, extended=1500, rv_range=20, oversampling=3)

def ccf_deprecated(wave, spec1, spec2, extended=1500, rv_range=45, oversampling=10, spec1_std=None):
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
        new_spec = interp1d(wave+j,spec2,kind='linear', bounds_error=False, fill_value='extrapolate')(wave)
        for k in np.arange(-rv_max,rv_max+1,1):
            new_spec2 = np.hstack([new_spec[-k:],new_spec[:-k]])
            convolution.append(np.nansum(new_spec2*spec1,axis=1)/sum_spec)
            convolution_std.append(np.sqrt(np.abs(np.nansum(new_spec2*spec1_std**2,axis=1)))/sum_spec)
            shift_save.append(j+k*dwave)
    shift_save = np.array(shift_save)
    sorting = np.argsort(shift_save)

    velocity = (299.792e6*10**shift_save[sorting])-299.792e6
    conv = np.array(convolution)[sorting]
    conv_std = np.array(convolution_std)[sorting]

    return velocity, conv, conv_std


vel_old, conv_old, dust = ccf_deprecated(wave, spec1, spec2, extended=1500, rv_range=20, oversampling=3, spec1_std=None)


# dwave = np.float64(8.23137128680429e-07)
# new_spec = array([[0., 0., 0., ..., 0., 0., 0.]], shape=(1, 180519))
# shift = array([0.])
# spec1 = array([[1., 1., 1., ..., 1., 1., 1.],
# [1., 1., 1., ..., 1., 1., 1.],
# [1., 1., 1., ..., 1., 1., 1.],
# ...,
# [1., 1., 1., ..., 1., 1., 1.],
# [1., 1., 1., ..., 1., 1., 1.],
# [1., 1., 1., ..., 1., 1., 1.]], shape=(165, 180519))
# sum_spec = np.float64(780.4864834666746)
# wave = np.float64(780.4864834666746)




