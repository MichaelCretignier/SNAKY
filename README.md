# SNAKY (a Spectroscopic Novel Analysis Kit of Yᴀʀᴀʀᴀ) v1.2.1

<p align="center">
  <img src="logo.png" alt="Project logo" width="400">
</p>

SNAKY is a fast execution code aiming to determine in a complete way an observed star from high resolution spectra in the visible (R>50'000).

The main outputs are: 

1) The master spectrum in stellar rest-frame
2) The stellar atmospheric parameters
3) The MHK activity index time-series
4) The stellar vsini (in validation...) + Stellar inclination

SNAKY is **NOT** an RV pipeline aiming at EPRV precision like YARARA.\
SNAKY is optimized for speed efficiency and star characterization, meaning steps may alter the EPRV precision.\
SNAKY includes a packaged version of [RASSINE](https://github.com/MichaelCretignier/Rassine_public)

## ① Release Date (Soon...!)

Please cite the relevant works:

- **RASSINE** — [Cretignier et al. 2020b](https://ui.adsabs.harvard.edu/abs/2021A%26A...653A..43C/abstract)

- **Atmospheric parameters** — [Cretignier et al. 2024b](https://ui.adsabs.harvard.edu/abs/2024MNRAS.535.2562C/abstract)

- **MHK activity index** — [Cretignier et al. 2024a](https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.2940C/abstract) + [Cretignier et al. 2024b](https://ui.adsabs.harvard.edu/abs/2024MNRAS.535.2562C/abstract)

- **VSINI** — Cretignier et al. (in prep.)

## ② Contact Me

If you have any problem, please contact me at: michael.cretignier@physics.ox.ac.uk

## ③ Installation

*Git Clone / Download this GitHub repository on your own machine and move in the directory.*

```bash
cd .../GitHub/SNAKY/
```

*You may use your existing Python environment, as SNAKY depends on only a few libraries (see the `requirements` files for details).   However, it is **strongly recommended** to use `scikit-learn==1.7.2` to ensure full compatibility.*

*Based on [benchmark](#flag3), the fastest version is the python 3.10.15. \
Other tested `../SNAKY/requirements/requirement_PYTHON_VERSION.txt` libraries versioning are available if needed. \
For **Mac Intel Chip**, python version 3.8.8 is recommended.*

### [Option 1] Conda install (Python 3.10.15)

```bash
conda create -n snaky python=3.10.15
conda activate snaky
pip install -r requirements_3.10.15.txt
```

*Check if the `snaky` environment exists and is active:*

```bash
conda env list
```

### [Option 2] Venv install (Python 3.10.15)

```bash
python3 -m venv snaky 
source snaky/bin/activate 
pip install --upgrade pip 
pip install -r requirements_3.10.15.txt
```

## ④ Build the code

*SNAKY contains a table too heavy for GitHub that need to be merged using the build script `snaky_build.py`*

```bash
cd ../SNAKY/src_snaky/
python snaky_build.py
```

*You should now have a `../SNAKY/Material_snaky/template_star_SNAKY_3900_6800.npy` file.*

## ⑤ Tutorial

*The code is close to being fully packaged.  
For now, if you want to run `snaky` from anywhere on your machine (without launching it from inside the `SNAKY/` directory), you can manually add `SNAKY/` to your `sys.path` in your Python scripts:*


```python
import sys
sys.path.append('.../GitHub/SNAKY/')

import src_snaky.run as snaky

...

```

### Step-by-step

*If you haven't add `SNAKY/` to your sys.path, first enter into your local git clone `SNAKY/` directory:*

```bash
cd .../GitHub/SNAKY/
```

*Launch an IPython shell:*

```bash
ipython
```

*SNAKY considers a dataset as **a collection of spectra for a given star and given instrument**. To run SNAKY on your spectra, you just need to specify:*

1) An output directory 

```python
output_dir = '/Users/cretignier/Desktop/Snaky/'
```

2) A list of spectra you want to process
3) The starname and the instrument

*This information is specified by:*

```python
import src_snaky.run as snaky

# let's use the SNAKY test dataset
files = snaky.glob.glob(snaky.myv.TEST_DATASET1)
#output_dir = '/Users/cretignier/Desktop/Snaky/'

job = snaky.start()
job.set_output_dir(output_dir)                     # define output dir 
job.set_dataset('HD12345', 'HARPS03_3.5', files)   # define the star + instrument + list of spectra
```

*We then initialize SNAKY, which creates the directory tree and normalizes the spectra using RASSINE:*


```python
# initialization
job.init_workspace() # 1) Create tree directories
job.preprocess()     # 1) Run RASSINE to normalise spectra
job.set_summary()    # 2) Create the summary table
job.check_spectra()  # 2) Quality flag control on the spectra
```

*Now the data preprocessed, we can finally launch the SNAKY pipeline:*

```python
# pipeline
job.compute_rv_sys()        # 3)  RV_sys + FWHM
job.compute_ccf()           # 4)  Compute CCF with (G2, Garfield, Kitty)
job.compute_master()        # 5)  Create a master spectrum
job.compute_atmos()         # 6)  Compute atmospheric parameters
job.compute_resolution()    # 7)  Compute instrumental resolution
job.compute_vsini()         # 8)  Compute stellar vsini
job.compute_abs_continuum() # 9)  Correct the continuum in UV
job.compute_activity()      # 10) Compute activity proxies
job.compute_mhk()           # 11) Compute MHK 
job.compute_spectroscopy()  # 12) Create the stellar spectrum in SRF
job.compute_mag_cycle()     # 13) FINCH magnetic cycle
job.cleaning()              # 14) Remove useless products
```
*You can check the figures created in the output directory `...IMAGES/`* \
*For now, let's erase our work with the `.reset()` method. For security, it is required to enter the command line twice in order to launch it:*

**WARNING: Do not add new files by yourself inside the created `output_dir`. This directory is a dedicated SNAKY workspace and should contain only files generated by SNAKY.**

```python

job.reset(suppression='all')
job.reset(suppression='all')

#job.reset()                 # remove all products except RASSINE normalised spectra
#job.reset(supression='all') # remove everything

```

### Simplified Call (.reduce)

*Let's start again, all the previous lines can be called shortly using the `.reduce()` method:* 

<a id="flag1"></a>

```python
import src_snaky.run as snaky

files = snaky.glob.glob(snaky.myv.TEST_DATASET1)
#output_dir = '/Users/cretignier/Desktop/Snaky/'

job = snaky.start()
job.set_output_dir(output_dir)
job.set_dataset('HD123456','HARPS03_3.5',files)  #HARPS03 = before fibre upgrade in 2015

job.reduce(begin=1, end=14) # check the sequence number with: job.reduce?
```

As you may have noticed, the `.reduce()` method also monitors RAM usage and execution time, and automatically saves this information in `REDUCTION_INFO/` for [benchmark](#flag3) purposes.

You may also have seen the yellow list printed at the start — this is the **trigger**. It indicates which SNAKY steps have already been completed and which are still pending.


### Automatic Restart from the Last Completed Step

Thanks to this trigger mechanism, `.reduce()` can automatically restart the pipeline from the last successfully completed step after a crash:

```python
# Simulate a crash

import src_snaky.run as snaky

files = snaky.glob.glob(snaky.myv.TEST_DATASET1)
#output_dir = '/Users/cretignier/Desktop/Snaky/'

job = snaky.start()
job.set_output_dir(output_dir)
job.set_dataset('HD22049', 'HARPS03_3.5', files)  # the real stellar name

# Stop at step 6 (crash before atmospheric parameters, step 7)
job.reduce(begin=1, end=6)

# Restart from the beginning
# automatic_db=True skips completed steps
job.reduce(begin=1, end=14, automatic_db=True)

# automatic_db is False by default in order to relaunch steps
# Example: recompute inclination with user specified Prot and Rs
job.set_star(prot=11,rs=0.74)
job.reduce(begin=8, end=8) # Eps.Eridani inclination ~26° !
```

## ⑥ Launching a RASSINE dataset

*SNAKY itself is very fast and scales as O(N). However, within the reduction pipeline, RASSINE is the most time-consuming step (about 20–30% of the total execution time in the previous single-spectrum example). RASSINE typically requires ~15 seconds per spectrum and can quickly dominate the total runtime, exceeding the SNAKY processing time by orders of magnitude when many (N>50) spectra are processed:*

$$
\text{Total Execution Time} =
\underbrace{15 N}_{\text{RASSINE}} 
+ 
\underbrace{
60 +  0.38 \times N}_{\text{SNAKY}}
\quad [\mathrm{s}]
$$

*But, maybe you already have RASSINE spectra saved on your computer? (If not yet, soon you will!).*

*To read them with SNAKY, you just have to specify manually with `.set_star()`the stellar coordinates (that are missing from the RASSINE metadata):*

<a id="flag2"></a>

```python
# Let's use the Alpha Cen B RASSINE dataset
import src_snaky.run as snaky

files = snaky.glob.glob(snaky.myv.TEST_DATASET2)
#output_dir = '/Users/cretignier/Desktop/Snaky/'

job = snaky.start()
job.set_output_dir(output_dir)
job.set_dataset('HD128621','HARPS15_3.3.6',files) #HARPS15 = after fibre upgrade in 2015

job.set_star(ra=219.90, dec=-60.84, prot=36) # ra and dec in degrees (prot optional)
job.reduce(begin=1, end=14,  copy_files=True) 
```

*The `copy_files=True` option copies the RASSINE files locally into the SNAKY output directory, as if they had been generated by the pipeline. This is not required, but `.p` pickle files can occasionally become corrupted, and copying them provides a safer workflow. This doubles the storage requirement and final decision is then left to the user.*

*Since RASSINE is the slowest step, it's recommanded to save the produced `RASSINE_*.p` normalised spectra somewhere on your local machine in order to send them as input of SNAKY rather than the usual `.fits` files.*

## ⑦ Large-Scale Processing (SLURM / sbatch parallelization)

*SNAKY is designed to process easily and rapidly thousands of datasets (a dataset corresponds to a star + instrument combination). For large runs, the recommended approach is to use `sbatch`.* \
*This is possible by using the `run_snaky_med.s` SLURM script, that calls the `snaky_trigger.py` Python script.*

```bash
sbatch run_snaky_med.s HD128621 HARPS15_3.3.6 1 14
```

## ⑧ Your favourite instrument missing?

<a id="flag4"></a>

SNAKY can process spectra from the following products from the following spectrographs:

| SPECTRO | DRS        | PRODUCT        |  SNAKY_CODE        | 
|---------------|---------------|---------------|---------------|
| CORALIE98 | irrelevant | S1D | CORALIE98_3.3 |
| CORALIE07 | irrelevant | S1D | CORALIE07_3.4 |
| CORALIE14 | irrelevant | S1D | CORALIE14_3.8 |
| [SOPHIE](http://atlas.obs-hp.fr/sophie/) | irrelevant | S1D | SOPHIE_0.5 |
| [SOPHIE-HE](http://atlas.obs-hp.fr/sophie/) (high efficiency) | irrelevant | S1D | SOPHIE-HE_0.5 |
| [HARPN](http://archives.ia2.inaf.it/tng/) | irrelevant | S1D |  HARPN_3.0.1 |
| [ESPRESSO](https://archive.eso.org/scienceportal/home) | irrelevant | S1D |  ESPRESSO_3.3.6 |
| [NEID](https://neid.ipac.caltech.edu) | irrelevant | E2DS |  NEID_1.0 |
| [NEID-HE](https://neid.ipac.caltech.edu) (high efficiency) | irrelevant | ED2S |  NEID-HE_1.0 |
| [HARPS03](https://archive.eso.org/scienceportal/home) | 3.5 (old) | S1D |  HARPS03_3.5 |
| [HARPS15](https://archive.eso.org/scienceportal/home) | 3.5 (old) | S1D |  HARPS15_3.5 |
| [HARPS03](https://dace.unige.ch) | 3.3.6 (new) | S1D |  HARPS03_3.3.6 |
| [HARPS15](https://dace.unige.ch) | 3.3.6 (new) | S1D |  HARPS15_3.3.6 |

HARPS spectra before and after the fiber upgrade (2015-05-23) have to be processed independently as respectively HARPS03 and HARPS15 spectra and the DRS version has to be correctly specified.

To add a new instrument you only need 5 information from its header:
1) jdb   [!mandatory!]
2) alpha [!mandatory!]
3) dec   [!mandatory!]
4) berv  [optional]
5) snr   [optional]

Then create your own function `read_espresso()` in `snaky_main.py`
and modify the `extract_header()` function too. 

If only e2ds spectra exist and not s1d, follow the `read_neid()` example.
NB: s1d spectra should always be preferred over e2ds/s2d spectra

## ⑨ BENCHMARK (Computation time)

<a id="flag3"></a>

*You can test your installation and speed with the following benchmark command-lines:*

*Benchmark Dataset 1 (N=1):*

```python
import src_snaky.run as snaky

#output_dir = '/Users/cretignier/Desktop/Snaky/'
snaky.benchmark1(output_dir) #check "[INFO] Processing achieved in ..."
```

*Benchmark Dataset 2 (N=20,RASSINE files already exist):*

```python
import src_snaky.run as snaky

#output_dir = '/Users/cretignier/Desktop/Snaky/'
snaky.benchmark2(output_dir) #check "[INFO] Processing achieved in ..."
```

| Computer | Processor        | VERSION        | LIBRARIES       | DATASET1    | DATASET2    |
|---------------|---------------|---------------|---------------|--------------|--------------|
| MC1 | Apple M4 MAX (2024) | SNAKY (1.2.1) | 3.10.15 | 00 min 54 s  | 00 min 54 s  |
| MC1 | Apple M4 MAX (2024) | SNAKY (1.2.1) | 3.12.5 | 01 min 45 s  | 01 min 30 s  |
| SA1 | Apple M2 (2022) | SNAKY (1.1.1) | 3.10.15 | 01 min 13 s  | 01 min 28 s  |
| MC2 | Intel Mac (2018) | SNAKY (1.2.1) | 3.8.8 | 03 min 45 s  | 06 min 47 s  |
| DF1 | Intel i5 12600k (2021) | SNAKY (1.0.7) | 3.12.5 | 03 min 25 s  | 03 min 28 s  |
| CC1 | Apple M4 PRO (2024) | SNAKY (1.0.1) | 3.10.15 | 01 min 30 s  | 01 min 05 s  |
| ? | Yours! ☺ | SNAKY (1.0.1) | ??? | ???  | ???  |

### RAM requirement

*The RAM requirement for SNAKY (without RASSINE preprocessing) scales as O(N) for HARPS spectra (~300k wavelength bins):*

$$
\text{Total RAM} =
4.2 + 0.72 \times \left( \frac{N}{100} \right) \quad [GB]
$$

## Uninstall

*You can delete the snaky Python environment with:*

```bash
conda remove --name snaky --all
```

