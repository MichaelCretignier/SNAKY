# SNAKY v1.7.0

<p align="center">
  <img src="logo.png" alt="Project logo" width="400">
</p>

SNAKY (a Spectroscopic Novel Analysis Kit of Yᴀʀᴀʀᴀ) is a fast execution code aiming to determine in a complete way an observed star from high resolution spectra in the visible (R>50'000). Derived parameters are:

1) Teff temperature
2) Log(g) surface gravity
3) [Fe/H] metallicity
4) Mass (from 1. + 2. + 3.)
5) Radius (from 1. + 2. + 3.)
6) MHK activity Level
7) Vsin(i) projected equatorial velocity
8) Sin(i) inclination (if rotational period user-specified)

The main useful outputs of SNAKY are: 

1) The master spectrum in stellar rest-frame
2) The stellar atmospheric parameters (cited above)
3) The MHK activity index time-series

SNAKY is **NOT** an RV pipeline aiming at EPRV precision like YARARA.\
SNAKY is optimized for speed efficiency and star characterization, meaning steps may alter the EPRV precision.\
SNAKY includes a packaged version of [RASSINE](https://github.com/MichaelCretignier/Rassine_public)

## ① Contact Me

If you have any problem, please contact me at: michael.cretignier@physics.ox.ac.uk

## ② Installation

*Git Clone / Download this GitHub repository on your own machine and move in the directory.*

```bash
cd .../GitHub/SNAKY/
```

*You may use your existing Python environment, as SNAKY depends on only a few libraries (see the `requirements` files for details).   However, it is **strongly recommended** to use `scikit-learn==1.7.2` to ensure full compatibility.*

*Based on [benchmark](#flag4), the fastest version is the python 3.10.15. \
Other tested `../SNAKY/requirements/requirement_PYTHON_VERSION.txt` libraries versioning are available if needed. 

Recommended Python versions:
- **Apple Silicon Macs (M1/M2/M3/M4):** Python **3.10.15**
- **Intel-based Macs:** Python **3.8.8**

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

*You can uninstall the environment at any time if needed (see [Uninstall](#flag_uninstall)).*

### [Option 2] Venv install (Python 3.10.15)

```bash
python3 -m venv snaky
source snaky/bin/activate 
pip install --upgrade pip 
pip install -r requirements_3.10.15.txt
```

## ③ Download the material

*SNAKY contains materials stored in a directory that need to be downloaded on Zenodo `https://doi.org/10.5281/zenodo.20659152`*.

Run the following code that will download the directory:

```bash
python .../GitHub/SNAKY/src_snaky/install.py
```

*You should now have a `../SNAKY/Material_snaky/` directory with files inside.*

<!--
## ④ Build the code

*SNAKY contains a table too heavy for GitHub that need to be merged using the build script `snaky_build.py`*

```bash
cd ../SNAKY/src_snaky/
python snaky_build.py
```

*You should now have a `../SNAKY/Material_snaky/template_star_SNAKY_3900_6800.npy` file.*

-->

## ④ Tutorial

*If you want to run `snaky` from anywhere on your machine (without launching it from inside the `../GitHub/SNAKY/` directory), you can manually add `../GitHub/SNAKY/` to your `sys.path` at the top of your Python scripts:*

```python
import sys
sys.path.append('.../GitHub/SNAKY/')

import src_snaky.run as snaky

...

```

*Otherwise you need to launch the code by moving inside the directory:*

```bash
cd .../GitHub/SNAKY/
```

### • Test your installation

<a id="flag3"></a>

*You can test if your installation is working and monitor your speed performance with the benchmark data sets.*

*Move in the `../GitHub/SNAKY/` and launch an IPython shell:*

```bash
cd .../GitHub/SNAKY/
ipython
```

*To run the two benchmarks, the command-lines are:*


```python
import src_snaky.run as snaky

# Benchmark Dataset 1 (N=1)
snaky.benchmark1() 
```

```python
import src_snaky.run as snaky

# Benchmark Dataset 2 (N=20, RASSINE files already exist)
snaky.benchmark2() 
```

*You should see a green text line "[INFO] The final cleaning of the output products was done." if the processing was successful.*

*All the products should also be visible in `.../GitHub/SNAKY/snaky_data` which is the default output directory.* 

### • Step-by-step 

*Launch an IPython shell:*

```bash
cd .../GitHub/SNAKY/
ipython
```

*To run SNAKY on your spectra, you just need to specify 3 elements:*

1) An output directory 

```python
output_dir = '/Users/cretignier/Desktop/Snaky'
```

The `output_dir` will be the same for all your stars and instruments. SNAKY is dealing to properly creating a tree for each of them. 

2) A list of spectra you want to process `files = [...]`
3) The starname and the instrument

*SNAKY considers a dataset as **a collection of spectra for a given star and given instrument (including the [DRS version](#flag4))**. This information is specified by:*

```python
import src_snaky.run as snaky

# let's use the SNAKY test dataset
files = snaky.glob.glob(snaky.myv.TEST_DATASET1)
#output_dir = '/Users/cretignier/Desktop/Snaky'

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
job.load_data()      # 2) Init the spectrum time-series object
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
*You can check the figures created in the output directory `.../IMAGES/`* \
*For now, let's erase our work with the `.reset()` method. For security, it is required to enter the command line twice in order to launch it:*

**WARNING: Do not add new files by yourself inside the created `output_dir`. This directory is a dedicated SNAKY workspace and should contain only files generated by SNAKY.**

```python

job.reset(suppression='all')
job.reset(suppression='all')

#job.reset()                 # remove all products except RASSINE normalised spectra
#job.reset(supression='all') # remove everything

```

### • Simplified Call (.reduce)

*Let's start again, all the previous lines can be called shortly using the `.reduce()` method:* 

<a id="flag1"></a>

```python
import src_snaky.run as snaky

files = snaky.glob.glob(snaky.myv.TEST_DATASET1)
#output_dir = '/Users/cretignier/Desktop/Snaky'

job = snaky.start()
job.set_output_dir(output_dir)
job.set_dataset('HD123456','HARPS03_3.5',files)  #HARPS03 = before fibre upgrade in 2015

job.reduce(begin=1, end=14) # check the sequence number with: job.reduce?
```

*As you may notice, the `.reduce()` method also monitors RAM usage and execution time, and automatically saves this information in `.../REDUCTION_INFO/` for [benchmark](#flag4) purposes.*

### • Automatic Restart from the Last Completed Step

*SNAKY contains a **trigger** able to assess which steps have already been completed and which are still pending.*

*Thanks to this, `.reduce()` can restart the pipeline from the last successfully completed step after a crash:*

```python
# Simulate a crash at step 7

import src_snaky.run as snaky

files = snaky.glob.glob(snaky.myv.TEST_DATASET1)
#output_dir = '/Users/cretignier/Desktop/Snaky'

job = snaky.start()
job.set_output_dir(output_dir)
job.set_dataset('HD22049', 'HARPS03_3.5', files)  # the real stellar name

# Stop at step 6 (crash before atmospheric parameters, step 7)
job.reduce(begin=1, end=6)
```
*Let's restart the job with the trigger by using the option `automatic_db=True`:*

```python
job.reduce(begin=1, end=14, automatic_db=True) 
# Notice that yellow green list at the start? That's the trigger checking the pending steps!
```

*Note that by default in `.reduce()`, `automatic_db=False` in order to relaunch some specific steps.*
*For instance, you would like to compute the stellar inclination with a user-specified rotational period (prot) and stellar radius (rs):*

```python
job.set_star(prot=11,rs=0.74)
job.reduce(begin=8, end=8) # Eps.Eridani inclination ~26° !
```

*If you process DRS datasets, SNAKY will provide the RASSINE normalised spectra saved in the directory `../WORKSPACE/RASSINE*.p`. Since RASSINE is the slowest step (see next section), it's recommended to save the produced `RASSINE_*.p` files somewhere on your local machine in order to send them as input of SNAKY rather than the usual DRS `.fits` files.*

## ⑤ Launching a RASSINE dataset

*SNAKY itself is very fast and scales as O(N). However, within the reduction pipeline, RASSINE is the most time-consuming step (about 20–30% of the total execution time in the previous single-spectrum example). RASSINE typically requires ~15 seconds per spectrum and can quickly dominate the total runtime, exceeding the SNAKY processing time by orders of magnitude when many (N>50) spectra are processed:*

$$
\text{Total Execution Time} =
\underbrace{15 \times N}_{\text{RASSINE}} 
+ 
\underbrace{
60 +  0.38 \times N}_{\text{SNAKY}}
\quad [\mathrm{s}]
$$

*But, maybe you already have RASSINE spectra saved on your computer? (You have now!).*

*To read them with SNAKY, you just have to specify manually with `.set_star()`the stellar coordinates (that are missing from the RASSINE metadata):*

<a id="flag2"></a>

```python
# Let's use the Alpha Cen B RASSINE dataset
import src_snaky.run as snaky

files = snaky.glob.glob(snaky.myv.TEST_DATASET2) # a list of RASSINE spectra
#output_dir = '/Users/cretignier/Desktop/Snaky'

job = snaky.start()
job.set_output_dir(output_dir)
job.set_dataset('HD128621','HARPS15_3.3.6',files) #HARPS15 = after fibre upgrade in 2015

job.set_star(ra=219.90, dec=-60.84, prot=36) # ra and dec in degrees (prot optional)
job.reduce(begin=1, end=14) 
```

<!-- T

*The `copy_rassine_files=True` option copies the RASSINE `.p` files locally into the SNAKY output directory, as if they had been generated by the pipeline. This is not required, but `.p` pickle files can occasionally become corrupted, and copying them provides a safer workflow. This doubles the storage requirement and final decision is then left to the user.*

-->

## ⑥ Public ESO + TNG + OHP archives queries and SNAKY processing

<a id="flag_download"></a>

*To even simplify further the processing, SNAKY contains a code `snaky_query.py` that can directly download the spectra on your machine. The shortest call of the function is:*

```bash
python snaky_query.py -s HD217014
```

The list of optional parameters are:

`-s`  List of target stars. A comma-separated list (e.g. `HD10700,HD22049`) or a `.csv` file with `'starname'` column \
`-o`  Output directory (e.g. `/Users/Desktop/SNAKY_WORSPACE`) \
`-n`  Maximum number of spectra to download per instrument (e.g. `-n 5`) \
`-b`  Starting SNAKY reduction stage (e.g. `-b 1`) from `.reduce()` \
`-e`  Ending SNAKY reduction stage (e.g. `-e 14`) from `.reduce()` \
`-a`  Value of the `automatic_db` parameter (`0` or `1`; e.g. `-a 1`) from `.reduce()` \
`-P`  Number of parallelization (e.g. `-P 1`) \
`-p`  Index of the current parallel process, between `1` and `P` (e.g. `-p 1`) \
`-i`  Instrument to SNAKY process (e.g. `-i HARPS`) \
`-v`  Toggle the verbose of the code (e.g `-v 0`) \
`-H`  Toggle the extended Help of the code (e.g `-H 1`)

*Let's process three stars rapidly:*

```bash
python snaky_query.py -s HD217014,HD120411,HD4628,HD22049,HD197481 -n 1
```


## ⑦ Large-Scale Processing (SLURM / sbatch parallelization)

*SNAKY is designed to process easily and rapidly thousands of datasets (as a recall a dataset corresponds to a star + an instrument combination). For large runs, the recommended approach is to use `sbatch`.* \
*This is possible by using the `run_snaky_med.s` SLURM script, that calls the `snaky_trigger.py` Python script.*

```bash
sbatch run_snaky_med.s HD128621 HARPS15_3.3.6 1 14
```

## ⑧ Your favourite instrument missing?

<a id="flag4"></a>

SNAKY can process spectra from the following products from the following spectrographs (high efficiency spectrograph mode should be specified with SPECTRO-HE):

| SPECTRO | DRS        | PRODUCT        |  SNAKY_CODE        | 
|---------------|---------------|---------------|---------------|
| CORALIE98 | irrelevant | S1D | CORALIE98_3.3 |
| CORALIE07 | irrelevant | S1D | CORALIE07_3.4 |
| CORALIE14 | irrelevant | S1D | CORALIE14_3.8 |
| PEPSI | irrelevant | S1D | PEPSI_1.0 |
| [SOPHIE](http://atlas.obs-hp.fr/sophie/) or [SOPHIE](#flag_download) | irrelevant | S1D | SOPHIE_0.5 |
| [SOPHIE-HE](http://atlas.obs-hp.fr/sophie/)  | irrelevant | S1D | SOPHIE-HE_0.5 |
| [HARPN](http://archives.ia2.inaf.it/tng/) or [HARPN](#flag_download)| >= 3.0.1 (new) | S1D |  HARPN_3.0.1 |
| [ESPRESSO](https://archive.eso.org/scienceportal/home) or [ESPRESSO](#flag_download)| irrelevant | S1D |  ESPRESSO_3.3.6 |
| [NEID](https://neid.ipac.caltech.edu) | irrelevant | E2DS |  NEID_1.0 |
| [NEID-HE](https://neid.ipac.caltech.edu) | irrelevant | E2DS |  NEID-HE_1.0 |
| [HARPS](https://archive.eso.org/scienceportal/home) or [HARPS](#flag_download)| 3.5 (old) | S1D |  HARPS_3.5 |
| [HARPS](https://dace.unige.ch) | >= 3.3.6 (new) | S1D |  HARPS_3.3.6 |
| [FEROS](#flag_download) | irrelevant | S1D |  FEROS_1.0 |
| [FIES](https://www.not.iac.es/observing/forms/fitsarchive/index.php?instrument=FIES) or [FIES](#flag_download) | irrelevenat | S1D |  FIES_1.0 |
| [UVES](#flag_download) | irrelevant | S1D |  UVES_1.0 |
| [HERMES](#flag_download) | irrelevant | S1D |  HERMES_1.0 |

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




## ⑨ An accurate multi-instruments MHK time-series

*The MHK index is a precise activity indicator, but its extraction depends on the stellar effective temperature (`Teff`). Currently, `Teff` is determined independently by SNAKY for each dataset. Therefore, when combining multi-instrument MHK time series, you should ensure that the same `Teff` value is used consistently across all instruments.*

*Let's process a star observed by two different instruments:* 

```python
import src_snaky.run as snaky

#output_dir = '/Users/cretignier/Desktop/Snaky'
files1 = snaky.glob.glob(snaky.myv.TEST_DATASET2) # Cen B HARPS15 (RASSINE)

job1 = snaky.start()
job1.set_output_dir(output_dir)
job1.set_dataset('HD128621','HARPS15_3.3.6',files)  
job1.set_star(ra=219.90, dec=-60.84, prot=36) 
job1.reduce(begin=1, end=14)

files2 = snaky.glob.glob(snaky.myv.TEST_DATASET3) # Cen B HARPS03 (RASSINE)

job2 = snaky.start()
job2.set_output_dir(output_dir)
job2.set_dataset('HD128621','HARPS03_3.3.6',files)  
job2.set_star(ra=219.90, dec=-60.84, prot=36) 
job2.reduce(begin=1, end=14)
```

*SNAKY derives a Teff of 5140K for both instruments in this case. But depending on the dataset, this may not be the case (and perhaps you have your own temperature estimate that is different!). You have two options to fix Teff in SNAKY before computing the MHK:*

*1) specify Teff manually in the `.set_star()`. Let's for instance extract again the activity time-series with `Teff=5100K`*:
```python
job1.set_star(teff=5100)
job1.reduce(begin=9, end=14) # steps from UV continuum to FINCH magnetic cycle

job2.set_star(teff=5100) 
job2.reduce(begin=9, end=14) 
```

*2) query the SNAKY atmospheric DB that you are actively producing using `.get_atmos_db()`:*
```python
job1.get_atmos_db()
```

*This later can be called directly from the `.reduce()` method with the option `atmos_db=True`:*
```python
job1.reduce(begin=9, end=14, atmos_db=True) 
job2.reduce(begin=9, end=14, atmos_db=True) 
```

## ⑩ Analysing SNAKY DB

*While SNAKY can perfectly be used for single-based star analysis, the code was developed for large DB processing.* 

*Dealing with databases can rapidly becomes a chaos, hopefully SNAKY contains some useful functionality. Assuming you ran the [two benchmarks](#flag3) above already:*

```python
import src_snaky.run as snaky
import src_snaky.run_db_analysis as snaky_db

#output_dir = '/Users/cretignier/Desktop/Snaky'

snaky_db.check_comp_rqm(output_dir,instrument='*')

snaky_db.check_snaky_processing(output_dir,instrument='*')
```

*You can notice you have 1 dataset with HARPS03_3.5 and one with HARPS15_3.3.6. The one on HARPS03_3.5 did not pass the magnetic cycle function which is expected since this dataset is made of a single spectrum, inefficient to fit a magnetic cycle. You can also have a look to the created table `..output_dir/database/Snaky_processing_db_HARPS03_3.5.csv`.*

*Let's see other functionality:*

```python

import src_snaky.run_db_analysis as snaky_db

#output_dir = '/Users/cretignier/Desktop/Snaky'

filename = 'All_stars'

#create a summary DB table
snaky_db.create_snaky_db(output_dir, filename=filename)

#create the FINCH DB table
snaky_db.create_snaky_finch_db(output_dir, filename=filename)

#create the RV DB table
snaky_db.create_snaky_rv_db(output_dir,filename=filename)

#create the CCF DB file
snaky_db.create_snaky_ccf_db(output_dir, filename=filename)

#create the spectra DB file
snaky_db.create_snaky_spec_db(output_dir, filename=filename, wave_min=6100, wave_max=6200)

#plot some SNAK parameters
snaky_db.plot_starinfo(output_dir, ins='*', xvar='Teff_SNAKY', yvar='FWHM_G2')


```

## ⑪ Citations

Please cite the relevant works:

- **RASSINE** — [Cretignier et al. 2020b](https://ui.adsabs.harvard.edu/abs/2021A%26A...653A..43C/abstract)

- **Atmospheric parameters** — [Cretignier et al. 2024b](https://ui.adsabs.harvard.edu/abs/2024MNRAS.535.2562C/abstract)

- **MHK activity index** — [Cretignier et al. 2024a](https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.2940C/abstract) + [Cretignier et al. 2024b](https://ui.adsabs.harvard.edu/abs/2024MNRAS.535.2562C/abstract)

- **VSINI** — Cretignier et al. (in prep.)

## ⑫ Uninstall

<a id="flag_uninstall"></a>

*You can delete the snaky Python environment with:*

```bash
conda remove --name snaky --all
```


### BENCHMARK (TABLE)

<a id="flag4"></a>

Dataset1 and Dataset2 are equivalent to the [Benchmark](#flag3)

| Computer | Processor        | VERSION        | LIBRARIES       | DATASET1    | DATASET2    |
|---------------|---------------|---------------|---------------|--------------|--------------|
| MC1 | Apple M4 MAX (2024) | SNAKY (1.2.5) | 3.10.15 | 00 min 58 s  | 00 min 57 s  |
| MC1 | Apple M4 MAX (2024) | SNAKY (1.2.5) | 3.12.5 | 01 min 45 s  | 01 min 30 s  |
| SA1 | Apple M2 (2022) | SNAKY (1.1.1) | 3.10.15 | 01 min 13 s  | 01 min 28 s  |
| MC2 | Intel Mac (2018) | SNAKY (1.2.5) | 3.8.8 | 03 min 31 s  | 06 min 47 s  |
| DF1 | Intel i5 12600k (2021) | SNAKY (1.0.7) | 3.12.5 | 03 min 25 s  | 03 min 28 s  |
| CC1 | Apple M4 PRO (2024) | SNAKY (1.0.1) | 3.10.15 | 01 min 30 s  | 01 min 05 s  |
| ? | Yours! ☺ | SNAKY (1.0.1) | ??? | ???  | ???  |

### RAM requirement

*The RAM requirement for SNAKY (without RASSINE preprocessing) scales as O(N) for HARPS spectra (~300k wavelength bins):*

$$
\text{Total RAM} =
4.2 + 0.72 \times \left( \frac{N}{100} \right) \quad [GB]
$$

