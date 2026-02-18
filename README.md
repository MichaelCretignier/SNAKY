# SNAKY (a Spectroscopic Novel Analysis Kit of Yarara) v1.0.2

<p align="center">
  <img src="logo.png" alt="Project logo" width="400">
</p>

SNAKY is a small code producing as main outputs: 

1) The master spectrum in stellar rest-frame
2) The stellar atmospheric parameters
3) The MHK activity index time-series
4) The stellar vsini (in validation...)

from high resolution spectra in the visible (R>50'000).

SNAKY is **NOT** an RV pipeline aiming at EPRV precision like YARARA.\
SNAKY includes a packaged version of [RASSINE](https://github.com/MichaelCretignier/Rassine_public)

## ① Release Date (Soon...!)

The code is missing some important tables that prevent it from running, the code is planned to be released in a few weeks/months.
Thanks for your patience.

Please cite the relevant works:

- **RASSINE** — [Cretignier et al. 2020b](https://ui.adsabs.harvard.edu/abs/2021A%26A...653A..43C/abstract)

- **Atmospheric parameters** — [Cretignier et al. 2024b](https://ui.adsabs.harvard.edu/abs/2024MNRAS.535.2562C/abstract)

- **MHK activity index** — [Cretignier et al. 2024a](https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.2940C/abstract) + [Cretignier et al. 2024b](https://ui.adsabs.harvard.edu/abs/2024MNRAS.535.2562C/abstract)

- **VSINI** — Cretignier et al. (in prep.)

## ② Contact Me

If you have any problem, please contact me at: michael.cretignier@physics.ox.ac.uk

## ③ Installation

*You can try with your own main Python environment since there are only
a few libraries used (check the requirements file to see which ones). It is however recommanded to have the same `scikit-learn` version. Otherwise use either the conda or the venv to create a python environment:*

*Based on [benchmark](#flag3), the fastest version is the python 3.10.15:*

### [Option 1] Conda install (Python 3.10.15)

```bash
conda create -n snaky python=3.10.15
conda activate snaky
pip install -r requirements_3.10.15.txt
```

Other tested `../SNAKY/requirements/requirement_PYTHON_VERSION.txt` libraries versioning are available if needed. 

Check if the snaky environment exists and is active:

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
cd ../SNAKY/src_snaky 
python snaky_build.py
```

*You should now have a `../SNAKY/Material_snaky/template_star_SNAKY_3900_6800.npy` file.*

## ⑤ Tutorial

### Step-by-step

*First enter into your local `SNAKY/` directory:*

```bash
cd .../GitHub/SNAKY/
```

*Launch an IPython shell:*

```bash
ipython
```

*To run SNAKY on your spectra, you just need to specify:*

1) An output directory 

```python
output_dir = '/Users/cretignier/Desktop/Snaky/'
```

2) A list of spectra you want to process
3) The starname and the instrument

*This information if specified by:*

```python
import src_snaky.run as snaky

# let's use the SNAKY test dataset
files = snaky.glob.glob(snaky.myv.TEST_DATASET1)
#output_dir = '/Users/cretignier/Desktop/Snaky/'

job = snaky.start()
job.set_output_dir(output_dir)                   # define output dir 
job.set_dataset('HD12345', 'HARPN_3.0.1', files) # define the star + instrument + list of spectra
```

*We then initialized SNAKY which will create the directory tree + normalise the spectra with RASSINE:*

```python
#initialization
job.init_workspace() # Create tree directories
job.preprocess()     # Run RASSINE to normalise spectra
job.set_summary()    # Create the summary table
job.check_spectra()  # Quality flag control on the spectra
```

*Now the data preprocessed, we can finally launch the SNAKY pipeline:*

```python
#pipeline
job.compute_rv_sys()
job.compute_ccf()
job.compute_master()
job.compute_atmos()
job.compute_resolution()
job.compute_vsini(Prot=None, Rs=None) # if Prot and Rs are known by the user
job.compute_abs_continuum()
job.compute_activity()
job.compute_mhk()
job.compute_spectroscopy()
job.compute_mag_cycle()
job.cleaning()
```
*You can check the figures created in the output directory `...IMAGES/`* \
*For now, let's erase our work with the `.reset()` method. For security, it is required to enter the command line twice in order to launch it:*

**WARNING: Do not add new files by yourself inside the created `output_directory/Snaky/`.**

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
# Benchmark Dataset1

import src_snaky.run as snaky

files = snaky.glob.glob(snaky.myv.TEST_DATASET1)
#output_dir = '/Users/cretignier/Desktop/Snaky/'

job = snaky.start()
job.set_output_dir(output_dir)
job.set_dataset('HD12345','HARPN_3.0.1',files) 

job.reduce(begin=1, end=14) # check the sequence number with: job.reduce?
```

You noticed but, the `.reduce()` also monitor the RAM and execution time as a bonus, before to save it in `REDUCTION_INFO/` for [benchmark](#flag3) purpose. Also, did you see that yellow color list printed at the start? This is the trigger. It indicates the SNAKY steps done or missing. 

### Automatic Restart from the Last Completed Step

Because of the trigger, `.reduce()` can also restart from a crash point automatically:

```python
# Simulate a crash

import src_snaky.run as snaky

files = snaky.glob.glob(snaky.myv.TEST_DATASET1)
#output_dir = '/Users/cretignier/Desktop/Snaky/'

job = snaky.start()
job.set_output_dir(output_dir)
job.set_dataset('HD66666', 'HARPN_3.0.1', files)  # fake new dataset

# Stop at step 6 (crash before atmospheric parameters, step 7)
job.reduce(begin=1, end=6)

# Restart from the beginning
# automatic_db=True (default) skips completed steps
job.reduce(begin=1, end=14, automatic_db=True)

# To force re-running a step, disable automatic_db
# Example: recompute inclination with manual Prot and Rs
job.reduce(begin=8, end=8, automatic_db=False, Prot=50, Rs=1.0)


```


## ⑥ Launching a RASSINE dataset

*SNAKY itself is very fast and scales approximately as O(N). However, within the reduction pipeline, RASSINE is the most time-consuming step (about 20–30% of the total execution time in the previous single-spectrum example). RASSINE typically requires ~15 seconds per spectrum and can quickly dominate the total runtime, exceeding the SNAKY processing time by orders of magnitude when many (N>300) spectra are processed:*

$$
\text{Total Execution Time} =
\underbrace{15 N}_{\text{RASSINE}} 
+ 
\underbrace{
116 + 0.76 N \times \left(\frac{\mathrm{FWHM}}{7.3}\right)}_{\text{SNAKY}}
\quad [\mathrm{s}]
$$

*But, maybe you already have RASSINE spectra saved on your computer? (Such a wise decision!)*

*To read them with SNAKY, you just have to specify manually in `.reduce()`the star coordinates (that are missing from the RASSINE metadata):*

<a id="flag2"></a>

```python
# Benchmark Dataset2

# Let's use the Alpha Cen B RASSINE dataset
import src_snaky.run as snaky

files = snaky.glob.glob(snaky.myv.TEST_DATASET2)
#output_dir = '/Users/cretignier/Desktop/Snaky/'

job = snaky.start()
job.set_output_dir(output_dir)
job.set_dataset('HD128621','HARPS15_3.3.6',files) 

job.reduce(begin=1, end=14, ra=219.90, dec=-60.84, copy_files=True) # ra and dec in degrees
```

*The `copy_files=True` option will copy the RASSINE files locally in the SNAKY output directory as if those were produced by the pipeline. This is not mandatory for the code to successfully run, but `.p` pickle files can be corrupted and copying them is a safer option. Naturally, this double the storage requirement by duplicating files in your computer. Final decision is let to the user.*

## ⑦ Large-Scale Processing (SLURM / sbatch parallelization)

*SNAKY is designed to process easily and rapidly thousands of datasets (a dataset corresponds to a star + instrument combination). For large runs, the recommended approach is to use `sbatch`.* \
*This is possible by using the `run_snaky_med.s` SLURM script, that calls the `snaky_trigger.py` Python script.*

```bash
sbatch run_snaky_med.s HD128621 HARPS15_3.3.6 1 14
```

## ⑧ Your favourite instrument missing?

SNAKY can process spectra from the following spectrographs:

1) ESPRESSO 
2) HARPS 
3) HARPN 
4) SOPHIE 
5) NEID 

to add a new instrument you only need 5 information from its header:
1) jdb   [!mandatory!]
2) alpha [!mandatory!]
3) dec   [!mandatory!]
4) berv  [optional]
5) snr   [optional]

Then create your own function `read_espresso()` in `snaky_main.py`
and modify the `extract_header()` function too. 

If only e2ds spectra exist and not s1d, follow NEID example.

## ⑨ BENCHMARK (Computation time)

<a id="flag3"></a>

| Computer | Processor        | VERSION        | LIBRARIES       | [DATASET1](#flag1)    | [DATASET2](#flag2)     |
|---------------|---------------|---------------|---------------|--------------|--------------|
| MC1 | Apple M4 MAX (2024) | SNAKY (1.0.1) | 3.10.15 | 01 min 19 s  | 01 min 23 s  |
| CC1 | Apple M4 PRO (2024) | SNAKY (1.0.1) | 3.10.15 | 01 min 30 s  | 01 min 05 s  |
| CC1 | Apple M4 PRO (2024) | SNAKY (1.0.1) | 3.12.5_latest | 01 min 40 s  | 01 min 36 s  |
| MC1 | Apple M4 MAX (2024) | SNAKY (1.0.1) | 3.12.5_latest | 01 min 52 s  | 01 min 42 s  |
| MC1 | Apple M4 MAX (2024) | SNAKY (1.0.1) | 3.12.5 | 01 min 58 s  | 01 min 47 s  |
| MC2 | Intel Mac (2018) | SNAKY (1.0.1) | 3.8.8 | 05 min 05 s  | 08 min 18 s  |
| ? | Yours! ☺ | SNAKY (1.0.1) | ??? | ???  | ???  |

## Uninstall

*You can delete the snaky Python environment with:*

```bash
conda remove --name snaky --all
```

