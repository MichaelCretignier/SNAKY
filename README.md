# SNAKY (a Spectrocopic Novel Analysis Kit of Yarara) v1.0.1

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
SNAKY contains a packaged version of RASSINE (https://github.com/MichaelCretignier/Rassine_public)

## ① Release Date (Soon...!)

The code is missing some important tables that prevent it from running, the code is planned to be released in a few weeks/months.
Thank for your patience.

Please cite the relevant works:

- **RASSINE** — Cretignier et al. (2021)  
  [ADS link](https://ui.adsabs.harvard.edu/abs/2021A%26A...653A..43C/abstract)

- **Atmospheric parameters** — Cretignier et al. (2024b)  
  [ADS link](https://ui.adsabs.harvard.edu/abs/2024MNRAS.535.2562C/abstract)

- **MHK activity index** — Cretignier et al. (2024a, 2024b)  
  [2024a ADS](https://ui.adsabs.harvard.edu/abs/2024MNRAS.527.2940C/abstract)  
  [2024b ADS](https://ui.adsabs.harvard.edu/abs/2024MNRAS.535.2562C/abstract)

- **VSINI** — Cretignier et al. (in prep.)

## ② Contact Me

If you have any problem, please contact me at: michael.cretignier@physics.ox.ac.uk

## ③ Installation

*You can try with your own main Python environment since there are only
a few libraries used. Otherwise:*

[Mac M4 Chip] Python environment (Conda install) (Python 3.13.5)

```bash
conda create -n snaky python=3.13.5 
conda activate snaky 
pip install -r requirements_3.13.5.txt
```

Python environment (Venv install)

```bash
python3 -m venv snaky 
source snaky/bin/activate 
pip install --upgrade pip 
pip install -r requirements_3.13.5.txt
```

## ④ Compile the code

*SNAKY contains a table too heavy for GitHub that need first to be merged back using the compiler function*

```bash
cd ../SNAKY/src_snaky 
python snaky_compile.py
```

## ⑤ Tutorial

*First enter into your local `SNAKY/` directory:*

```bash
cd .../GitHub/SNAKY/
```

*Launch an IPython shell:*

```bash
ipython
```

*To run SNAKY on your spectra, you just need to specify:*

1) An output directory (see the **warning** below)
2) A list of spectra you want to process
3) The starname and the instrument 

*This information has to be provided as:*

```python
import src_snaky.run as snaky

# let's use the SNAKY test dataset
files = glob.glob(snaky.myv.TEST_DATASET1+'/HARPN_3.0.1/RAW/*.fits')

job = snaky.start()
job.set_output_dir('/Users/cretignier/Desktop/test/') # define output dir
job.set_dataset('HD12345', 'HARPN_3.0.1', files)      # define the star + instrument + list of spectra
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

**[WARNING] It's dangerous to add any files not created by SNAKY in your output_directory!** 

```python

job.reset(supression='all')
job.reset(supression='all')

#job.reset()                 # remove all products except RASSINE normalised spectra
#job.reset(supression='all') # remove everything

```

*Let's start again, all the previous lines can be called shortly using the `.reduce()` method:* 

```python
#Shortest call
import src_snaky.run as snaky

files = glob.glob(snaky.myv.TEST_DATASET1+'HARPN_3.0.1/RAW/*.fits')

job = snaky.start()
job.set_output_dir('/Users/cretignier/Desktop/test/')
job.set_dataset('HD12345','HARPN_3.0.1',files) 

job.reduce(begin=1, end=14)
# check the sequence number with job.reduce?
```

See that yellow color list printed at the start? This is the trigger.
It indicates the SNAKY steps done or missing. 

Because of that, `.reduce()` can also restart from a crash point automatically:

```python

# Example of crash
import src_snaky.run as snaky

files = glob.glob(snaky.myv.TEST_DATASET1+'HARPN_3.0.1/RAW/*.fits')

job = snaky.start()
job.set_output_dir('/Users/cretignier/Desktop/test/')
job.set_dataset('HD66666','HARPN_3.0.1',files) # new name to mimic a new dataset

# let's mimic a crash at step of the atmospheric parameters (step=7)

job.reduce(begin=1, end=6) # No the pipeline stopped at stage 6!

# let's rerun the sequence from the start with automatic_db (automatic_db = True by default)
# automatic_db will automatically skip the steps already done

job.reduce(begin=1, end=14, automatic_db=True)

# See? The first stages have been skipped.
# If you want to force the rerun of a specific step, disable the automatic_db option

job.reduce(begin=8, end=8, automatic_db=False, Prot=50, Rs=1.0)


```

You noticed but, the `.reduce()` also monitor the RAM and execution time as a bonus! And save it in `REDUCTION_INFO/` for benchmark purpose.

## ⑥ Launching a RASSINE dataset

*RASSINE is among the longest step in the pipeline (41% of the execution time in the previous example!). But, maybe you already have RASSINE spectra saved on your computer! (I'm sincerely honored of it!)*

*Theoritically, you can't process them directly by SNAKY since those will miss the metadata of the star coordinates (shame on the past me...)*

*However you can specify manually those values in `.reduce()`:*

```python
#Let's use the Alpha Cen B RASSINE dataset
import src_snaky.run as snaky

files = glob.glob(snaky.myv.TEST_DATASET2+'HARPS15_3.3.6/RAW/RASSINE*.p')

job = snaky.start()
job.set_output_dir('/Users/cretignier/Desktop/test/')
job.set_dataset('HD128621','HARPS15_3.3.6',files) 

job.reduce(begin=1, end=14, ra=219.90, dec=-60.84, copy_files=True) # ra and dec in degrees
```

*The `copy_files` option will copy the RASSINE files your pointing on locally in the SNAKY directory as if those were produced by the pipeline. It's not mandatory, but pickle file can be corrupted and copying them is a safer option. Naturally, this double the storage required by duplicating files in your computer. Final decision is let to the user.*

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

## ⑨ BENCHMARK

| VERSION        | DATASET1 (5)    | DATASET2 (6)     |
|---------------|--------------|--------------|
| SNAKY (1.0.1) | 02 min 51 s  | 02 min 01 s  |


