# SNAKY (a Spectroscopic Novel Analysis Kit of Yarara) v1.0.1

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
a few libraries used (check the requirements file to see which ones). Otherwise:*

[Conda install] Python environment (Python 3.12.5)

```bash
conda create -n snaky python=3.12.5 
conda activate snaky 
pip install -r requirements_3.12.5.txt
```

Check if the snaky environment exists:

```bash
conda env list
```

[Venv install] Python environment (Python 3.12.5)

```bash
python3 -m venv snaky 
source snaky/bin/activate 
pip install --upgrade pip 
pip install -r requirements_3.12.5.txt
```

## ④ Compile the code

*SNAKY contains a table too heavy for GitHub that need first to be merged back using the compiler function*

```bash
cd ../SNAKY/src_snaky 
python snaky_compile.py
```

*You should now have a `../SNAKY/Material_snaky/template_star_SNAKY_3900_6800.npy` file.*

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

1) An output directory
2) A list of spectra you want to process
3) The starname and the instrument 

*This information has to be provided as:*

```python
import src_snaky.run as snaky

# let's use the SNAKY test dataset
files = snaky.glob.glob(snaky.myv.TEST_DATASET1+'/HARPN_3.0.1/RAW/*.fits')
output_dir = '/Users/cretignier/Desktop/' # <--- change it!

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

## ⑤ Tutorial (Simplified call)

*Let's start again, all the previous lines can be called shortly using the `.reduce()` method:* 

<a id="flag1"></a>

```python
# Benchmark Dataset1
import src_snaky.run as snaky

files = snaky.glob.glob(snaky.myv.TEST_DATASET1+'HARPN_3.0.1/RAW/*.fits')
output_dir = '/Users/cretignier/Desktop/'  # <--- change it!

job = snaky.start()
job.set_output_dir(output_dir)
job.set_dataset('HD12345','HARPN_3.0.1',files) 

job.reduce(begin=1, end=14) # check the sequence number with: job.reduce?
```

You noticed but, the `.reduce()` also monitor the RAM and execution time as a bonus, before to save it in `REDUCTION_INFO/` for [benchmark](#flag3) purpose. Also, did you see that yellow color list printed at the start? This is the trigger. It indicates the SNAKY steps done or missing. 

Because of that, `.reduce()` can also restart from a crash point automatically:

```python

# Example of crash
import src_snaky.run as snaky

files = snaky.glob.glob(snaky.myv.TEST_DATASET1+'HARPN_3.0.1/RAW/*.fits')
output_dir = '/Users/cretignier/Desktop/'  # <--- change it!

job = snaky.start()
job.set_output_dir(output_dir)
job.set_dataset('HD66666','HARPN_3.0.1',files) # new name to mimic a new dataset

# let's mimic a crash at step of the atmospheric parameters (step=7)

job.reduce(begin=1, end=6) # No the pipeline stopped at stage 6!

# let's rerun the sequence from the start with automatic_db (automatic_db = True by default)
# automatic_db will automatically skip the steps already done

job.reduce(begin=1, end=14, automatic_db=True)

# See? The first stages have been skipped.
# If you want to force the rerun of a specific step, disable the automatic_db option

# For instance SNAKY can compute the inclination angle if Prot is specified manually
job.reduce(begin=8, end=8, automatic_db=False, Prot=50, Rs=1.0)


```


## ⑥ Launching a RASSINE dataset

*SNAKY itself is very fast, but RASSINE is among the longest step in the pipeline (~20-30% of the execution time in the previous example for a single spectrum!). Computational time of RASSINE is usually ~15s per spectrum.*

*But, maybe you already have RASSINE spectra saved on your computer? (I'm sincerely honoured of it!)*

*Theoretically, you can't process them directly by SNAKY since those will miss the metadata of the star coordinates (shame on the past myself...)*

*However you can specify manually those values in `.reduce()`:*

<a id="flag2"></a>

```python
# Benchmark Dataset2
# Let's use the Alpha Cen B RASSINE dataset
import src_snaky.run as snaky

files = snaky.glob.glob(snaky.myv.TEST_DATASET2+'HARPS15_3.3.6/RAW/RASSINE*.p')
output_dir = '/Users/cretignier/Desktop/'   # <--- change it!

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

| Processor        | VERSION        | [DATASET1](#flag1)    | [DATASET2](#flag2)     |
|---------------|---------------|--------------|--------------|
| Apple M4 (2024) | SNAKY (1.0.1) | 01 min 58 s  | 01 min 52 s  |
| Intel Mac (2018) | SNAKY (1.0.1) | 04 min 57 s  | 09 min 00 s  |
| Yours! ☺ | SNAKY (1.0.1) | ???  | ???  |

## Uninstall

```bash
[TERMINAL] 
conda remove --name snaky --all
```

