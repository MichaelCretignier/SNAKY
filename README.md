# SNAKY (a Spectrocopic Novel Analysis Kit of Yarara) v1.0.1

<p align="center">
  <img src="logo.png" alt="Project logo" width="400">
</p>

SNAKY is a small code producing as main outputs: 

1) The master spectrum in stellar rest-frame
2) The stellar atmospheric parameters
3) The MHK activity index time-series
4) The stellar vsini (in validation...)

from high resolution spectra in the visible (R>50'000).\
SNAKY is **NOT** an RV pipeline aiming at EPRV precision like YARARA.

## Release Date (Soon...!)

The code is missing some important tables that prevent it from running, the code is planned to be released in a few weeks/months.
Thank for your patience.

## Contact Me

If you have any problem, please contact me at:

michael.cretignier@physics.ox.ac.uk

## Installation

You can try with your own main Python environment since there are only
a few libraries used. Otherwise:

[Mac M4 Chip] Python environment (Conda install) (Python 3.13.5)

```
[TERMINAL] 
conda create -n snaky python=3.13.5 
conda activate snaky 
pip install -r requirements_3.13.5.txt
```

Python environment (Venv install)

```
[TERMINAL] 
python3 -m venv snaky 
source snaky/bin/activate 
pip install --upgrade pip 
pip install -r requirements_3.13.5.txt
```

## Compile the code

*SNAKY contains a table too heavy for GitHub that need first to be merged back using the compiler function*

```
[TERMINAL] 
cd ../SNAKY/src_snaky \
python snaky_compile.py
```

## Tutorial

*first enter into your local Snaky directory*

```
[TERMINAL] 
cd ../SNAKY/Python
```

*Launch an iPython shell*

```bash
[TERMINAL] 
ipython
```

*Create SNAKY directory tree with a STARNAME and INSTRUMENT (SPECTRO_DRS name)*

```ipython
[IPYTHON] 
import src_snaky.run as mrun

job_id = np.random.choice(np.arange(0,9999,1))
job = mrun.run(job_id=0)
job.set_output_dir('/Users/cretignier/Desktop/test/')

#files = glob.glob('/Users/cretignier/Documents/Python/SNAKY/Snaky_data/MY_STAR/data/s1d/HARPN_3.0.1/RAW'+'/*.fits')
#job.set_dataset('HD12345','HARPN_3.0.1',files)

files = glob.glob('/Users/cretignier/Documents/Yarara/HD128621/data/s1d/HARPS15_3.3.6/WORKSPACE/RASSINE_*.p')
job.set_dataset('HD128621','HARPS15_3.3.6',files2)

#initialization
job.init_workspace()
job.preprocess()
job.set_summary()
job.check_spectra()

#pipeline
job.compute_rv_sys()
job.compute_ccf()
job.compute_master()
job.compute_atmos()
job.compute_resolution()
job.compute_vsini(Prot=None, Rs=None) #if Prot and Rs are not provided 
job.compute_abs_continuum()
job.compute_activity()
job.compute_mhk()
job.compute_spectroscopy()
job.compute_mag_cycle()
job.cleaning()

```

Since all the sequence is common for all the stars, this sequence was also reduced in: 



# Your favourite instrument missing?

SNAKY can process spectra from the following spectrographs:

ESPRESSO \
HARPS \
HARPN \
SOPHIE \
NEID 

to add a new instrument you only need 5 information from its header: \
1) jdb   [!mandatory!]
2) berv  [optional]
3) snr   [optional]
4) alpha [~mandatory (manual input)]
4) dec   [~mandatory (manual input)]

then create your own function *read_espresso()* in snaky_main.py
and modify the *extract_header()* function too. 

If only e2ds spectra exist and not s1d, follow NEID example.