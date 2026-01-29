# SNAKY (a Spectrocopic Novel Analysis Kit of Yarara)

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

# Release Date (Soon...!)

The code is missing some important tables that prevent it from running, the code is planned to be released in a few weeks.
Thank for your patience.

# Contact Me

If you have any problem, please contact me at:

michael.cretignier@physics.ox.ac.uk

# Installation

You can try with your own main Python environment since there are only
a few libraries used. Otherwise:

# Installation

Python environment (Conda install)

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

# Compile the code

*SNAKY contains a table too heavy for GitHub that need first to be merged back using the compiler function*

```
[TERMINAL] 
cd ../SNAKY/Python \
python snaky_compiler.py
```

# Tutorial

*first enter into your local Snaky directory*

```
[TERMINAL] 
cd ../SNAKY/Python
```

*Launch an iPython shell*

```
[TERMINAL] 
ipython
```

*Create SNAKY directory tree with a STARNAME and INSTRUMENT (SPECTRO_DRS name)*

```
[IPYTHON] 
run snaky.py -s MY_STAR -i HARPN_3.0.1 -b 0 -e 0
```

*Put the spectra in the specified directory (follow Snaky instruction) and then process the spectra*

```
[IPYTHON] 
run snaky.py -s MY_STAR -i HARPN_3.0.1 -b 1 -e 12
```

*Once satisfied by the output, erase the optional subproducts to reduce size directory*

```
[IPYTHON] 
run snaky.py -s MY_STAR -i HARPN_3.0.1 -b 13 -e 13
```

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