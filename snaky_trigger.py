"""
SNAKY — Spectroscopic Novel Analysis Kit of Yarara
"""

import getopt
import sys

ins = 'SOPHIE_0.5'
star = ''
sub_dico = 'matching_diff'
begin = 0
end = 0
debug = False
Rs = None
Prot = None
dir_raw = None
automatic_db = False

if len(sys.argv)>1:
    optlist,args =  getopt.getopt(sys.argv[1:],'i:s:b:e:H:S:A:P:R:')
    for j in optlist:
        elif j[0] == '-s':
            star = j[1]     
        elif j[0] == '-S':
            sub_dico = 'matching_'+j[1]
        elif j[0] == '-i':
            ins = j[1]
        elif j[0] == '-b':
            begin = int(j[1])
        elif j[0] == '-e':
            end = int(j[1])
        elif j[0] == '-H':
            debug = bool(int(j[1]))
        elif j[0] == '-A':
            automatic_db = bool(int(j[1]))
        elif j[0] == '-P':
            Prot = float(j[1])
        elif j[0] == '-R':
            Rs = float(j[1])


import src_snaky.run as snaky

job = snaky.start()
job.set_output_dir('/Users/cretignier/Desktop/test/')

files = glob.glob(snaky.myv.TEST_DATASET+'/HARPN_3.0.1/RAW'+'/*.fits')
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

# all this sequence is identical 

import src_snaky.run as snaky

job = snaky.start()
job.set_output_dir('/Users/cretignier/Desktop/test/')

files = glob.glob(snaky.myv.TEST_DATASET+'/HARPN_3.0.1/RAW'+'/*.fits')
job.set_dataset('HD99999','HARPN_3.0.1',files) #fictive name to mimic a new dataset

job.reduce(begin=1,end=14)
# check the sequence number with job.reduce?

import src_snaky.run as snaky

job = snaky.start()
job.set_output_dir('/Users/cretignier/Desktop/test/')
job.set_dataset('HD12345','HARPN_3.0.1',files)

# let's mimic a crash at step 3
job.reduce(begin=1,end=3)

# let's rerun the sequence from the start with automatic_db
# automatic_db will automatically skip the steps already done
job.reduce(begin=1,end=14, automatic_db=True) 

