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
automatic_db = False

if len(sys.argv)>1:
    optlist,args =  getopt.getopt(sys.argv[1:],'i:s:b:e:H:S:A:P:R:')
    for j in optlist:
        if j[0] == '-s':
            star = j[1]     
        elif j[0] == '-S':
            sub_dico = j[1]
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


files = glob.glob(snaky.myv.TEST_DATASET+'/HARPN_3.0.1/RAW'+'/*.fits')
output_dir = '/'.join(snaky.myv.ROOT_DIR.split('/')[0:-2])+'/Snaky/'

import src_snaky.run as snaky

job = snaky.start()
job.set_output_dir(output_dir)
job.set_dataset(star,ins,files) 
job.reduce(begin=begin, end=end, Prot=Prot, Rs=Rs, debug=debug, automatic_db=automatic_db)


