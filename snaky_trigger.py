"""
SNAKY — Spectroscopic Novel Analysis Kit of Yarara
"""

import getopt
import sys

ins = 'SOPHIE_0.5'
star = ''
sub_dico = 'matching_diff'
begin = 1
end = 14
debug = False
rs = None
prot = None
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
            prot = float(j[1])
        elif j[0] == '-R':
            rs = float(j[1])


files = TODEFINE # glob.glob() to select all the spectra you want or read a csv DB table
output_dir = TODEFINE # Best practice is to use the same directory all stars processed

import src_snaky.run as snaky

job = snaky.start()
job.set_output_dir(output_dir)
job.set_dataset(star,ins,files) 

job.set_star(prot=Prot, rs=Rs)
job.reduce(begin=begin, end=end, debug=debug, automatic_db=automatic_db)


