#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 08:51:15 2020

@author: cretignier
"""

import getopt
import glob as glob
import os
import subprocess
import sys

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from colorama import Fore
import time

from my_functions import rsync_from_lesta, touch_dir

cwd = os.getcwd()
root = '/'.join(cwd.split('/')[:-1])

stars = None
entry1 = '1'

light = '1'
delete_old = False
instrument = None
output_directory = None
products = ['s1d']
external_drive = True

disk_out,computer,entry2 = ('/hpcstorage/cretigni','lesta0','3')
disk_out,computer,entry2 = ('/srv/scratch/cretigni','bonsai0','1')

if len(sys.argv)>1:
    optlist,args =  getopt.getopt(sys.argv[1:],'s:d:o:H:')
    for j in optlist:
        if j[0] == '-s':
            stars = j[1].split(',')      
        elif j[0] == '-d':
            delete_old = bool(int(j[1]))
        elif j[0] == '-o':
            output_directory = j[1]
        elif j[0] == '-H':
            external_drive = bool(int(j[1]))

if (os.path.exists('/Volumes/MyPassport/Yarara'))&external_drive:
    root = '/Volumes/MyPassport'
    print(Fore.RED +'[INFO] You are downloading the data on the external HD (YOUR PAST YOU)',Fore.RESET+'')


def test_ssh_login(entry1):
    host = f"cretigni@login0{entry1}.astro.unige.ch"
    try:
        # This command runs 'true' on the remote host via SSH
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, "true"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode == 0:
            print(f"[OK] SSH successful to {host}")
            return True
        else:
            print(f"[FAIL] SSH failed to {host}")
            return False
    except Exception as e:
        print(f"[ERROR] Exception connecting to {host}: {e}")
        return False

#path = 'cretigni@login01.astro.unige.ch:/home/astro/cretigni'
#path = 'cretigni@login01.astro.unige.ch:/hpcstorage/cretigni'

def ossystem(line,nb_try=3):
    a=3072
    for j in range(nb_try):
        a = os.system(line)
        if a==0:
            break
    return a

# Test both login01 and login02
for entry1 in ["1", "2"]:
    warning = test_ssh_login(entry1)
    if warning:
        break

for entry2 in ['1','2']:
    warning = ossystem('timeout 10 rsync -av --progress -e "ssh -A cretigni@login0'+entry1 +'.astro.unige.ch ssh" cretigni@'+computer+entry2 +':'+disk_out+'/test.txt '+root,nb_try=1) 
    if warning==0:
        break

print('[INFO] Login selection = login0%s -> bonsai0%s'%(entry1,entry2))

for star in stars:
    time.sleep(3)
    os.system('cd '+root+'/Snaky')
    code = ossystem('rsync -av --progress -e "ssh -A cretigni@login0'+entry1 +'.astro.unige.ch ssh" cretigni@'+computer+entry2 +':'+disk_out+'/Snaky/%s '%(star)+root+'/Snaky')
    print(code)