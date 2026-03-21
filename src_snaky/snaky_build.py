import glob as glob
import sys
import numpy as np
import os

import logging
from utils.logger import setup
logger = setup(logging.DEBUG)

cwd = os.getcwd()
root = '/'.join(cwd.split('/')[:-1])

def main():
    files_to_compile = np.sort(glob.glob(root+'/Material_snaky/compile_split_*.npy'))
    process = []
    for f in files_to_compile:
        filename = f.split('_of_file_')[-1]
        splitting = f.split('/')[-1].split('_of_file_')[0]
        axis = splitting.split('_')[2]
        process.append([f,filename,splitting,axis])
    process = np.array(process)

    if len(process)==0:
        logger.warning('No files were found to be merged. Did you already build the code?')
    else:
        for p in np.unique(process[:,1]):
            split_files = process[process[:,1]==p]
            files = split_files[np.argsort(process[:,2])][:,0]
            axis = np.unique(split_files[:,-1])[0]
            filename = np.unique(split_files[:,1])[0]
            merged = []
            for f in files:
                merged.append(np.load(f))
            tp = type(merged[0][0,0])

            if axis=='X':
                merged = np.hstack(merged)
            if axis=='Y':
                merged = np.vstack(merged)

            np.save(root+'/Material_snaky/'+filename,merged.astype(tp))
            logger.info('The table was recreated: ',root+'/Material_snaky/'+filename)

            logger.info('The splited subparts will be erased...')
            for f in files:
                logger.info(f'{f:s} was deleted')
                os.system('rm '+f)


if __name__ == "__main__":
    main()
