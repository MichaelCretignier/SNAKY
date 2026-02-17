import glob as glob
import numpy as np 
import os

cwd = os.getcwd()
root = '/'.join(cwd.split('/'))

files_to_compile = np.sort(glob.glob(root+'/Material_snaky/compile_split_*.npy'))
process = []
for f in files_to_compile:
    filename = f.split('&')[-1]
    splitting = f.split('/')[-1].split('&')[0]
    axis = splitting.split('_')[2]
    process.append([f,filename,splitting,axis])
process = np.array(process)

print(' [INFO] Files that will be merged:')
print(process[:,0])

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
    print(' [INFO] The table was recreated: ',root+'/Material_snaky/'+filename)

    print(' [INFO] The splited subparts will be erased...')
    for f in files:
        print(' [INFO] %s was deleted'%(f))
        os.system('rm '+f)
