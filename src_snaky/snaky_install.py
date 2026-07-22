import os

print('\n[INFO] Loading Material_snaky directory from the Zenodo repository... Wait...\n')

os.system(
    "curl -L https://zenodo.org/records/20659152/files/Material_snaky.zip -o Material_snaky.zip && unzip -o Material_snaky.zip"
)

os.system('rm -r __MACOSX/')

try:
    os.system('mv Material_snaky/ ../')
    os.system('rm Material_snaky.zip')
except:
    print(' [ERROR] Unzip did not work. Unzip the directory by yourself and move it inside SNAKY/')
