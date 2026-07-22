from pathlib import Path
import shutil
import urllib.request
import zipfile

this_file = Path(__file__).resolve()
snaky_root = this_file.parent.parent

url = "https://zenodo.org/records/20659152/files/Material_snaky.zip"
zip_path = snaky_root / "Material_snaky.zip"

print("\n[INFO] Downloading Material_snaky...\n")

urllib.request.urlretrieve(url, zip_path)

with zipfile.ZipFile(zip_path) as z:
    z.extractall(snaky_root)

shutil.rmtree(snaky_root / "__MACOSX", ignore_errors=True)
zip_path.unlink()
