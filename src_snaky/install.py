from pathlib import Path
import shutil
import zipfile
import requests
from tqdm import tqdm

this_file = Path(__file__).resolve()
snaky_root = this_file.parent.parent

url = "https://zenodo.org/records/20659152/files/Material_snaky.zip"
zip_path = snaky_root / "Material_snaky.zip"
material_dir = snaky_root / "Material_snaky"

print("\n[INFO] Downloading Material_snaky from Zenodo, wait...\n")

# Remove any previous installation
shutil.rmtree(material_dir, ignore_errors=True)
shutil.rmtree(snaky_root / "__MACOSX", ignore_errors=True)

# Download with progress bar
response = requests.get(url, stream=True)
response.raise_for_status()

total = int(response.headers.get("content-length", 0))

with open(zip_path, "wb") as f, tqdm(
    desc="Material_snaky.zip",
    total=total,
    unit="B",
    unit_scale=True,
    unit_divisor=1024,
) as bar:
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            f.write(chunk)
            bar.update(len(chunk))

print("\n[INFO] Extracting archive...\n")

with zipfile.ZipFile(zip_path) as z:
    z.extractall(snaky_root)

# Clean up
shutil.rmtree(snaky_root / "__MACOSX", ignore_errors=True)
zip_path.unlink(missing_ok=True)

print(f"[INFO] Material_snaky successfully installed in:\n       {material_dir}")
