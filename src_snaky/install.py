from pathlib import Path
import shutil
import urllib.request
import zipfile

this_file = Path(__file__).resolve()
snaky_root = this_file.parent.parent

url = "https://zenodo.org/records/20659152/files/Material_snaky.zip"
zip_path = snaky_root / "Material_snaky.zip"
material_dir = snaky_root / "Material_snaky"

print("\n[INFO] Downloading Material_snaky from Zenodo...\n")

# Remove any previous installation
shutil.rmtree(material_dir, ignore_errors=True)
shutil.rmtree(snaky_root / "__MACOSX", ignore_errors=True)

# Download the archive
urllib.request.urlretrieve(url, zip_path)

# Extract into the SNAKY root directory
with zipfile.ZipFile(zip_path) as z:
    z.extractall(path=snaky_root)

# Clean up
shutil.rmtree(snaky_root / "__MACOSX", ignore_errors=True)
zip_path.unlink(missing_ok=True)

# Verify installation
if material_dir.exists():
    print(f"[INFO] Material_snaky successfully installed in:\n       {material_dir}")
else:
    raise RuntimeError(
        f"Material_snaky was not found after extraction.\n"
        f"Expected location: {material_dir}"
    )
