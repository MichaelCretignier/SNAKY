from pathlib import Path
import shutil
import os

this_file = Path(__file__).resolve()
snaky_root = this_file.parent.parent

zip_path = snaky_root / "Material_snaky.zip"
material_dir = snaky_root / "Material_snaky"

print("\n[INFO] Downloading Material_snaky from the Zenodo repository... Wait...\n")

# Remove any previous installation
shutil.rmtree(material_dir, ignore_errors=True)
shutil.rmtree(snaky_root / "__MACOSX", ignore_errors=True)

# Download with curl (progress bar included)
os.system(
    f'curl -L --retry 10 --retry-delay 5 '
    f'"https://zenodo.org/records/20659152/files/Material_snaky.zip" '
    f'-o "{zip_path}"'
)

# Extract in the SNAKY directory
os.system(
    f'unzip -o "{zip_path}" -d "{snaky_root}"'
)

# Clean up
shutil.rmtree(snaky_root / "__MACOSX", ignore_errors=True)
zip_path.unlink(missing_ok=True)

if material_dir.exists():
    print(f"\n[INFO] Material_snaky successfully installed in:\n       {material_dir}")
else:
    print("\n[ERROR] Material_snaky was not found after extraction.")
