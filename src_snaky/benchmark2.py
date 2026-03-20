import sys

sys.path.append("./")  # Changer par mon path
import src_snaky.run as snaky

output_dir = "out/"
snaky.benchmark2(output_dir)  # check "[INFO] Processing achieved in ..."
