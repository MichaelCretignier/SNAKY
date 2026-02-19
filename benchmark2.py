import src_snaky.run as snaky

files = snaky.glob.glob(snaky.myv.TEST_DATASET2)
output_dir = "./out/"

job = snaky.start()
job.set_output_dir(output_dir)
job.set_dataset("HD128621", "HARPS15_3.3.6", files)

job.set_star(ra=219.90, dec=-60.84, prot=36)  # ra and dec in degrees (prot optional)
job.reduce(begin=1, end=14, copy_files=True)
