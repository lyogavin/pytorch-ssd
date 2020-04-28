
import pathlib
import glob
import os


for dir in glob.glob("./*.*"):
    if dir.rfind(".mp4") == len(dir) - 4 or dir.rfind(".mkv") == len(dir) - 4 or dir.rfind(".webm") == len(dir) - 5:
        #print(f"found {dir}")
        new_dir = dir[0:dir.rfind(".")]
        print(f"renaming {dir} to {new_dir}")
        #os.rename(dir, new_dir)