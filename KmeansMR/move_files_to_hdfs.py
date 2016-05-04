##This script doesn't work, but it was a nice way to gather my thoughts.

import os
import subprocess

THIS_FILE = os.getcwd()
HDFS_PATH = os.path.join(THIS_FILE, "kmeans_hdfs/kmeans/")
DATA_PATH = os.path.join(THIS_FILE,"../../data/smallwiki/")
print HDFS_PATH
subprocess.Popen("hadoop fs -mkdir " + HDFS_PATH)
subprocess.Popen("hadoop fs -mkdir " + HDFS_PATH + "cluster0")

for item in ("tfidf.txt", "dictionary.txt", "cluster0/cluster0.txt"):
    subprocess.Popen("hadoop fs -put " + DATA_PATH + item + " " + HDFS_PATH + "tfidf.txt")
