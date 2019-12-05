Use a command line interface to run this program.

Make sure the following packages are installed:
os, sys, csv, math, numpy, operator, matplotlib, skimage, scipy, pandas, sklearn, sift_encoder, sift, argparse, time, json

Input the following depending on which task needs to be run.

Task 1:
usage: task1.py [-h] [-k K] -f F -u U

Label Images Dorsal/Palmar using latent semantics

optional arguments:
  -h, --help  show this help message and exit
  -k K        The number of latent semantics to generate
  -f F        the folder where labeled images are stored
  -u U        the folder where unlabeled images are stored
	

Task 2: 
usage: task2.py [-h] [-c C] -f F -u U

Label Images Dorsal/Palmar using latent semantics

optional arguments:
  -h, --help  show this help message and exit
  -c C        The number of clusters to generate
  -f F        the folder where labeled images are stored
  -u U        the folder where unlabeled images are stored

Task 3:
usage: task3.py [-h] [-k K] [-K K] -f F -i I I I

Find most similar K images for Personalized Page Rank given 3 restarts

optional arguments:
  -h, --help  show this help message and exit
  -k K        The number of nearest neighbors
  -K K        The number of popular images to be visualized
  -f F        the folder where images are stored
  -i I I I    the image IDs

Task 4: 
usage: task4.py [-h] [-k K] -f F -u U -c C

Label Images Dorsal/Palmar using classifiers

optional arguments:
  -h, --help  show this help message and exit
  -k K        PPR - number of edges
  -f F        the folder where labeled images are stored
  -u U        the folder where unlabeled images are stored
  -c C        classifier type: dtree/svm/ppr

Task 5:  
usage: task5.py [-h] -l L -k K -i I [-q Q] [-t T]

Create LSH in memory index. Get t most similar images

optional arguments:
  -h, --help  show this help message and exit
  -l L    	  Number of Layers
  -k K    	  Number of hashes per layer
  -i I    	  Path of the image DB(folder)
  -q Q        Query image path
  -t T    	  Number of similar images

Task 6:
usage: task6.py [-h] [-s S]

Relevant feedback system

optional arguments:
  -h, --help  show this help message and exit
  -s S        Feedback system prob/svm/dt/ppr

