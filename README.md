# ECE 284: Final Project

# Title
Parallelizing Pairwise HMMs

## Members

Neil Chen (pychen@ucsd.edu)
Yotam Blum (yblum@ucsd.edu)

## Overview

We implement a parallel algorithm to perform 2 sequences alignment using pairwise HMMs.

## Test Datasets

There are a variant of different test datasets under *data* folder.
The number in the filename indicates how long the sequence is.
For instance, the length of the sequence in *ref128.txt* is 128.
We synthesised the sequences in a way that there are multiple matches and multiple insertions.
We use those to test our algorithm.

## Setting Up

We migrate our code from our personal laptop to UC San Diego's Data Science/Machine Learning Platform (DSMLP).
We simpliy copied the setup from our assignment and reuse the docker image.

## Code Testing

In order to test our code, the user can clone our GitHub repository to his home directory on the DSML platform.
Then run the following command:

```
ssh yturakhia@dsmlp-login.ucsd.edu /opt/launch-sh/bin/launch.sh -c 8 -g 1 -m 16 -i yatisht/ece284-wi23:latest -f ${HOME}/
ECE284-Final-Project/run-commands.sh
```

The default length of the test dataset is 128, the user can select a different length of test dataset by modifying the command line in *run-commands.sh*, just like what we do in the assignment.
