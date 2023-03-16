#!/bin/sh

# Change directory (DO NOT CHANGE!)
repoDir=$(dirname "$(realpath "$0")")
echo $repoDir
cd $repoDir

# Recompile if necessary (DO NOT CHANGE!)
mkdir -p build
nvcc -std=c++11 src/main.cu -o build/main
cd build
#cmake  -DTBB_DIR=${repoDir}/../oneTBB-2019_U9  -DCMAKE_PREFIX_PATH=${repoDir}/../oneTBB-2019_U9/cmake ..
#make -j4
#export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${PWD}/tbb_cmake_build/tbb_cmake_build_subdir_release

## basic run command
./main 16 1
