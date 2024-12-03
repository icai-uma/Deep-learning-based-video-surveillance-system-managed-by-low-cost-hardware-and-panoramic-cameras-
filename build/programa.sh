#!/bin/bash
echo "Building program..."
rm CMakeCache.txt
rm cmake_install.cmake
rm Makefile
cmake .. -DCMAKE_BUILD_TYPE=Release
make
echo "Executing program..."
./integracion2
 
