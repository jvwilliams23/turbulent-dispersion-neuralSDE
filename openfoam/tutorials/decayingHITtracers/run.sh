#!/bin/bash

nproc=$(grep "numberOfSubdomains" system/decomposeParDict | sed "s/;/ /g" | awk '{print $2}')

mpirun -np ${nproc} uncoupledMPPICFoamHW_fluidFirst -parallel | tee logRun

reconstructPar -withZero

