#!/bin/bash

decomposePar -force
mpirun -np 28 uncoupledMPPICFoamHW_fluidFirst -parallel > logRun

