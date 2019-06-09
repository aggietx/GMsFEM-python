Simple python code of using Generalized Mulitscale Finite Element Method (GMsFEM)
to solve elliptic equation in a rectangle with uniform mesh

This code is written by Shubin Fu

To run it, you need to install python2, numpy, scipy and mpi4py.



cgfem is using continuous Galerkin to solve elliptic equation

GMsFEM is using GMsFEM to solve elliptic equation without parallel

MsFEM_para is using GMsFEM to solve elliptic equation with parallel for find the multiscale
partion of unity and spectral basis (harmonic extension snapshot is used). More optimization is required.

use  mpirun -n 4 python GMsFEM_para to run the parallel code, 4 is the number of processors you want to use

Reference:
Generalized multiscale finite element methods (GMsFEM)
http://www.sciencedirect.com/science/article/pii/S0021999113003392
