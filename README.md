# SDC ODE Integrator

This is a fourth-order SDC integrator for ODEs for GPUs.

It relies on AMReX for compilation and definition of the Real type.

Uses analytic solution for the sparse linear solve computed with https://github.com/dwillcox/gauss-jordan-solver

The entire integration is run on the GPU in a single kernel to maintain cache locality.

There is a chemical kinetics example in `Examples/kinetics` with a Readme.

Tested with:

- CUDA 9.2.148, GCC 7.4.0
- CUDA 10.0, GCC 7.3
- CUDA 10.1.168, GCC 8.3.0
