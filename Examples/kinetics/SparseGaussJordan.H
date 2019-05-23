#ifndef _SPARSE_GAUSS_JORDAN_H
#define _SPARSE_GAUSS_JORDAN_H
#include "AMReX_REAL.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_Extension.H"

class SparseGaussJordan {
public:
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  static void solve(Real* A, Real* x, Real* b) {
    Real scratch_0 = 1.0/A[0];
    Real scratch_1 = b[0]*scratch_0;
    Real scratch_2 = A[2]*scratch_0;
    Real scratch_3 = A[1]*scratch_0;
    Real scratch_4 = 1.0/(-A[3]*scratch_3 + A[4]);
    Real scratch_5 = scratch_4*(-A[3]*scratch_2 + A[5]);
    Real scratch_6 = scratch_4*(-A[3]*scratch_1 + b[1]);
    Real scratch_7 = (-A[6]*scratch_6 + b[2])/(-A[6]*scratch_5 + A[7]);

    x[0] = scratch_1 - scratch_3*scratch_6 - scratch_7*(scratch_2 - scratch_3*scratch_5);
    x[1] = -scratch_5*scratch_7 + scratch_6;
    x[2] = scratch_7;
  }
};

#endif
