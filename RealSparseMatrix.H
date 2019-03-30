#ifndef _REAL_SPARSE_MATRIX_H
#define _REAL_SPARSE_MATRIX_H
#include <iostream>
#include <cassert>
#include <cmath>
#include "AMReX_REAL.H"
#include "AMReX_CONSTANTS.H"
#include "RealVector.H"
#include "IntegerVector.H"
#include "RealSparseMatrixLayout.H"

using namespace amrex;

template<size_t N, size_t NNZ> class RealSparseMatrix : public RealSparseMatrixLayout<N,NNZ> {
  RealVector<NNZ> data;
public:
  RealSparseMatrix() : RealSparseMatrixLayout<N,NNZ>(data) {}

  RealSparseMatrix(IntegerVector<NNZ>& input_col_index,
		   IntegerVector<N+1>& input_row_count) : RealSparseMatrixLayout<N,NNZ>(input_col_index,
											input_row_count,
											data) {}
};
#endif
