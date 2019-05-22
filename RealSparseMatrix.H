#ifndef _REAL_SPARSE_MATRIX_H
#define _REAL_SPARSE_MATRIX_H
#include <iostream>
#include <cassert>
#include <cmath>
#include "AMReX_REAL.H"
#include "AMReX_CONSTANTS.H"
#include "IntegerVector.H"
#include "RealVector.H"

using namespace amrex;

template<size_t N, size_t NNZ, size_t vector_length> class RealSparseMatrixSet : public RealVectorSet<NNZ, vector_length> {
    IntegerVector<NNZ> col_index;
    IntegerVector<N+1> row_count;
public:

  using ThisType = RealSparseMatrixSet<N, NNZ, vector_length>;
  using RealVectorSet<NNZ, vector_length>::operator[];
  using RealVectorSet<NNZ, vector_length>::operator=;
  using RealVectorSet<NNZ, vector_length>::operator+=;
  using RealVectorSet<NNZ, vector_length>::operator-=;
  using RealVectorSet<NNZ, vector_length>::operator*=;
  using RealVectorSet<NNZ, vector_length>::operator/=;

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  RealSparseMatrixSet() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE    
  RealSparseMatrixSet(IntegerVector<NNZ>& input_col_index,
                      IntegerVector<N+1>& input_row_count) {
    set_csr_layout(input_col_index, input_row_count);
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE    
  ~RealSparseMatrixSet() {}

  void print_layout() {
    std::cout << "col index: ";
    for (auto& x : col_index) {
      std::cout << x << " ";
    }
    std::cout << std::endl << std::endl;
    std::cout << "row count: ";
    for (auto& x : row_count) {
      std::cout << x << " ";
    }
    std::cout << std::endl << std::endl;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& set_csr_layout(IntegerVector<NNZ>& input_col_index,
                           IntegerVector<N+1>& input_row_count) {
    col_index = input_col_index;
    row_count = input_row_count;

    WORKER_SYNC();
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  int lookup_csr_location(size_t irow, size_t icol) {
    int num_in_row = row_count[irow+1] - row_count[irow];
    int row_start_loc = row_count[irow];
    int row_end_loc = row_start_loc + num_in_row - 1;

    int csr_loc = -1;
    for (int i = row_start_loc; i <= row_end_loc; i++) {
      if (col_index[i] == icol) {
        csr_loc = i;
        break;
      }
    }
    return csr_loc;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  RealVector<vector_length>& operator()(size_t i, size_t j) {
    // i: row index [0, N-1]
    // j: col index [0, N-1]
    int csr_loc = lookup_csr_location(i, j);
    assert(csr_loc != -1);
    return (*this)[csr_loc];
  }
};

#endif
