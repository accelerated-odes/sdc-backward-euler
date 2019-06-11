#ifndef _REAL_SPARSE_MATRIX_H
#define _REAL_SPARSE_MATRIX_H
#include <iostream>
#include <cassert>
#include <cmath>

#include "AMReX_REAL.H"
#include "AMReX_CONSTANTS.H"

#include "MathVectorSet.H"
#include "MathVector.H"

using namespace amrex;

template<size_t N, size_t NNZ, size_t vector_length> class RealSparseMatrixSet {
  MathVector<int, NNZ> col_index;
  MathVector<int, N+1> row_count;
  MathVectorSet<Real, NNZ, vector_length> data;
public:

  using ThisType = RealSparseMatrixSet<N, NNZ, vector_length>;

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  RealSparseMatrixSet() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  RealSparseMatrixSet(MathVector<int, NNZ>& input_col_index,
                      MathVector<int, N+1>& input_row_count) {
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

  template<typename AnyMathType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& operator=(AnyMathType source) {
    data = source;
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MathVector<Real, vector_length>& operator[](size_t i) {
    return data[i];
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& set_csr_layout(MathVector<int, NNZ>& input_col_index,
                           MathVector<int, N+1>& input_row_count) {
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
  MathVector<Real, vector_length>& operator()(size_t i, size_t j) {
    // i: row index [0, N-1]
    // j: col index [0, N-1]
    int csr_loc = lookup_csr_location(i, j);
    assert(csr_loc != -1);
    return (*this)[csr_loc];
  }
};

#endif
