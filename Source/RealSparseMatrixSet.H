#ifndef _REAL_SPARSE_MATRIX_H
#define _REAL_SPARSE_MATRIX_H
#include <iostream>
#include <cassert>
#include <cmath>

#include "AMReX_REAL.H"
#include "AMReX_CONSTANTS.H"

#include "MathVectorSet.H"
#include "MathVector.H"
#include "VectorStorage.H"

using namespace amrex;

template<long N, long vector_length, long NNZ, template<typename, long> typename InternalStorageType>
class RealSparseMatrixSet {

  MathVector<int, NNZ, InternalStorageType<int, NNZ>> col_index;
  MathVector<int, N+1, InternalStorageType<int, N+1>> row_count;

public:

  MathVectorSet<Real, vector_length, NNZ, InternalStorageType<Real, NNZ>> data;

  using ThisType = RealSparseMatrixSet<N, vector_length, NNZ, InternalStorageType>;

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  RealSparseMatrixSet() {}

  template<typename AnyStorageType1, typename AnyStorageType2>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  RealSparseMatrixSet(MathVector<int, NNZ, AnyStorageType1>& input_col_index,
                      MathVector<int, N+1, AnyStorageType2>& input_row_count) {
    set_csr_layout(input_col_index, input_row_count);
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ~RealSparseMatrixSet() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void resize() {
    data.resize();
    col_index.resize();
    row_count.resize();
  }

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
  MathVector<Real, NNZ, InternalStorageType<Real, NNZ>>& operator[](long i) {
    return data[i];
  }

  template<typename AnyMathType, typename AnyStorageType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& operator*=(MathVector<AnyMathType, vector_length, AnyStorageType>& source) {
    VECTOR_SET_LAMBDA(NNZ, vector_length,
                      [&](const long& iset, const long& ivec) {
                        data[ivec][iset] *= source[ivec];
                      });
    return *this;
  }

  template<typename AnyStorageType1, typename AnyStorageType2>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ThisType& set_csr_layout(MathVector<int, NNZ, AnyStorageType1>& input_col_index,
                           MathVector<int, N+1, AnyStorageType2>& input_row_count) {
    col_index = input_col_index;
    row_count = input_row_count;

    WORKER_SYNC();
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  int lookup_csr_location(long irow, long icol) {
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
  Real& operator()(long i, long j, long k) {
    // i: row index [0, N-1]
    // j: col index [0, N-1]
    int csr_loc = lookup_csr_location(i, j);
    assert(csr_loc != -1);
    return (*this)[k][csr_loc];
  }
};

#endif
