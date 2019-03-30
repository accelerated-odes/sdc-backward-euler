#ifndef _REAL_SPARSE_MATRIX_LAYOUT_H
#define _REAL_SPARSE_MATRIX_LAYOUT_H
#include <iostream>
#include <cassert>
#include <cmath>
#include "AMReX_REAL.H"
#include "AMReX_CONSTANTS.H"
#include "IntegerVector.H"

using namespace amrex;

template<size_t N, size_t NNZ> class RealSparseMatrixLayout {
  Real* data_p;
  IntegerVector<NNZ> col_index;
  IntegerVector<N+1> row_count;
public:
  RealSparseMatrixLayout() {}

  RealSparseMatrixLayout(Real* external_data_p) : data_p(nullptr) {
    map(external_data_p);
  }

  RealSparseMatrixLayout(RealVector<NNZ>& external_data_vector) : data_p(nullptr) {
    map(external_data_vector.dataPtr());
  }

  RealSparseMatrixLayout(IntegerVector<NNZ>& input_col_index,
			 IntegerVector<N+1>& input_row_count) : data_p(nullptr) {
    set_csr_layout(input_col_index, input_row_count);
  }

  RealSparseMatrixLayout(IntegerVector<NNZ>& input_col_index,
			 IntegerVector<N+1>& input_row_count,
			 RealVector<NNZ>& data) {
    set_csr_layout(input_col_index, input_row_count);
    map(data.dataPtr());
  }

  RealSparseMatrixLayout(IntegerVector<NNZ>& input_col_index,
			 IntegerVector<N+1>& input_row_count,
			 Real* external_data_p) {
    set_csr_layout(input_col_index, input_row_count);
    map(external_data_p);
  }

  ~RealSparseMatrixLayout() {}

  __host__ __device__
  Real* begin() {
    return data_p;
  }

  __host__ __device__
  Real* end() {
    if (mapped())
      return (data_p + NNZ);
    else
      return nullptr;
  }

  __host__ __device__
  Real* dataPtr() {
    return begin();
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

  __host__ __device__
  bool mapped() {
    return (data_p != nullptr);
  }

  __host__ __device__
  void map(Real* external_data_p) {
    data_p = external_data_p;
  }

  __host__ __device__
  void unmap() {
    data_p = nullptr;
  }

  __host__ __device__
  void set_csr_layout(IntegerVector<NNZ>& input_col_index,
		      IntegerVector<N+1>& input_row_count) {
    col_index = input_col_index;
    row_count = input_row_count;
  }

  __host__ __device__
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

  __host__ __device__
  Real& ij(size_t i, size_t j) {
    // i: row index [0, N-1]
    // j: col index [0, N-1]
    int csr_loc = lookup_csr_location(i, j);
    assert(csr_loc != -1);
    assert(mapped());
    return data_p[csr_loc];
  }

  __host__ __device__
  Real& operator[] (unsigned int i) {
    assert(mapped());
    return data_p[i];
  }

  __host__ __device__
  RealSparseMatrixLayout<N, NNZ>& operator=(RealSparseMatrixLayout<N, NNZ>& other) {
    assert(mapped());
    for (size_t i = 0; i < NNZ; i++) {
      data_p[i] = other.data_p[i];
    }
    return *this;
  }

  __host__ __device__
  RealSparseMatrixLayout<N, NNZ>& operator=(Real scalar) {
    assert(mapped());
    for (size_t i = 0; i < NNZ; i++) {
      data_p[i] = scalar;
    }
    return *this;
  }

  __host__ __device__
  RealSparseMatrixLayout<N, NNZ>& operator+(RealSparseMatrixLayout<N, NNZ>& other) {
    assert(mapped());
    for (size_t i = 0; i < NNZ; i++) {
      data_p[i] += other.data_p[i];
    }
    return *this;
  }

  __host__ __device__
  RealSparseMatrixLayout<N, NNZ>& operator+(Real scalar) {
    assert(mapped());
    for (size_t i = 0; i < NNZ; i++) {
      data_p[i] += scalar;
    }
    return *this;
  }

  __host__ __device__
  RealSparseMatrixLayout<N, NNZ>& operator-(RealSparseMatrixLayout<N, NNZ>& other) {
    assert(mapped());
    for (size_t i = 0; i < NNZ; i++) {
      data_p[i] -= other.data_p[i];
    }
    return *this;
  }

  __host__ __device__
  RealSparseMatrixLayout<N, NNZ>& operator-(Real scalar) {
    assert(mapped());
    for (size_t i = 0; i < NNZ; i++) {
      data_p[i] -= scalar;
    }
    return *this;
  }

  __host__ __device__
  RealSparseMatrixLayout<N, NNZ>& operator*(RealSparseMatrixLayout<N, NNZ>& other) {
    assert(mapped());
    for (size_t i = 0; i < NNZ; i++) {
      data_p[i] *= other.data_p[i];
    }
    return *this;
  }

  __host__ __device__
  RealSparseMatrixLayout<N, NNZ>& operator*(Real scalar) {
    assert(mapped());
    for (size_t i = 0; i < NNZ; i++) {
      data_p[i] *= scalar;
    }
    return *this;
  }

  __host__ __device__
  RealSparseMatrixLayout<N, NNZ>& operator/(RealSparseMatrixLayout<N, NNZ>& other) {
    assert(mapped());
    for (size_t i = 0; i < NNZ; i++) {
      data_p[i] /= other.data_p[i];
    }
    return *this;
  }

  __host__ __device__
  RealSparseMatrixLayout<N, NNZ>& operator/(Real scalar) {
    assert(mapped());
    for (size_t i = 0; i < NNZ; i++) {
      data_p[i] /= scalar;
    }
    return *this;
  }

  __host__ __device__
  RealSparseMatrixLayout<N, NNZ>& abs() {
    assert(mapped());
    for (size_t i = 0; i < NNZ; i++) {
      data_p[i] = fabs(data_p[i]);
    }
    return *this;
  }

  __host__ __device__
  RealSparseMatrixLayout<N, NNZ>& inv() {
    assert(mapped());
    for (size_t i = 0; i < NNZ; i++) {
      data_p[i] = one/data_p[i];
    }
    return *this;
  }
};

#endif
