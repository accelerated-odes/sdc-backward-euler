#ifndef _REAL_SPARSE_MATRIX_H
#define _REAL_SPARSE_MATRIX_H
#include <cassert>
#include <cmath>
#include "AMReX_REAL.H"
#include "AMReX_CONSTANTS.H"
#include "RealVector.H"
#include "IntegerVector.H"

using namespace amrex;

template<size_t N, size_t NNZ> class RealSparseMatrix {
    RealVector<NNZ> data;
    IntegerVector<NNZ> col_index;
    IntegerVector<N+1> row_count;
public:
    RealSparseMatrix() {}
    
    RealSparseMatrix(IntegerVector<NNZ>& input_col_index,
                     IntegerVector<N+1>& input_row_count) {
        set_csr_layout(input_col_index, input_row_count);
    }
    
    ~RealSparseMatrix() {}

    void set_csr_layout(IntegerVector<NNZ>& input_col_index,
                        IntegerVector<N+1>& input_row_count) {
        col_index = input_col_index;
        row_count = input_row_count;
    }

    __host__ __device__
    Real* begin() {
        return data.begin();
    }

    __host__ __device__
    Real* end() {
        return data.end();
    }

    __host__ __device__
    Real* dataPtr() {
        return begin();
    }

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
        return data[csr_loc];
    }

    __host__ __device__
    RealSparseMatrix<N, NNZ>& operator=(RealSparseMatrix<N, NNZ>& other) {
        data = other.data;
        return *this;
    }

    __host__ __device__
    RealSparseMatrix<N, NNZ>& operator=(Real scalar) {
        data = scalar;
        return *this;
    }

    __host__ __device__
    Real& operator[] (unsigned int i) {
        return data[i];
    }

    __host__ __device__
    RealSparseMatrix<N, NNZ>& operator+(RealSparseMatrix<N, NNZ>& other) {
        data + other.data;
        return *this;
    }

    __host__ __device__
    RealSparseMatrix<N, NNZ>& operator+(Real x) {
        data + x;
        return *this;
    }

    __host__ __device__
    RealSparseMatrix<N, NNZ>& operator-(RealSparseMatrix<N, NNZ>& other) {
        data - other.data;
        return *this;
    }

    __host__ __device__
    RealSparseMatrix<N, NNZ>& operator-(Real x) {
        data - x;
        return *this;
    }

    __host__ __device__
    RealSparseMatrix<N, NNZ>& operator*(RealSparseMatrix<N, NNZ>& other) {
        data * other.data;
        return *this;
    }

    __host__ __device__
    RealSparseMatrix<N, NNZ>& operator*(Real x) {
        data * x;
        return *this;
    }

    __host__ __device__
    RealSparseMatrix<N, NNZ>& operator/(RealSparseMatrix<N, NNZ>& other) {
        data / other.data;
        return *this;
    }

    __host__ __device__
    RealSparseMatrix<N, NNZ>& operator/(Real x) {
        data / x;
        return *this;
    }

    __host__ __device__
    void negate() {
        data.negate();
    }

    __host__ __device__
    RealSparseMatrix<N, NNZ>& abs() {
        data.abs();
        return *this;
    }

    __host__ __device__
    RealSparseMatrix<N, NNZ>& inv() {
        data.inv();
        return *this;
    }

    __host__ __device__
    Real max_norm() {
        return data.max_norm();
    }

    __host__ __device__
    Real min() {
        return data.min();
    }    
};

#endif
