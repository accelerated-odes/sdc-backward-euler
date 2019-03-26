#ifndef _REAL_SPARSE_MATRIX_CONTAINER_H
#define _REAL_SPARSE_MATRIX_CONTAINER_H
#include <cassert>
#include <cmath>
#include "AMReX_REAL.H"
#include "AMReX_CONSTANTS.H"
#include "IntegerVector.H"

using namespace amrex;

template<size_t N, size_t NNZ> class RealSparseMatrixContainer {
    Real* data;
    IntegerVector<NNZ> col_index;
    IntegerVector<N+1> row_count;
public:
    RealSparseMatrix(Real* data_location,
                     IntegerVector<NNZ>& input_col_index,
                     IntegerVector<N+1>& input_row_count) {
        data = data_location;
        col_index = input_col_index;
        row_count = input_row_count;
    }
    
    ~RealSparseMatrix() {}

    __host__ __device__
    Real* begin() {
        return data;
    }

    __host__ __device__
    Real* end() {
        return data + NNZ;
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
    Real& operator[] (unsigned int i) {
        return data[i];
    }
};

#endif
