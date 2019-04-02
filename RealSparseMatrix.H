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

template<size_t N, size_t NNZ> class RealSparseMatrix : public RealVector<NNZ> {
    IntegerVector<NNZ> col_index;
    IntegerVector<N+1> row_count;
public:

    using RealVector<NNZ>::element;
    using RealVector<NNZ>::equals;
    using RealVector<NNZ>::add;
    using RealVector<NNZ>::sub;
    using RealVector<NNZ>::mul;
    using RealVector<NNZ>::div;    

    AMREX_GPU_HOST_DEVICE
    RealSparseMatrix() {}

    AMREX_GPU_HOST_DEVICE    
    RealSparseMatrix(IntegerVector<NNZ>& input_col_index,
                           IntegerVector<N+1>& input_row_count) {
        RealSparseMatrix<N,NNZ>::set_csr_layout(*this, input_col_index, input_row_count);
    }

    AMREX_GPU_HOST_DEVICE    
    ~RealSparseMatrix() {}

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

    AMREX_GPU_HOST_DEVICE
    static void set_csr_layout(RealSparseMatrix<N,NNZ>& ml,
                               IntegerVector<NNZ>& input_col_index,
                               IntegerVector<N+1>& input_row_count) {
        IntegerVector<NNZ>::equals(ml.col_index, input_col_index);
        IntegerVector<N+1>::equals(ml.row_count, input_row_count);
    }

    AMREX_GPU_HOST_DEVICE
    static int lookup_csr_location(RealSparseMatrix<N,NNZ>& ml,
                                   size_t irow, size_t icol) {
        int num_in_row = ml.row_count.data[irow+1] - ml.row_count.data[irow];
        int row_start_loc = ml.row_count.data[irow];
        int row_end_loc = row_start_loc + num_in_row - 1;

        int csr_loc = -1;
        for (int i = row_start_loc; i <= row_end_loc; i++) {
            if (ml.col_index.data[i] == icol) {
                csr_loc = i;
                break;
            }
        }

        return csr_loc;
    }

    AMREX_GPU_HOST_DEVICE
    static Real& ij(RealSparseMatrix<N,NNZ>& ml, size_t i, size_t j) {
        // i: row index [0, N-1]
        // j: col index [0, N-1]
        int csr_loc = RealSparseMatrix<N,NNZ>::lookup_csr_location(ml, i, j);
        assert(csr_loc != -1);
        return RealSparseMatrix<N,NNZ>::element(ml, csr_loc);
    }
};

#endif
