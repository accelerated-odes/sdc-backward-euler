#ifndef VODE_SYSTEM_H
#define VODE_SYSTEM_H

#include "RealVector.H"
#include "IntegerVector.H"
#include "RealSparseMatrixLayout.H"

class VodeSystem {
public:
    static const size_t neqs = 3;
    static const size_t nnz = 8;

    IntegerVector<nnz> csr_col_index;
    IntegerVector<neqs+1> csr_row_count;

    VodeSystem() {
        csr_col_index[0] = 0;
        csr_col_index[1] = 1;
        csr_col_index[2] = 2;
        csr_col_index[3] = 0;
        csr_col_index[4] = 1;
        csr_col_index[5] = 2;
        csr_col_index[6] = 1;
        csr_col_index[7] = 2;

        csr_row_count[0] = 0;
        csr_row_count[1] = 3;
        csr_row_count[2] = 6;
        csr_row_count[3] = 8;
    }

    __host__ __device__
    static void evaluate(Real time, RealVector<neqs>& y, RealVector<neqs>& rhs) {
        rhs[0] = -0.04 * y[0] + 1.e4 * y[1] * y[2];
        rhs[1] =  0.04 * y[0] - 1.e4 * y[1] * y[2] - 3.e7 * y[1] * y[1];
        rhs[2] =  3.e7 * y[1] * y[1];
    }    

    __host__ __device__
    static void evaluate(Real time, RealVector<neqs>& y, RealVector<neqs>& rhs,
                         RealSparseMatrixLayout<neqs,nnz>& jac) {
        rhs[0] = -0.04 * y[0] + 1.e4 * y[1] * y[2];
        rhs[1] =  0.04 * y[0] - 1.e4 * y[1] * y[2] - 3.e7 * y[1] * y[1];
        rhs[2] =  3.e7 * y[1] * y[1];

        jac.ij(0,0) = -0.04e0;
        jac.ij(0,1) =  1.e4 * y[2];
        jac.ij(0,2) =  1.e4 * y[1];

        jac.ij(1,0) =  0.04e0;
        jac.ij(1,1) = -1.e4 * y[2] - 6.e7 * y[1];
        jac.ij(1,2) = -1.e4 * y[1];

        jac.ij(2,1) = 6.e7 * y[1];
        jac.ij(2,2) = 0.0e0;
    }
};
#endif
