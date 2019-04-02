#ifndef VODE_SYSTEM_H
#define VODE_SYSTEM_H

#include "RealVector.H"
#include "IntegerVector.H"
#include "RealSparseMatrix.H"

class VodeSystem {
public:
    static const size_t neqs = 3;
    static const size_t nnz = 8;

    IntegerVector<nnz> csr_col_index;
    IntegerVector<neqs+1> csr_row_count;

    __host__ __device__
    VodeSystem() {
        csr_col_index.data[0] = 0;
        csr_col_index.data[1] = 1;
        csr_col_index.data[2] = 2;
        csr_col_index.data[3] = 0;
        csr_col_index.data[4] = 1;
        csr_col_index.data[5] = 2;
        csr_col_index.data[6] = 1;
        csr_col_index.data[7] = 2;

        csr_row_count.data[0] = 0;
        csr_row_count.data[1] = 3;
        csr_row_count.data[2] = 6;
        csr_row_count.data[3] = 8;
    }

    __host__ __device__
    static void evaluate(Real time, RealVector<neqs>& y, RealVector<neqs>& rhs) {
        rhs.data[0] = -0.04 * y.data[0] + 1.e4 * y.data[1] * y.data[2];
        rhs.data[1] =  0.04 * y.data[0] - 1.e4 * y.data[1] * y.data[2] - 3.e7 * y.data[1] * y.data[1];
        rhs.data[2] =  3.e7 * y.data[1] * y.data[1];
    }    

    __host__ __device__
    static void evaluate(Real time, RealVector<neqs>& y, RealVector<neqs>& rhs,
                         RealSparseMatrix<neqs,nnz>& jac) {
        rhs.data[0] = -0.04 * y.data[0] + 1.e4 * y.data[1] * y.data[2];
        rhs.data[1] =  0.04 * y.data[0] - 1.e4 * y.data[1] * y.data[2] - 3.e7 * y.data[1] * y.data[1];
        rhs.data[2] =  3.e7 * y.data[1] * y.data[1];

        jac.data[0] = -0.04e0;
        jac.data[1] =  1.e4 * y.data[2];
        jac.data[2] =  1.e4 * y.data[1];

        jac.data[3] =  0.04e0;
        jac.data[4] = -1.e4 * y.data[2] - 6.e7 * y.data[1];
        jac.data[5] = -1.e4 * y.data[1];

        jac.data[6] = 6.e7 * y.data[1];
        jac.data[7] = 0.0e0;
    }
};
#endif
