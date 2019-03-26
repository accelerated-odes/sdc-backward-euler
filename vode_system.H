#ifndef VODE_SYSTEM_H
#define VODE_SYSTEM_H

#include "RealVector.H"
#include "IntegerVector.H"
#include "RealSquareMatrix.H"

class VodeSystem {
public:
    static const size_t neqs = 3;
    static const size_t nnz = 8;

    IntegerVector<nnz> csr_col_index;
    IntegerVector<neqs+1> csr_row_count;

    VodeSystem() {
        csr_col_index[0] = 1;
        csr_col_index[1] = 2;
        csr_col_index[2] = 3;
        csr_col_index[3] = 1;
        csr_col_index[4] = 2;
        csr_col_index[5] = 3;
        csr_col_index[6] = 2;
        csr_col_index[7] = 3;

        csr_row_count[0] = 1;
        csr_row_count[1] = 4;
        csr_row_count[2] = 7;
        csr_row_count[3] = 9;        
    }

    __host__ __device__
    static void evaluate(Real time, RealVector<neqs>& y, RealVector<neqs>& rhs) {
        rhs[0] = -0.04 * y[0] + 1.e4 * y[1] * y[2];
        rhs[1] =  0.04 * y[0] - 1.e4 * y[1] * y[2] - 3.e7 * powf(y[1], two);
        rhs[2] =  3.e7 * powf(y[1], two);
    }    

    __host__ __device__
    static void evaluate(Real time, RealVector<neqs>& y, RealVector<neqs>& rhs,
                         RealSparseMatrix<neqs,nnz>& jac) {
        rhs[0] = -0.04 * y[0] + 1.e4 * y[1] * y[2];
        rhs[1] =  0.04 * y[0] - 1.e4 * y[1] * y[2] - 3.e7 * powf(y[1], two);
        rhs[2] =  3.e7 * powf(y[1], two);

        jac.ij(0,0) = -0.04e0;
        jac.ij(0,1) =  1.e4 * y[2];
        jac.ij(0,2) =  1.e4 * y[1];

        jac.ij(1,0) =  0.04e0;
        jac.ij(1,1) = -1.e4 * y[2] - 6.e7 * y[1];
        jac.ij(1,2) = -1.e4 * y[1];

        //        jac.ij(2,0) = 0.0e0;
        jac.ij(2,1) = 6.e7 * y[1];
        jac.ij(2,2) = 0.0e0;
    }
};
#endif
