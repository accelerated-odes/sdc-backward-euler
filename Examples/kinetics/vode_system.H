#ifndef VODE_SYSTEM_H
#define VODE_SYSTEM_H

#include "MathVector.H"
#include "RealSparseMatrixSet.H"

template<typename InternalStorageType>
class VodeSystem {
public:
  static const size_t neqs = 3;
  static const size_t nnz = 8;

  MathVector<int, nnz, InternalStorageType> csr_col_index;
  MathVector<int, neqs+1, InternalStorageType> csr_row_count;

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
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

  template<size_t Nvec, typename AnyLambdaType,
           typename AnyStorageType1,
           typename AnyStorageType2,
           typename AnyStorageType3>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  static void evaluate(const MathVector<Real, Nvec, AnyStorageType1>& time_vec,
                       const MathVectorSet<Nvec, neqs, AnyStorageType2>& y_vec,
                       MathVectorSet<Nvec, neqs, AnyStorageType3>& dydt_vec,
                       AnyLambdaType check_todo) {

    VECTOR_SET_LAMBDA(neqs, Nvec,
                      [&](const size_t& ieq, const size_t& ivec) {
                        if (check_todo(ivec)) {
                          auto y = y_vec[ivec];
                          auto dydt = dydt_vec[ivec];

                          switch(ieq)
                            {
                            case 0:
                              dydt[0] = -0.04 * y[0] + 1.e4 * y[1] * y[2];
                            case 1:
                              dydt[1] =  0.04 * y[0] - 1.e4 * y[1] * y[2] - 3.e7 * y[1] * y[1];
                            case 2:
                              dydt[2] =  3.e7 * y[1] * y[1];
                            }
                        }
                      });

    WORKER_SYNC();
  }

  template<size_t Nvec, typename AnyLambdaType,
           typename AnyStorageType1,
           typename AnyStorageType2,
           typename AnyStorageType3,
           typename AnyStorageType4>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  static void evaluate(const MathVector<Real, Nvec, AnyStorageType1>& time_vec,
                       const MathVectorSet<Nvec, neqs, AnyStorageType2>& y_vec,
                       MathVectorSet<Nvec, neqs, AnyStorageType3>& dydt_vec,
                       RealSparseMatrixSet<SystemClass::neqs, vector_length, SystemClass::nnz, AnyStorageType4> jac_vec,
                       AnyLambdaType check_todo) {

    evaluate<Nvec>(time_vec, y_vec, dydt_vec, check_todo);

    VECTOR_SET_LAMBDA(nnz, Nvec,
                      [&](const size_t& inz, const size_t& ivec) {
                        if (check_todo(ivec)) {
                          auto y = y_vec[ivec];
                          auto jac = jac_vec[ivec];

                          switch(inz)
                            {
                            case 0:
                              jac[0] = -0.04e0;
                            case 1:
                              jac[1] =  1.e4 * y[2];
                            case 2:
                              jac[2] =  1.e4 * y[1];

                            case 3:
                              jac[3] =  0.04e0;
                            case 4:
                              jac[4] = -1.e4 * y[2] - 6.e7 * y[1];
                            case 5:
                              jac[5] = -1.e4 * y[1];

                            case 6:
                              jac[6] = 6.e7 * y[1];
                            case 7:
                              jac[7] = 0.0e0;
                            }
                        }
                      });

    WORKER_SYNC();
  }
};
#endif
