#ifndef VODE_SYSTEM_H
#define VODE_SYSTEM_H

#include "MathVector.H"
#include "RealSparseMatrixSet.H"
#include "VectorStorage.H"

class VodeSystem {
public:
  static const size_t neqs = 3;
  static const size_t nnz = 8;

  MathVector<int, nnz, StackCreate<int, nnz>> csr_col_index;
  MathVector<int, neqs+1, StackCreate<int, neqs+1>> csr_row_count;

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  VodeSystem() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ~VodeSystem() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void initialize() {
    csr_col_index.resize();
    csr_row_count.resize();
    SINGLE_LAMBDA([&]() {
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
                  });

    WORKER_SYNC();
  }

  template<size_t Nvec, size_t Nnodes,
           template<typename, size_t> typename AnyStorageType1,
           template<typename, size_t> typename AnyStorageType2,
           template<typename, size_t> typename AnyStorageType3,
           template<typename, size_t> typename AnyStorageType4,
           template<typename, size_t> typename AnyStorageType5>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  static void evaluate(MathVectorSet<Real, Nnodes, Nvec, AnyStorageType1<Real, Nvec>>& time_vec,
                       MathVectorSet<Real, Nvec, neqs, AnyStorageType2<Real, neqs>>* y_vec,
                       MathVectorSet<Real, Nvec, neqs, AnyStorageType3<Real, neqs>>* dydt_vec,
                       MathVector<size_t, Nvec, AnyStorageType4<size_t, Nvec>>& node_map,
                       MathVector<size_t, Nvec,
                                  AnyStorageType5<size_t, Nvec>>& active_map,
                       unsigned int num_active) {

    VECTOR_SET_LAMBDA(neqs, static_cast<size_t>(num_active),
                      [&](const size_t& ieq, const size_t& ivec) {
                        const size_t& kvec = active_map[ivec];

                        auto& y = y_vec[node_map[kvec]][kvec];
                        auto& dydt = dydt_vec[node_map[kvec]][kvec];

                        switch(ieq)
                          {
                          case 0:
                            dydt[0] = -0.04 * y[0] + 1.e4 * y[1] * y[2];
                          case 1:
                            dydt[1] =  0.04 * y[0] - 1.e4 * y[1] * y[2] - 3.e7 * y[1] * y[1];
                          case 2:
                            dydt[2] =  3.e7 * y[1] * y[1];
                          }
                      });

    WORKER_SYNC();
  }

  template<size_t Nvec, size_t Nnodes,
           template<typename, size_t> typename AnyStorageType1,
           template<typename, size_t> typename AnyStorageType2,
           template<typename, size_t> typename AnyStorageType3,
           template<typename, size_t> typename AnyStorageType4,
           template<typename, size_t> typename AnyStorageType5,
           template<typename, size_t> typename AnyStorageType6>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  static void evaluate(MathVectorSet<Real, Nnodes, Nvec, AnyStorageType1<Real, Nvec>>& time_vec,
                       MathVectorSet<Real, Nvec, neqs, AnyStorageType2<Real, neqs>>* y_vec,
                       MathVectorSet<Real, Nvec, neqs, AnyStorageType3<Real, neqs>>* dydt_vec,
                       RealSparseMatrixSet<neqs, Nvec, nnz, AnyStorageType4>& jac_vec,
                       MathVector<size_t, Nvec, AnyStorageType5<size_t, Nvec>>& node_map,
                       MathVector<size_t, Nvec,
                                  AnyStorageType6<size_t, Nvec>>& active_map,
                       unsigned int num_active) {

    evaluate<Nvec>(time_vec, y_vec, dydt_vec, node_map, active_map, num_active);

    VECTOR_SET_LAMBDA(nnz, static_cast<size_t>(num_active),
                      [&](const size_t& inz, const size_t& ivec) {
                        size_t& kvec = active_map[ivec];

                        auto& y = y_vec[node_map[kvec]][kvec];
                        auto& jac = jac_vec[kvec];

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
                      });

    WORKER_SYNC();
  }
};
#endif
