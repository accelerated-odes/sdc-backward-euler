#ifndef _MATH_VECTOR_SET_H
#define _MATH_VECTOR_SET_H
#include <iostream>
#include <cassert>
#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_Extension.H"
#include "MathVector.H"

#include "VectorParallelUtil.H"
#include "VectorStorage.H"

#ifndef AMREX_USE_CUDA
using std::min;
using std::max;
#endif

template<typename MathType, size_t Nset, size_t Nvec, typename StorageType>
class MathVectorSet {
public:

  using MVS = MathVectorSet<MathType, Nset, Nvec, StorageType>;

  MathVector<MathType, Nvec, StorageType> data[Nset];

  amrex::Dim3 lo, hi;

  size_t stored_size;

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MathVectorSet() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ~MathVectorSet() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void resize(size_t new_size = Nvec) {
    // This does not actually change the memory
    // usage but by updating stored_size we
    // can resize the loop bounds for operations.
    SINGLE_LAMBDA([&](){
                    stored_size = new_size;
                    for (size_t i = 0; i < Nset; i++) {
                      data[i].resize(stored_size);
                    }
                  });
  }

  // begin() and end() are iterators for sets
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MathType* begin() {
    return data;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MathType* end() {
    return data + Nset;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MathType* dataPtr() {
    return begin();
  }

  template<typename AnyMathType, typename SourceStoreClass>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator=(MathVectorSet<AnyMathType, Nset, Nvec, SourceStoreClass>& source) {
    resize(min(Nvec, source.stored_size));

    WORKER_SYNC();

    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec)
                      {
                        data[iset][ivec] = static_cast<MathType>(source[iset][ivec]);
                      });

    return *this;
  }

  template<typename AnyMathType, typename SourceStoreClass>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator=(MathVector<AnyMathType, Nvec, SourceStoreClass>& source) {
    resize(min(Nvec, source.stored_size));

    WORKER_SYNC();

    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] = static_cast<MathType>(source[ivec]);
                      });
    return *this;
  }

  template<typename AnyMathType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator=(AnyMathType scalar) {
    resize();

    WORKER_SYNC();

    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] = static_cast<MathType>(scalar);
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& map(MathType* const array, size_t array_comp_size, const int chunk_index = -1,
           const int src_comp = 0, const int dst_comp = 0, const int num_comp = 1) {

    size_t chunk_offset = 0;
#ifdef AMREX_USE_CUDA
    // Each threadblock gets a chunk of each component of the array
    // by default indexed by the threadblock index, unless chunk_index is passed
    if (chunk_index < 0)
      chunk_offset = Nvec * blockIdx.x;
    else
      chunk_offset = Nvec * static_cast<size_t>(chunk_index);
#else
    // If we're not using CUDA, then we need the chunk index to play the role
    // of the threadblock index above.
    assert(chunk_index >= 0);
    chunk_offset = Nvec * static_cast<size_t>(chunk_index);
#endif

    size_t chunk_size = array_comp_size - chunk_offset;

    for (size_t i = 0; i < num_comp; i++) {
      size_t component_chunk_offset = chunk_offset + (i + src_comp) * array_comp_size;
      data[i + dst_comp].map(&array[component_chunk_offset], chunk_size);
    }

    SINGLE_LAMBDA([&](){
                    stored_size = min(Nvec, chunk_size);
                  });

    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& map(MathType* fabptr,
           const amrex::Dim3& lo, const amrex::Dim3& hi,
           const amrex::Dim3& tile_size, const amrex::Dim3& tile_idx,
           const int src_comp = 0, const int dst_comp = 0,
           const int num_comp = 1) {

    // Maps num_comp components from the Fab,
    // starting at src_comp into the MathVector
    // indexed by dst_comp in the MathVectorSet.

    // lo, hi should be the lo, hi for the Fab.

    for (size_t i = 0; i < num_comp; i++) {
      data[i + dst_comp].map(fabptr, lo, hi,
                             tile_size, tile_idx,
                             i + src_comp);
    }

    SINGLE_LAMBDA([&](){
                    stored_size = min(Nvec, static_cast<size_t>(tile_size.x * tile_size.y * tile_size.z));
                  });

    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& save(MathType* array, size_t array_comp_size, const int chunk_index = -1,
            const int src_comp = 0, const int dst_comp = 0, const int num_comp = 1) {

    size_t chunk_offset = 0;
#ifdef AMREX_USE_CUDA
    // Each threadblock saves a chunk of each component of the array
    // by default indexed by the threadblock index, unless chunk_index is passed
    if (chunk_index < 0)
      chunk_offset = Nvec * blockIdx.x;
    else
      chunk_offset = Nvec * static_cast<size_t>(chunk_index);
#else
    // If we're not using CUDA, then we need the chunk index to play the role
    // of the threadblock index above.
    assert(chunk_index >= 0);
    chunk_offset = Nvec * static_cast<size_t>(chunk_index);
#endif

    for (size_t i = 0; i < num_comp; i++) {
      size_t component_chunk_offset = chunk_offset + (i + src_comp) * array_comp_size;
      data[i + dst_comp].save(&array[component_chunk_offset]);
    }

    return *this;
  }

  template<typename AnyIndexType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MathVector<MathType, Nvec, StorageType>& operator[](AnyIndexType i) {
    return data[i];
  }

  template<typename AnyMathType, typename AnyStorageType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator+=(MathVectorSet<AnyMathType, Nset, Nvec, AnyStorageType>& y) {
    size_t looplen = min(stored_size, y.stored_size);
    VECTOR_SET_LAMBDA(Nset, looplen,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] += static_cast<MathType>(y[iset][ivec]);
                      });
    return *this;
  }

  template<typename AnyMathType, typename AnyStorageType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator+=(MathVector<AnyMathType, Nvec, AnyStorageType>& y) {
    size_t looplen = min(stored_size, y.stored_size);
    VECTOR_SET_LAMBDA(Nset, looplen,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] += static_cast<MathType>(y[ivec]);
                      });
    return *this;
  }

  template<typename AnyMathType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator+=(AnyMathType scalar) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] += static_cast<MathType>(scalar);
                      });
    return *this;
  }

  template<typename AnyMathType, typename AnyStorageType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator-=(MathVectorSet<AnyMathType, Nset, Nvec, AnyStorageType>& y) {
    size_t looplen = min(stored_size, y.stored_size);
    VECTOR_SET_LAMBDA(Nset, looplen,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] -= static_cast<MathType>(y[iset][ivec]);
                      });
    return *this;
  }

  template<typename AnyMathType, typename AnyStorageType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator-=(MathVector<AnyMathType, Nvec, AnyStorageType>& y) {
    size_t looplen = min(stored_size, y.stored_size);
    VECTOR_SET_LAMBDA(Nset, looplen,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] -= static_cast<MathType>(y[ivec]);
                      });
    return *this;
  }

  template<typename AnyMathType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator-=(AnyMathType scalar) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] -= static_cast<MathType>(scalar);
                      });
    return *this;
  }

  template<typename AnyMathType, typename AnyStorageType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator*=(MathVectorSet<AnyMathType, Nset, Nvec, AnyStorageType>& y) {
    size_t looplen = min(stored_size, y.stored_size);
    VECTOR_SET_LAMBDA(Nset, looplen,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] *= static_cast<MathType>(y[iset][ivec]);
                      });
    return *this;
  }

  template<typename AnyMathType, typename AnyStorageType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator*=(MathVector<AnyMathType, Nvec, AnyStorageType>& y) {
    size_t looplen = min(stored_size, y.stored_size);
    VECTOR_SET_LAMBDA(Nset, looplen,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] *= static_cast<MathType>(y[ivec]);
                      });
    return *this;
  }

  template<typename AnyMathType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator*=(AnyMathType scalar) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] *= static_cast<MathType>(scalar);
                      });
    return *this;
  }

  template<typename AnyMathType, typename AnyStorageType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator/=(MathVectorSet<AnyMathType, Nset, Nvec, AnyStorageType>& y) {
    size_t looplen = min(stored_size, y.stored_size);
    VECTOR_SET_LAMBDA(Nset, looplen,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] /= static_cast<MathType>(y[iset][ivec]);
                      });
    return *this;
  }

  template<typename AnyMathType, typename AnyStorageType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator/=(MathVector<AnyMathType, Nvec, AnyStorageType>& y) {
    size_t looplen = min(stored_size, y.stored_size);
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] /= static_cast<MathType>(y[ivec]);
                      });
    return *this;
  }

  template<typename AnyMathType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator/=(AnyMathType scalar) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] /= static_cast<MathType>(scalar);
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& negate() {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] = -data[iset][ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& abs(int comp = -1) {
    // apply absolute value to the desired component, or all components
    if (comp >= 0)
      data[comp].abs();
    else
      VECTOR_SET_LAMBDA(Nset, stored_size,
                        [&](size_t& iset,size_t& ivec) {
                          data[iset].abs(ivec);
                        });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& inv(int comp = -1) {
    // apply inverse to the desired component, or all components
    if (comp >= 0)
      data[comp].inv();
    else
      VECTOR_SET_LAMBDA(Nset, stored_size,
                        [&](size_t& iset,size_t& ivec) {
                          data[iset].inv(ivec);
                        });
    return *this;
  }

  template<typename AnyMathType1, typename AnyStorageType1,
           typename AnyMathType2, typename AnyStorageType2>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& dot(MathVectorSet<AnyMathType1, Nset, Nvec, AnyStorageType1>& y,
           MathVector<AnyMathType2, Nvec, AnyStorageType2>& result) {
    // calculate the dot product with y with respect to components and store it in result
    result = 0.0;
    WORKER_SYNC();
    size_t looplen = min(stored_size, y.stored_size);
    VECTOR_SET_LAMBDA(Nset, looplen,
                      [&](size_t& iset, size_t& ivec) {
                        atomicAdd(&result[ivec], static_cast<AnyMathType2>(data[iset][ivec] * static_cast<MathType>(y[iset][ivec])));
                      });
    WORKER_SYNC();
    return *this;
  }

  template<typename AnyMathType, typename AnyStorageType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& max_norm(MathVector<AnyMathType, Nvec, AnyStorageType>& result) {
    // calculate the max norm with respect to components and store it in result
    result = 0.0;
    WORKER_SYNC();
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        atomicMax(&result[ivec], fabs(static_cast<AnyMathType>(data[iset][ivec])));
                      });
    WORKER_SYNC();
    return *this;
  }

  template<typename AnyMathType1, typename AnyStorageType1,
           typename AnyMathType2, typename AnyStorageType2>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& wrms_norm(MathVectorSet<AnyMathType1, Nset, Nvec, AnyStorageType1>& weights,
                 MathVector<AnyMathType2, Nvec, AnyStorageType2>& result) {
    // calculate the component-wise wrms norm with respect to weights and store it in result
    result = 0.0;
    WORKER_SYNC();
    size_t looplen = min(stored_size, weights.stored_size);
    VECTOR_SET_LAMBDA(Nset, looplen,
                      [&](size_t& iset, size_t& ivec) {
                        atomicAdd(&result[ivec],
                          static_cast<AnyMathType2>(data[iset][ivec] *
                                                    data[iset][ivec] *
                            static_cast<MathType>(weights[iset][ivec] *
                                                  weights[iset][ivec])));
                      });
    WORKER_SYNC();
    VECTOR_LAMBDA(looplen,
                  [&](size_t& k) {
                    result[k] = sqrt(result[k] / Nset);
                  });
    WORKER_SYNC();
    return *this;
  }

  template<typename AnyMathType, typename AnyStorageType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& rms_norm(MathVector<AnyMathType, Nvec, AnyStorageType>& result) {
    // calculate the component-wise rms norm and store it in result
    result = 0.0;
    WORKER_SYNC();
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        atomicAdd(&result[ivec], static_cast<AnyMathType>(data[iset][ivec] *
                                                                          data[iset][ivec]));
                      });
    WORKER_SYNC();
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& k) {
                    result[k] = sqrt(result[k] / Nset);
                  });
    WORKER_SYNC();
    return *this;
  }

  template<typename AnyMathType, typename AnyStorageType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& frobenius_norm(MathVector<AnyMathType, Nvec, AnyStorageType>& result) {
    // calculate the component-wise frobenius norm and store it in result
    result = 0.0;
    WORKER_SYNC();
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        atomicAdd(&result[ivec], static_cast<AnyMathType>(data[iset][ivec] *
                                                                          data[iset][ivec]));
                      });
    WORKER_SYNC();
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& k) {
                    result[k] = sqrt(result[k]);
                  });
    WORKER_SYNC();
    return *this;
  }

  template<typename AnyMathType, typename AnyStorageType>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& minimum(MathVector<AnyMathType, Nvec, AnyStorageType>& result) {
    // calculate the min with respect to components and store it in result
    VECTOR_LAMBDA(stored_size,
                  [&](size_t& k) {
                    result[k] = static_cast<AnyMathType>(data[0][k]);
                  });
    WORKER_SYNC();
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        atomicMin(&result[ivec], static_cast<AnyMathType>(data[iset][ivec]));
                      });
    WORKER_SYNC();
    return *this;
  }
};
#endif
