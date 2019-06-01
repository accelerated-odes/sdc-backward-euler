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

template<class MathType, size_t Nset, size_t Nvec, class StorageType>
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
  void resize(size_t new_size = 0) {
    // This does not actually change the memory
    // usage but by updating stored_size we
    // can resize the loop bounds for operations.
    stored_size = new_size;
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

  virtual void print() {
    size_t i = 0;
    for (size_t i = 0; i < Nset; i++) {
      data[i].print();
      if (i < Nset - 1) std::cout << std::endl;
      i++;
    }
  }

  template<class SourceStoreClass>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator=(MathVectorSet<MathType, Nset, Nvec, SourceStoreClass>& source) {
    SINGLE_LAMBDA([&]() { resize(min(Nvec, source.stored_size)); });
    WORKER_SYNC();
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] = source[iset][ivec];
                      });
    return *this;
  }

  template<class SourceStoreClass>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator=(MathVector<MathType, Nvec, SourceStoreClass>& source) {
    SINGLE_LAMBDA([&]() { resize(min(Nvec, source.stored_size)); });
    WORKER_SYNC();
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] = source[ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator=(MathType scalar) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] = scalar;
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& map(MathType* const array, size_t array_comp_size, int chunk_index = -1,
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
  MVS& save(MathType* array, size_t array_comp_size, int chunk_index = -1,
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

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MathVector<MathType, Nvec, StorageType>& operator[](unsigned int i) {
    return data[i];
  }

  template<class SourceStoreClass>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator+=(MathVectorSet<MathType, Nset, Nvec, SourceStoreClass>& y) {
    size_t looplen = min(stored_size, y.stored_size);
    VECTOR_SET_LAMBDA(Nset, looplen,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] += y[iset][ivec];
                      });
    return *this;
  }

  template<class SourceStoreClass>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator+=(MathVector<MathType, Nvec, SourceStoreClass>& y) {
    size_t looplen = min(stored_size, y.stored_size);
    VECTOR_SET_LAMBDA(Nset, looplen,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] += y[ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator+=(MathType scalar) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] += scalar;
                      });
    return *this;
  }

  template<class SourceStoreClass>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator-=(MathVectorSet<MathType, Nset, Nvec, SourceStoreClass>& y) {
    size_t looplen = min(stored_size, y.stored_size);
    VECTOR_SET_LAMBDA(Nset, looplen,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] -= y[iset][ivec];
                      });
    return *this;
  }

  template<class SourceStoreClass>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator-=(MathVector<MathType, Nvec, SourceStoreClass>& y) {
    size_t looplen = min(stored_size, y.stored_size);
    VECTOR_SET_LAMBDA(Nset, looplen,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] -= y[ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator-=(MathType scalar) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] -= scalar;
                      });
    return *this;
  }

  template<class SourceStoreClass>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator*=(MathVectorSet<MathType, Nset, Nvec, SourceStoreClass>& y) {
    size_t looplen = min(stored_size, y.stored_size);
    VECTOR_SET_LAMBDA(Nset, looplen,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] *= y[iset][ivec];
                      });
    return *this;
  }

  template<class SourceStoreClass>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator*=(MathVector<MathType, Nvec, SourceStoreClass>& y) {
    size_t looplen = min(stored_size, y.stored_size);
    VECTOR_SET_LAMBDA(Nset, looplen,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] *= y[ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator*=(MathType scalar) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] *= scalar;
                      });
    return *this;
  }

  template<class SourceStoreClass>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator/=(MathVectorSet<MathType, Nset, Nvec, SourceStoreClass>& y) {
    size_t looplen = min(stored_size, y.stored_size);
    VECTOR_SET_LAMBDA(Nset, looplen,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] /= y[iset][ivec];
                      });
    return *this;
  }

  template<class SourceStoreClass>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator/=(MathVector<MathType, Nvec, SourceStoreClass>& y) {
    size_t looplen = min(stored_size, y.stored_size);
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] /= y[ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator/=(MathType scalar) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] /= scalar;
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
};
#endif
