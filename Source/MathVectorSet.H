#ifndef _MATH_VECTOR_SET_H
#define _MATH_VECTOR_SET_H
#include <iostream>
#include <cassert>
#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_Extension.H"
#include "MathVector.H"
#include "VectorGpuMacros.H"
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

  size_t stored_size;

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MathVectorSet() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ~MathVectorSet() {}

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

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator=(MVS& source) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] = source[iset][ivec];
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
  MVS& map(const amrex::Array4<MathType>& array,
           const amrex::Dim3& lo, const amrex::Dim3& hi,
           const int src_comp = 0, const int dst_comp = 0,
           const int num_comp = 1) {

    // Maps num_comp components from the Array4,
    // starting at src_comp into the MathVector
    // indexed by dst_comp in the MathVectorSet.
    //
    // The MathVectorSet will not be zero-padded.
    // It should be previously initialized if
    // Nvec > box_number_cells(lo, hi).

    // For CUDA, each block will load Nvec cells.

    for (size_t i = 0; i < num_comp; i++) {
      data[i + dst_comp].map(array, lo, hi, i + src_comp);
    }

    SINGLE_LAMBDA([&](){
                    stored_size = min(Nvec, box_number_cells(lo, hi));
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

    size_t chunk_size = array_comp_size - chunk_offset;

    for (size_t i = 0; i < num_comp; i++) {
      size_t component_chunk_offset = chunk_offset + (i + src_comp) * array_comp_size;
      data[i + dst_comp].save(&array[component_chunk_offset]);
    }

    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& save(amrex::Array4<MathType>& array,
            const amrex::Dim3& lo, const amrex::Dim3& hi,
            const int src_comp = 0, const int dst_comp = 0,
            const int num_comp = 1) {

    // Saves num_comp components to the Array4,
    // starting at src_comp of the MathVectorSet
    // and at dst_comp of the Array4.

    save(array.p, box_number_cells(lo, hi),
         src_comp = src_comp, dst_comp = dst_comp, num_comp = num_comp);

    return *this;
  }

//   AMREX_GPU_HOST_DEVICE AMREX_INLINE
//   MVS& map(const amrex::Array4<MathType>& array,
//            const amrex::Dim3& lo, const amrex::Dim3& hi,
//            const int src_comp = 0, const int dst_comp = 0,
//            const int num_comp = 1) {

//     // Loads num_comp components from the Array4,
//     // starting at src_comp into the RealVector
//     // indexed by dst_comp in the MathVectorSet.
//     //
//     // The MathVectorSet will be zero-padded.
//     // For CUDA, each block will load Nvec cells.

//     VECTOR_SET_LAMBDA(static_cast<size_t>(num_comp), Nvec,
//                       [&](size_t& iset, size_t& ivec) {
//                         size_t cell_index = ivec;
// #ifdef AMREX_USE_CUDA
//                         cell_index += Nvec * blockIdx.x;
// #endif
//                         if (cell_index < box_number_cells(lo, hi))
//                           data[iset + dst_comp][ivec] = array.p[cell_index + (iset + src_comp) * box_number_cells(lo, hi)];
//                         else
//                           data[iset + dst_comp][ivec] = 0.0;
//                       });
//     WORKER_SYNC();
//     return *this;
//   }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MathVector<MathType, Nvec, StorageType>& operator[](unsigned int i) {
    return data[i];
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator+=(MVS& y) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] += y[iset][ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator+=(MathVector<MathType, Nvec, StackMemory<MathType, Nvec>>& scalar) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] += scalar[ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator+=(MathVector<MathType, Nvec, HeapMemoryWindow<MathType>>& scalar) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] += scalar[ivec];
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

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator-=(MVS& y) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] -= y[iset][ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator-=(MathVector<MathType, Nvec, StackMemory<MathType, Nvec>>& scalar) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] -= scalar[ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator-=(MathVector<MathType, Nvec, HeapMemoryWindow<MathType>>& scalar) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] -= scalar[ivec];
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

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator*=(MVS& y) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] *= y[iset][ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator*=(MathVector<MathType, Nvec, StackMemory<MathType, Nvec>>& scalar) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] *= scalar[ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator*=(MathVector<MathType, Nvec, HeapMemoryWindow<MathType>>& scalar) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] *= scalar[ivec];
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

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator/=(MVS& y) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] /= y[iset][ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator/=(MathVector<MathType, Nvec, StackMemory<MathType, Nvec>>& scalar) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] /= scalar[ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator/=(MathVector<MathType, Nvec, HeapMemoryWindow<MathType>>& scalar) {
    VECTOR_SET_LAMBDA(Nset, stored_size,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] /= scalar[ivec];
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
