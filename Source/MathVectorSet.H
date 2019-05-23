#ifndef _MATH_VECTOR_SET_H
#define _MATH_VECTOR_SET_H
#include <iostream>
#include <cassert>
#include "AMReX_GpuQualifiers.H"
#include "AMReX_Extension.H"
#include "MathVector.H"
#include "VectorGpuMacros.H"

template<class MathType, size_t Nset, size_t Nvec> class MathVectorSet {
public:

  using MVS = MathVectorSet<MathType, Nset, Nvec>;
  
  MathVector<MathType, Nvec> data[Nset];

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
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {
                        data[iset][ivec] = source[iset][ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator=(MathType scalar) {
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {      
                        data[iset][ivec] = scalar;
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& load(MathType* array, size_t array_max_size) {
    for (size_t i = 0; i < Nset; i++)
#ifdef AMREX_USE_CUDA
      data[i].load(&array[i * Nvec * blockDim.x], array_max_size);
#else
      data[i].load(&array[i * Nvec], array_max_size);
#endif
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& save(MathType* array, size_t array_max_size) {
    for (size_t i = 0; i < Nset; i++)
#ifdef AMREX_USE_CUDA
      data[i].save(&array[i * Nvec * blockDim.x], array_max_size);
#else
      data[i].save(&array[i * Nvec], array_max_size);
#endif
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MathVector<MathType, Nvec>& operator[](unsigned int i) {
    return data[i];
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator+=(MVS& y) {
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {      
                        data[iset][ivec] += y[iset][ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator+=(MathVector<MathType, Nvec>& scalar) {
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {      
                        data[iset][ivec] += scalar[ivec];
                      });
    return *this;
  }
  
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator+=(MathType scalar) {
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {      
                        data[iset][ivec] += scalar;
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator-=(MVS& y) {
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {      
                        data[iset][ivec] -= y[iset][ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator-=(MathVector<MathType, Nvec>& scalar) {
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {      
                        data[iset][ivec] -= scalar[ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator-=(MathType scalar) {
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {      
                        data[iset][ivec] -= scalar;
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator*=(MVS& y) {
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {      
                        data[iset][ivec] *= y[iset][ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator*=(MathVector<MathType, Nvec>& scalar) {
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {      
                        data[iset][ivec] *= scalar[ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator*=(MathType scalar) {
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {      
                        data[iset][ivec] *= scalar;
                      });                        
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator/=(MVS& y) {
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {      
                        data[iset][ivec] /= y[iset][ivec];
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator/=(MathVector<MathType, Nvec>& scalar) {
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {      
                        data[iset][ivec] /= scalar[ivec];
                      });
    return *this;
  }
  
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& operator/=(MathType scalar) {
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {      
                        data[iset][ivec] /= scalar;
                      });
    return *this;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  MVS& negate() {
    VECTOR_SET_LAMBDA(Nset, Nvec,
                      [&](size_t& iset, size_t& ivec) {      
                        data[iset][ivec] = -data[iset][ivec];
                      });
    return *this;
  }
};
#endif
