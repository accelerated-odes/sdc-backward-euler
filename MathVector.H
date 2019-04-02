#ifndef _MATH_VECTOR_H
#define _MATH_VECTOR_H
#include <iostream>
#include <cassert>
#include "AMReX_GpuQualifiers.H"

template<class MathType, size_t N> class MathVector {
public:
    MathType data[N];

    AMREX_GPU_HOST_DEVICE
    MathVector() {}

    AMREX_GPU_HOST_DEVICE
    ~MathVector() {}

    AMREX_GPU_HOST_DEVICE
    static MathType* begin(MathVector<MathType, N>& v) {
        return v.data;
    }

    AMREX_GPU_HOST_DEVICE
    static MathType* end(MathVector<MathType, N>& v) {
        return v.data + N;
    }

    AMREX_GPU_HOST_DEVICE
    static MathType* dataPtr(MathVector<MathType, N>& v) {
        return begin();
    }

  virtual void print() {
    size_t i = 0;
    for (size_t i = 0; i < N; i++) {
      std::cout << this->data[i];
      if (i < N-1) std::cout << " ";
      i++;
    }
  }

    AMREX_GPU_HOST_DEVICE
    static void equals(MathVector<MathType, N>& destination,
                       MathVector<MathType, N>& source) {
        for (size_t i = 0; i < N; i++) {
            destination.data[i] = source.data[i];
        }
    }

  AMREX_GPU_HOST_DEVICE
  static void equals(MathVector<MathType, N>& destination, MathType scalar) {
    for (size_t i = 0; i < N; i++) {
      destination.data[i] = scalar;
    }
  }

    AMREX_GPU_HOST_DEVICE
    static MathType& element(MathVector<MathType, N>& v, unsigned int i) {
        return v.data[i];
    }

    AMREX_GPU_HOST_DEVICE
    static void add(MathVector<MathType, N>& x, MathVector<MathType, N>& y) {
        for (size_t i = 0; i < N; i++) {
            x.data[i] += y.data[i];
        }
    }

    AMREX_GPU_HOST_DEVICE
    static void add(MathVector<MathType, N>& x, MathType scalar) {
        for (size_t i = 0; i < N; i++) {
            x.data[i] += scalar;
        }
    }

    AMREX_GPU_HOST_DEVICE
    static void sub(MathVector<MathType, N>& x, MathVector<MathType, N>& y) {
        for (size_t i = 0; i < N; i++) {
            x.data[i] -= y.data[i];
        }
    }

    AMREX_GPU_HOST_DEVICE
    static void sub(MathVector<MathType, N>& x, MathType scalar) {
        for (size_t i = 0; i < N; i++) {
            x.data[i] -= scalar;
        }
    }

    AMREX_GPU_HOST_DEVICE
    static void mul(MathVector<MathType, N>& x, MathVector<MathType, N>& y) {
        for (size_t i = 0; i < N; i++) {
            x.data[i] *= y.data[i];
        }
    }

    AMREX_GPU_HOST_DEVICE
    static void mul(MathVector<MathType, N>& x, MathType scalar) {
        for (size_t i = 0; i < N; i++) {
            x.data[i] *= scalar;
        }
    }

    AMREX_GPU_HOST_DEVICE
    static void div(MathVector<MathType, N>& x, MathVector<MathType, N>& y) {
        for (size_t i = 0; i < N; i++) {
            x.data[i] /= y.data[i];
        }
    }

    AMREX_GPU_HOST_DEVICE
    static void div(MathVector<MathType, N>& x, MathType scalar) {
        for (size_t i = 0; i < N; i++) {
            x.data[i] /= scalar;
        }
    }

    AMREX_GPU_HOST_DEVICE
    static void negate(MathVector<MathType, N>& v) {
        for (size_t i = 0; i < N; i++) {
            v.data[i] = -v.data[i];
        }
    }    
};
#endif
