#ifndef _MATH_VECTOR_H
#define _MATH_VECTOR_H
#include <iostream>
#include <cassert>

template<class MathType, size_t N> class MathVector {
public:
    MathType data[N];

    __host__ __device__
    MathVector() {}

    __host__ __device__
    ~MathVector() {}

    __host__ __device__
    static MathType* begin(MathVector<MathType, N>& v) {
        return v.data;
    }

    __host__ __device__
    static MathType* end(MathVector<MathType, N>& v) {
        return v.data + N;
    }

    __host__ __device__
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

    __host__ __device__
    static void equals(MathVector<MathType, N>& destination,
                       MathVector<MathType, N>& source) {
        for (size_t i = 0; i < N; i++) {
            destination.data[i] = source.data[i];
        }
    }

  __host__ __device__
  static void equals(MathVector<MathType, N>& destination, MathType scalar) {
    for (size_t i = 0; i < N; i++) {
      destination.data[i] = scalar;
    }
  }

    __host__ __device__
    static MathType& element(MathVector<MathType, N>& v, unsigned int i) {
        return v.data[i];
    }

    __host__ __device__
    static void add(MathVector<MathType, N>& x, MathVector<MathType, N>& y) {
        for (size_t i = 0; i < N; i++) {
            x.data[i] += y.data[i];
        }
    }

    __host__ __device__
    static void add(MathVector<MathType, N>& x, MathType scalar) {
        for (size_t i = 0; i < N; i++) {
            x.data[i] += scalar;
        }
    }

    __host__ __device__
    static void sub(MathVector<MathType, N>& x, MathVector<MathType, N>& y) {
        for (size_t i = 0; i < N; i++) {
            x.data[i] -= y.data[i];
        }
    }

    __host__ __device__
    static void sub(MathVector<MathType, N>& x, MathType scalar) {
        for (size_t i = 0; i < N; i++) {
            x.data[i] -= scalar;
        }
    }

    __host__ __device__
    static void mul(MathVector<MathType, N>& x, MathVector<MathType, N>& y) {
        for (size_t i = 0; i < N; i++) {
            x.data[i] *= y.data[i];
        }
    }

    __host__ __device__
    static void mul(MathVector<MathType, N>& x, MathType scalar) {
        for (size_t i = 0; i < N; i++) {
            x.data[i] *= scalar;
        }
    }

    __host__ __device__
    static void div(MathVector<MathType, N>& x, MathVector<MathType, N>& y) {
        for (size_t i = 0; i < N; i++) {
            x.data[i] /= y.data[i];
        }
    }

    __host__ __device__
    static void div(MathVector<MathType, N>& x, MathType scalar) {
        for (size_t i = 0; i < N; i++) {
            x.data[i] /= scalar;
        }
    }

    __host__ __device__
    static void negate(MathVector<MathType, N>& v) {
        for (size_t i = 0; i < N; i++) {
            v.data[i] = -v.data[i];
        }
    }    
};
#endif
