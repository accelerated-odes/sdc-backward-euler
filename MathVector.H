#ifndef _MATH_VECTOR_H
#define _MATH_VECTOR_H
#include <iostream>
#include <cassert>

template<class MathType, size_t N> class MathVector {
    MathType data[N];
public:
    MathVector() {}
    
    ~MathVector() {}

    __host__ __device__
    MathType* begin() {
        return data;
    }

    __host__ __device__
    MathType* end() {
        return data + N;
    }

    __host__ __device__
    MathType* dataPtr() {
        return begin();
    }

    void print() {
        size_t i = 0;
        for (const auto& x : (*this)) {
            std::cout << x;
            if (i < N-1) std::cout << " ";
            i++;
        }
    }

    __host__ __device__
    MathVector<MathType, N>& operator=(MathVector<MathType, N>& other) {
        for (size_t i = 0; i < N; i++) {
            data[i] = other.data[i];
        }
        return *this;
    }

    __host__ __device__
    MathVector<MathType, N>& operator=(MathType scalar) {
        for (auto& x : *this) {
            x = scalar;
        }
        return *this;
    }

    __host__ __device__
    MathType& operator[] (unsigned int i) {
        assert(i >= 0 && i < N);
        return *(data + i);
    }

    __host__ __device__
    MathVector<MathType, N>& operator+(MathVector<MathType, N>& other) {
        for (size_t i = 0; i < N; i++) {
            data[i] += other.data[i];
        }
        return *this;
    }

    __host__ __device__
    MathVector<MathType, N>& operator+(MathType x) {
        for (size_t i = 0; i < N; i++) {
            data[i] += x;
        }
        return *this;
    }

    __host__ __device__
    MathVector<MathType, N>& operator-(MathVector<MathType, N>& other) {
        for (size_t i = 0; i < N; i++) {
            data[i] -= other.data[i];
        }
        return *this;
    }

    __host__ __device__
    MathVector<MathType, N>& operator-(MathType x) {
        for (size_t i = 0; i < N; i++) {
            data[i] -= x;
        }
        return *this;
    }

    __host__ __device__
    MathVector<MathType, N>& operator*(MathVector<MathType, N>& other) {
        for (size_t i = 0; i < N; i++) {
            data[i] *= other.data[i];
        }
        return *this;
    }

    __host__ __device__
    MathVector<MathType, N>& operator*(MathType x) {
        for (size_t i = 0; i < N; i++) {
            data[i] *= x;
        }
        return *this;
    }

    __host__ __device__
    MathVector<MathType, N>& operator/(MathVector<MathType, N>& other) {
        for (size_t i = 0; i < N; i++) {
            data[i] /= other.data[i];
        }
        return *this;
    }

    __host__ __device__
    MathVector<MathType, N>& operator/(MathType x) {
        for (size_t i = 0; i < N; i++) {
            data[i] /= x;
        }
        return *this;
    }

    __host__ __device__
    void negate() {
        for (auto& x : *this) {
            x = -x;
        }
    }    
};

// template<class VectorType>
// std::ostream& operator<<(std::ostream& output_stream, VectorType& vec) {
//     vec.stream_vector(output_stream, vec);
// }
#endif
