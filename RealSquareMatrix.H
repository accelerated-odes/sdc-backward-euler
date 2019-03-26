#ifndef _REAL_SQUARE_MATRIX_H
#define _REAL_SQUARE_MATRIX_H
#include <cassert>
#include <cmath>
#include "AMReX_REAL.H"
#include "AMReX_CONSTANTS.H"

using namespace amrex;

template<size_t N> class RealSquareMatrix {
    Real data[N*N];
public:
    RealSquareMatrix() {}
    
    ~RealSquareMatrix() {}

    __host__ __device__
    Real* begin() {
        return data;
    }

    __host__ __device__
    Real* end() {
        return data + N*N;
    }

    __host__ __device__
    Real* dataPtr() {
        return begin();
    }

    __host__ __device__
    Real& ij(size_t i, size_t j) {
        // i: row index [0, N-1]
        // j: col index [0, N-1]
        // rows are contiguous in memory
        return *(data + i * N + j);
    }

    __host__ __device__
    RealSquareMatrix<N>& operator=(RealSquareMatrix<N>& other) {
        for (size_t i = 0; i < N*N; i++) {
            data[i] = other.data[i];
        }
        return *this;
    }

    __host__ __device__
    RealSquareMatrix<N>& operator=(Real scalar) {
        for (auto& x : *this) {
            x = scalar;
        }
        return *this;
    }

    __host__ __device__
    Real& operator[] (unsigned int i) {
        assert(i >= 0 && i < N*N);
        return *(data + i);
    }

    __host__ __device__
    RealSquareMatrix<N>& operator+(RealSquareMatrix<N>& other) {
        for (size_t i = 0; i < N*N; i++) {
            data[i] += other.data[i];
        }
        return *this;
    }

    __host__ __device__
    RealSquareMatrix<N>& operator+(Real x) {
        for (size_t i = 0; i < N*N; i++) {
            data[i] += x;
        }
        return *this;
    }

    __host__ __device__
    RealSquareMatrix<N>& operator-(RealSquareMatrix<N>& other) {
        for (size_t i = 0; i < N*N; i++) {
            data[i] -= other.data[i];
        }
        return *this;
    }

    __host__ __device__
    RealSquareMatrix<N>& operator-(Real x) {
        for (size_t i = 0; i < N*N; i++) {
            data[i] -= x;
        }
        return *this;
    }

    __host__ __device__
    RealSquareMatrix<N>& operator*(RealSquareMatrix<N>& other) {
        for (size_t i = 0; i < N*N; i++) {
            data[i] *= other.data[i];
        }
        return *this;
    }

    __host__ __device__
    RealSquareMatrix<N>& operator*(Real x) {
        for (size_t i = 0; i < N*N; i++) {
            data[i] *= x;
        }
        return *this;
    }

    __host__ __device__
    RealSquareMatrix<N>& operator/(RealSquareMatrix<N>& other) {
        for (size_t i = 0; i < N*N; i++) {
            data[i] /= other.data[i];
        }
        return *this;
    }

    __host__ __device__
    RealSquareMatrix<N>& operator/(Real x) {
        for (size_t i = 0; i < N*N; i++) {
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

    __host__ __device__
    RealSquareMatrix<N>& abs() {
        for (auto& x : *this) {
            x = fabs(x);
        }
        return *this;
    }

    __host__ __device__
    RealSquareMatrix<N>& inv() {
        for (auto& x : *this) {
            x = one/x;
        }
        return *this;
    }

    __host__ __device__
    Real max_norm() {
        Real max = zero;
        for (auto& x : *this) {
            max = (fabs(x) > max) ? fabs(x) : max;
        }
        return max;
    }

    __host__ __device__
    Real min() {
        Real xmin = (*this)[0];
        for (auto& x : *this) {
            xmin = (x < xmin) ? x : xmin;
        }
        return xmin;
    }    
};

#endif
