#ifndef _REAL_SQUARE_MATRIX_H
#define _REAL_SQUARE_MATRIX_H
#include <cassert>
#include <cmath>
#include "AMReX_REAL.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_Extension.H"

using namespace amrex;

template<size_t N> class RealSquareMatrix {
    Real data[N*N];
public:
    RealSquareMatrix() {}
    
    ~RealSquareMatrix() {}

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real* begin() {
        return data;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real* end() {
        return data + N*N;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real* dataPtr() {
        return begin();
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real& ij(size_t i, size_t j) {
        // i: row index [0, N-1]
        // j: col index [0, N-1]
        // rows are contiguous in memory
        return *(data + i * N + j);
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    RealSquareMatrix<N>& operator=(RealSquareMatrix<N>& other) {
        for (size_t i = 0; i < N*N; i++) {
            data[i] = other.data[i];
        }
        return *this;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    RealSquareMatrix<N>& operator=(Real scalar) {
        for (auto& x : *this) {
            x = scalar;
        }
        return *this;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real& operator[] (unsigned int i) {
        assert(i >= 0 && i < N*N);
        return *(data + i);
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    RealSquareMatrix<N>& operator+(RealSquareMatrix<N>& other) {
        for (size_t i = 0; i < N*N; i++) {
            data[i] += other.data[i];
        }
        return *this;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    RealSquareMatrix<N>& operator+(Real x) {
        for (size_t i = 0; i < N*N; i++) {
            data[i] += x;
        }
        return *this;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    RealSquareMatrix<N>& operator-(RealSquareMatrix<N>& other) {
        for (size_t i = 0; i < N*N; i++) {
            data[i] -= other.data[i];
        }
        return *this;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    RealSquareMatrix<N>& operator-(Real x) {
        for (size_t i = 0; i < N*N; i++) {
            data[i] -= x;
        }
        return *this;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    RealSquareMatrix<N>& operator*(RealSquareMatrix<N>& other) {
        for (size_t i = 0; i < N*N; i++) {
            data[i] *= other.data[i];
        }
        return *this;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    RealSquareMatrix<N>& operator*(Real x) {
        for (size_t i = 0; i < N*N; i++) {
            data[i] *= x;
        }
        return *this;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    RealSquareMatrix<N>& operator/(RealSquareMatrix<N>& other) {
        for (size_t i = 0; i < N*N; i++) {
            data[i] /= other.data[i];
        }
        return *this;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    RealSquareMatrix<N>& operator/(Real x) {
        for (size_t i = 0; i < N*N; i++) {
            data[i] /= x;
        }
        return *this;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    void negate() {
        for (auto& x : *this) {
            x = -x;
        }
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    RealSquareMatrix<N>& abs() {
        for (auto& x : *this) {
            x = fabs(x);
        }
        return *this;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    RealSquareMatrix<N>& inv() {
        for (auto& x : *this) {
            x = one/x;
        }
        return *this;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real max_norm() {
        Real max = zero;
        for (auto& x : *this) {
            max = (fabs(x) > max) ? fabs(x) : max;
        }
        return max;
    }

    AMREX_GPU_HOST_DEVICE AMREX_INLINE
    Real min() {
        Real xmin = (*this)[0];
        for (auto& x : *this) {
            xmin = (x < xmin) ? x : xmin;
        }
        return xmin;
    }    
};

#endif
