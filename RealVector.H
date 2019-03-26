#ifndef _REAL_VECTOR_H
#define _REAL_VECTOR_H
#include <cassert>
#include <cmath>
#include "AMReX_REAL.H"
#include "AMReX_CONSTANTS.H"
#include "MathVector.H"

using namespace amrex;

template<size_t N> class RealVector : public MathVector<Real,N> {
public:
    using MathVector<Real,N>::operator=;
    using MathVector<Real,N>::begin;
    using MathVector<Real,N>::end;
    
    __host__ __device__
    RealVector<N>& abs() {
        for (auto& x : *this) {
            x = fabs(x);
        }
        return *this;
    }

    __host__ __device__
    RealVector<N>& inv() {
        for (auto& x : *this) {
            x = one/x;
        }
        return *this;
    }

    __host__ __device__
    Real dot(RealVector<N>& other) {
        Real sum = zero;
        for (size_t i = 0; i < N; i++) {
            sum += (*this)[i] * other[i];
        }
        return sum;
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
    Real wrms_norm(RealVector<N>& weights) {
        Real sum = zero;
        for (size_t i = 0; i < N; i++) {
            sum += pow((*this)[i] * weights[i], two);
        }
        sum = sqrt(sum / N);
        return sum;
    }

    __host__ __device__
    Real rms_norm() {
        Real sum = zero;
        for (size_t i = 0; i < N; i++) {
            sum += pow((*this)[i], two);
        }
        sum = sqrt(sum / N);
        return sum;
    }

    __host__ __device__
    Real min() {
        Real xmin = (*this)[0];
        for (auto& x : *this) {
            xmin = (x < xmin) ? x : xmin;
        }
        return xmin;
    }

    __host__ __device__
    void sv_linear_combination(size_t num_elements, Real* scalars, RealVector<N>* vectors) {
        // For scalars[i] and vectors[i], i = 0, ..., num_elements-1 :
        // Set this vector rv as: rv = sum_{i = 0, num_elements-1} (scalars[i] * vectors[i])
        for (size_t j = 0; j < N; j++) {
            (*this)[j] = zero;
            for (size_t i = 0; i < num_elements; i++) {
                (*this)[j] += scalars[i] * vectors[i][j];
            }
        }
    }
    
};

#endif
