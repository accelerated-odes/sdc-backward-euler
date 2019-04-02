#ifndef _REAL_VECTOR_H
#define _REAL_VECTOR_H
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include "AMReX_REAL.H"
#include "AMReX_CONSTANTS.H"
#include "MathVector.H"

using namespace amrex;

template<size_t N> class RealVector : public MathVector<Real,N> {
public:
  using MathVector<Real,N>::begin;
  using MathVector<Real,N>::end;

  virtual void print() {
    size_t i = 0;
    for (const auto& x : (*this)) {
      std::cout << x;
      if (i < N-1) std::cout << " ";
      i++;
    }
  }

  __host__ __device__
  static void abs(RealVector<N>& v) {
    for (auto& x : v) {
      x = fabs(x);
    }
  }

  __host__ __device__
  static void inv(RealVector<N>& v) {
    for (auto& x : v) {
      x = one/x;
    }
  }

  __host__ __device__
  static Real dot(RealVector<N>& x, RealVector<N>& y) {
    Real sum = zero;
    for (size_t i = 0; i < N; i++) {
      sum += x.data[i] * y.data[i];
    }
    return sum;
  }

  __host__ __device__
  static Real max_norm(RealVector<N>& v) {
    Real max = zero;
    for (auto& x : v) {
      max = (fabs(x) > max) ? fabs(x) : max;
    }
    return max;
  }

  __host__ __device__
  static Real wrms_norm(RealVector<N>& v, RealVector<N>& weights) {
    Real sum = zero;
    for (size_t i = 0; i < N; i++) {
      sum += pow(v.data[i] * weights.data[i], two);
    }
    sum = sqrt(sum / N);
    return sum;
  }

  __host__ __device__
  static Real rms_norm(RealVector<N>& v) {
    Real sum = zero;
    for (size_t i = 0; i < N; i++) {
      sum += pow(v.data[i], two);
    }
    sum = sqrt(sum / N);
    return sum;
  }

  __host__ __device__
  static Real frobenius_norm(RealVector<N>& v) {
    Real sum = zero;
    for (size_t i = 0; i < N; i++) {
      sum += pow(v.data[i], two);
    }
    sum = sqrt(sum);
    return sum;
  }

  __host__ __device__
  static Real min(RealVector<N>& v) {
    Real xmin = v.data[0];
    for (auto& x : v) {
      xmin = (x < xmin) ? x : xmin;
    }
    return xmin;
  }
};

#endif
