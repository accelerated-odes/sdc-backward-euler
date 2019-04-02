#ifndef _INTEGER_VECTOR_H
#define _INTEGER_VECTOR_H

#include "MathVector.H"

template<size_t N> class IntegerVector : public MathVector<int, N> {
public:
  using MathVector<int, N>::equals;
};

#endif
