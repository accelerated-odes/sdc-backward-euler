#ifndef VECTOR_GPU_MACROS_H
#define VECTOR_GPU_MACROS_H

// The Vector Lambda iterates over the vector elements
// in the set with all the threads.
#ifdef __CUDA_ARCH__
// The VECTOR_SET_LAMBDA logically flattens a lambda
// operation across Nset components and Nvec vector elements
// by mapping thread indexes to the flattened index (idx).
// We then get the component (iset) and vector (ivec) indexes
// from the flat index for each thread.
#define VECTOR_SET_LAMBDA(Nset, Nvec, lambda) ({        \
    size_t nstrides = Nset * Nvec / blockDim.x;         \
    if ((Nset * Nvec) % blockDim.x > 0.0) nstrides++;   \
                                                        \
    for (size_t n = 0; n < nstrides; n++) {             \
      size_t idx = threadIdx.x + n * blockDim.x;        \
      if (idx < Nset * Nvec) {                          \
        size_t iset = idx / Nvec;                       \
        size_t ivec = idx % Nvec;                       \
        lambda(iset, ivec);                             \
      }                                                 \
    }                                                   \
  })
// The VECTOR_LAMBDA logically flattens a lambda
// operation across an index space of size N
// by striding through that index space with a block of threads.
#define VECTOR_LAMBDA(N, lambda) ({                     \
    size_t nstrides = N / blockDim.x;                   \
    if (N % blockDim.x > 0.0) nstrides++;               \
                                                        \
    for (size_t n = 0; n < nstrides; n++) {             \
      size_t idx = threadIdx.x + n * blockDim.x;        \
      if (idx < N) {                                    \
        lambda(idx);                                    \
      }                                                 \
    }                                                   \
  })
// The WORKER_SYNC just synchronizes the threads in the block.
#define WORKER_SYNC() ({__syncthreads();})
#define SINGLE_LAMBDA(lambda) ({                        \
    if (threadIdx.x == 0)                               \
      lambda();                                         \
  })
#else
// At the moment, without CUDA, these macros reduce
// to serial loops and lambda function calls.
// We could use OpenMP, alternately.
#define VECTOR_LAMBDA(N, lambda) ({                     \
    for (size_t i = 0; i < N; i++)                      \
      lambda(i);                                        \
  })
#define WORKER_SYNC() ({})
#define SINGLE_LAMBDA(lambda) ({                        \
    lambda();                                           \
  })
#endif
#endif
