#ifndef VECTOR_PARALLEL_UTIL_H
#define VECTOR_PARALLEL_UTIL_H

#include "AMReX_Array.H"
#include "AMReX_Box.H"
#include <cassert>

#ifdef __CUDA_ARCH__
#include "cuda_runtime.h"
#include "AMReX_GpuUtility.H"
using namespace amrex::Gpu::Atomic::detail;
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

void get_tiling(const amrex::Box& bx, const size_t grid_size_per_block,
                amrex::Dim3& kernel_tile_size, amrex::Dim3& kernel_num_blocks) {

  auto box_length = bx.length();

  kernel_tile_size.x = box_length[0];
  kernel_tile_size.y = box_length[1];
  kernel_tile_size.z = box_length[2];

  kernel_num_blocks.x = 1;
  kernel_num_blocks.y = 1;
  kernel_num_blocks.z = 1;

  int total_blocks = 1;

  auto tilelen = [&]()
                 { return static_cast<size_t>(kernel_tile_size.x * kernel_tile_size.y * kernel_tile_size.z); };

  while(true) {
    if ((kernel_tile_size.z % 2) == 0 &&
        box_length[2] % (kernel_tile_size.z / 2) == 0 &&
        tilelen() > grid_size_per_block) {
      kernel_tile_size.z /= 2;
      kernel_num_blocks.z *= 2;
      total_blocks *= 2;
      continue;
    } else if ((kernel_tile_size.y % 2) == 0 &&
               box_length[1] % (kernel_tile_size.y / 2) == 0 &&
               tilelen() > grid_size_per_block) {
      kernel_tile_size.y /= 2;
      kernel_num_blocks.y *= 2;
      total_blocks *= 2;
      continue;
    } else if ((kernel_tile_size.x % 2) == 0 &&
               box_length[0] % (kernel_tile_size.x / 2) == 0 &&
               tilelen() > grid_size_per_block) {
      kernel_tile_size.x /= 2;
      kernel_num_blocks.x *= 2;
      total_blocks *= 2;
      continue;
    } else {
      break;
    }
  }

  assert(tilelen() <= grid_size_per_block);
}

#define PRINT_DIM3(id, d) std::cout << id << " " << d.x << " " << d.y << " " << d.z << "\n"

// Get the number of cells in a Box defined by
// lo and hi Dim3 objects.
AMREX_GPU_HOST_DEVICE AMREX_INLINE
size_t box_number_cells(const amrex::Dim3& lo, const amrex::Dim3& hi) {
  return static_cast<size_t>(  (hi.x - lo.x + 1)
#if AMREX_SPACEDIM >= 2
                             * (hi.y - lo.y + 1)
#endif
#if AMREX_SPACEDIM == 3
                             * (hi.z - lo.z + 1)
#endif
                             );
}

#define PRAGMA_EXPRESSION(x) _Pragma(#x)

// The Vector Lambda macros iterate over the vector elements
// in one or more sets with all the worker threads.

#ifdef __CUDA_ARCH__

// These will be used inside a global kernel so we don't
// need to do anything special to get threads.
#define PARALLEL_REGION

#define PARALLEL_SHARED __shared__

#define PARALLEL_TILE_INDEX_X(ix) blockIdx.x

#define PARALLEL_TILE_INDEX_Y(iy) blockIdx.y

#define PARALLEL_TILE_INDEX_Z(iz) blockIdx.z

#define PARALLEL_SIMT_SIZE 32 // number of threads per warp

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

#define VECTOR_SET_LAMBDA_CG(group, Nset, Nvec, lambda) ({             \
    size_t nstrides = static_cast<size_t>(Nset * Nvec) / group.size(); \
    if ((Nset * Nvec) % group.size() > 0.0) nstrides++;                \
                                                                       \
    for (size_t n = 0; n < nstrides; n++) {                            \
      size_t idx = group.thread_rank() + n * group.size();             \
      if (idx < Nset * Nvec) {                                         \
        size_t iset = idx / Nvec;                                      \
        size_t ivec = idx % Nvec;                                      \
        lambda(iset, ivec);                                            \
      }                                                                \
    }                                                                  \
  })

#define VECTOR_LAMBDA_CG(group, N, lambda) ({              \
    size_t nstrides = static_cast<size_t>(N)/group.size(); \
    if (N % group.size() > 0) nstrides++;                  \
                                                           \
    for (size_t n = 0; n < nstrides; n++) {                \
      size_t idx = group.thread_rank() + n * group.size(); \
      if (idx < N) {                                       \
        lambda(idx);                                       \
      }                                                    \
    }                                                      \
  })

// The WORKER_SYNC just synchronizes the threads in the block.
#define WORKER_SYNC() ({ __syncthreads(); })

// The WORKER_SIMT_SYNC synchronizes the threads in the current warp.
#define WORKER_SIMT_SYNC() ({ __syncwarp(); })

#define SINGLE_LAMBDA(lambda) ({                        \
    if (threadIdx.x == 0)                               \
      lambda();                                         \
  })

#elif defined(_OPENMP)

// Use OpenMP to parallelize vector operations
// this uses static scheduling because it presumes
// we have decomposed the code into simple enough
// pieces that load balancing is unnecessary. We
// could make this an option later ...

#define PARALLEL_REGION PRAGMA_EXPRESSION(omp parallel default(shared))

#define PARALLEL_SHARED

#define PARALLEL_TILE_INDEX_X(ix) ix

#define PARALLEL_TILE_INDEX_Y(iy) iy

#define PARALLEL_TILE_INDEX_Z(iz) iz

#define PARALLEL_SIMT_SIZE 1

#define VECTOR_SET_LAMBDA(Nset, Nvec, lambda) ({        \
    PRAGMA_EXPRESSION(omp for schedule(static) collapse(2))   \
    for (size_t iset = 0; iset < Nset; iset++) {        \
      for (size_t ivec = 0; ivec < Nvec; ivec++) {      \
        lambda(iset, ivec);                             \
      }                                                 \
    }                                                   \
  })

#define VECTOR_LAMBDA(N, lambda) ({                     \
    PRAGMA_EXPRESSION(omp for schedule(static))         \
    for (size_t i = 0; i < N; i++)                      \
      lambda(i);                                        \
  })

#define WORKER_SYNC() ({PRAGMA_EXPRESSION(omp barrier)})

#define WORKER_SIMT_SYNC() ({})

#define SINGLE_LAMBDA(lambda) ({                        \
  if (omp_get_thread_num() == 0)                        \
    lambda();                                           \
  })

#else

// In serial mode these macros reduce
// to serial loops and lambda function calls.

#define PARALLEL_REGION

#define PARALLEL_SHARED

#define PARALLEL_TILE_INDEX_X(ix) ix

#define PARALLEL_TILE_INDEX_Y(iy) iy

#define PARALLEL_TILE_INDEX_Z(iz) iz

#define PARALLEL_SIMT_SIZE 1

#define VECTOR_SET_LAMBDA(num_set, num_vec, lambda) ({  \
    for (size_t iset = 0; iset < num_set; iset++) {     \
      for (size_t ivec = 0; ivec < num_vec; ivec++) {   \
        lambda(iset, ivec);                             \
      }                                                 \
    }                                                   \
  })

#define VECTOR_LAMBDA(N, lambda) ({                     \
    for (size_t i = 0; i < N; i++)                      \
      lambda(i);                                        \
  })

#define WORKER_SYNC() ({})

#define WORKER_SIMT_SYNC() ({})

#define SINGLE_LAMBDA(lambda) ({                        \
    lambda();                                           \
  })
#endif

#endif
