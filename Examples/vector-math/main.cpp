#include <iostream>

#ifdef AMREX_USE_CUDA
#include <cuda_profiler_api.h>
#endif

#include <AMReX_REAL.H>

#include "VectorParallelUtil.H"
#include "VectorStorage.H"
#include "MathVectorSet.H"
#include "WallTimer.H"

using namespace amrex;

template<size_t vector_set_length, size_t vector_length, class StorageType>
#ifdef AMREX_USE_CUDA
__global__
#endif
void doit(Real* y_initial, Real* y_final, size_t array_comp_size,
          int array_chunk_index = -1)

{
  PARALLEL_SHARED MathVectorSet<Real, vector_set_length, vector_length, StorageType> x_initial;
  PARALLEL_SHARED MathVectorSet<Real, vector_set_length, vector_length, StorageType> x_final;

  PARALLEL_REGION
    {
      x_initial.map(y_initial, array_comp_size, array_chunk_index,
                    0, 0, vector_set_length);
      x_final.map(y_final, array_comp_size, array_chunk_index,
                  0, 0, vector_set_length);

      WORKER_SYNC();

      VECTOR_SET_LAMBDA(vector_set_length, vector_length,
                        [&](const size_t& iset, const size_t& ivec) {
                          x_final[iset][ivec] = 0.0;
                          x_initial[iset][ivec] += 1.0;
                          x_initial[iset][ivec] *= 2.0;
                          x_initial[iset][ivec] /= 4.0;
                          x_initial[iset][ivec] *= 2.0;
                          x_final[iset][ivec] = x_initial[iset][ivec];
                        });

      WORKER_SYNC();

      x_final.save(y_final, array_comp_size, array_chunk_index,
                   0, 0, vector_set_length);
    }
}


int main(int argc, char* argv[]) {

#ifdef AMREX_USE_CUDA
  cudaProfilerStart();
#endif

  const int vector_storage_type = 1; // 0 = StackCreate, 1 = HeapWindow

  const int N = 5120;

  WallTimer timer;

  Real* y_initial;
  Real* y_final;

  y_initial = new Real[N];
  y_final = new Real[N];

#ifdef AMREX_USE_CUDA
  cudaError_t cuda_status = cudaSuccess;
  void* vp;
  cuda_status = cudaMalloc(&vp, sizeof(Real) * N);
  assert(cuda_status == cudaSuccess);

  Real* y_initial_d = static_cast<Real*>(vp);

  cuda_status = cudaMalloc(&vp, sizeof(Real) * N);
  assert(cuda_status == cudaSuccess);

  Real* y_final_d = static_cast<Real*>(vp);
#endif

  // initialize systems
  for (size_t i = 0; i < N; i++) {
    y_initial[i] = 1.0;
  }

#ifdef AMREX_USE_CUDA
  cuda_status = cudaMemcpy(y_initial_d, y_initial, sizeof(Real) * N,
                           cudaMemcpyHostToDevice);
  assert(cuda_status == cudaSuccess);
#endif

  // size of each vector and number of vector sets in a given shared memory structure
  const int nVectorSize = 128;   // use vectors of length 128
  const int nVectorSets = 10;      // each vectorset contains 2 component vectors

  // how many threads per block to use for CUDA
  const int nThreads = 64;

  // each block operates on a number of elements M in the global array
  // M = nVectorSize * nVectorSets
  // so we need ceil(N/M) threadblocks
  const int nBlocks = static_cast<int>(ceil(((double) N)/((double) nVectorSize * nVectorSets)));

  const size_t size_per_component = static_cast<size_t>(N/nVectorSets);

  timer.start_wallclock();

#ifdef AMREX_USE_CUDA
  if (vector_storage_type == 0) {
    doit<nVectorSets, nVectorSize,
         StackCreate<Real, nVectorSize>><<<nBlocks, nThreads>>>(y_initial_d, y_final_d, size_per_component);

  } else {
    doit<nVectorSets, nVectorSize,
         HeapWindow<Real>><<<nBlocks, nThreads>>>(y_initial_d, y_final_d, size_per_component);
  }
  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);
#else
  for (int i = 0; i < nBlocks; i++) {
    if (vector_storage_type == 0) {
      doit<nVectorSets, nVectorSize,
           StackCreate<Real, nVectorSize>>(y_initial, y_final, size_per_component, i);
    } else {
      doit<nVectorSets, nVectorSize,
           HeapWindow<Real>>(y_initial, y_final, size_per_component, i);
    }
  }
#endif

  timer.stop_wallclock();

#ifdef AMREX_USE_CUDA
  cuda_status = cudaMemcpy(y_final, y_final_d, sizeof(Real) * N,
                           cudaMemcpyDeviceToHost);
  assert(cuda_status == cudaSuccess);
#endif

  //  std::cout << std::endl << "Final Vector -------------------" << std::endl;
  Real ysum = 0.0;
  for (size_t i = 0; i < N; i++) {
    //std::cout << std::setprecision(std::numeric_limits<Real>::digits10 + 1);
    //  std::cout << "y_final[" << i << "] = " << y_final[i] << std::endl;
    ysum += y_final[i];
  }

  std::cout << "sum of y: " << ysum << std::endl;

  if (ysum == N * 2.0)
    std::cout << "Success!" << std::endl;

#ifndef AMREX_USE_CUDA
  std::cout << "Finished execution on host CPU" << std::endl;
#else
  std::cout << "Finished execution on device" << std::endl;
#endif

  std::cout << std::endl << "walltime (s): " << timer.get_walltime() << std::endl;

#ifndef AMREX_USE_CUDA
  delete[] y_initial;
  delete[] y_final;
#else
  cuda_status = cudaFree(y_initial);
  assert(cuda_status == cudaSuccess);
  cuda_status = cudaFree(y_final);
  assert(cuda_status == cudaSuccess);

  cudaProfilerStop();
#endif

  return 0;
}
