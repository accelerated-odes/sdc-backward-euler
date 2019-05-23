#include <iostream>

#ifdef AMREX_USE_CUDA
#include <cuda_profiler_api.h>
#endif

#include "RealVectorSet.H"
#include "WallTimer.H"

#ifdef AMREX_USE_CUDA
template<size_t vector_set_length, size_t vector_length>
__global__
void do_gpu_kernel(Real* y_initial, Real* y_final, size_t array_max_size) {
  __shared__ RealVectorSet<vector_set_length, vector_length> x_initial, x_final;

  x_initial.load(y_initial, array_max_size);
  x_final = 0.0;
  x_initial += 1.0;
  x_initial *= 2.0;
  x_initial /= 4.0;
  x_initial *= 2.0;
  x_final = x_initial;
  x_final.save(y_final, array_max_size);
}
#endif

template<size_t vector_set_length, size_t vector_length>
void do_cpu_kernel(Real* y_initial, Real* y_final, size_t array_max_size) {
  RealVectorSet<vector_set_length, vector_length> x_initial, x_final;

  x_initial.load(y_initial, array_max_size);
  x_final = 0.0;
  x_initial += 1.0;
  x_initial *= 2.0;
  x_initial /= 4.0;
  x_initial *= 2.0;
  x_final = x_initial;
  x_final.save(y_final, array_max_size);
}

int main(int argc, char* argv[]) {

#ifdef AMREX_USE_CUDA
  cudaProfilerStart();
#endif

  const size_t N = 5120;

  WallTimer timer;

  Real* y_initial;
  Real* y_final;

#ifndef AMREX_USE_CUDA
  y_initial = new Real[N];
  y_final = new Real[N];
#else
  cudaError_t cuda_status = cudaSuccess;
  void* vp;
  cuda_status = cudaMallocManaged(&vp, sizeof(Real) * N);
  assert(cuda_status == cudaSuccess);

  y_initial = static_cast<Real*>(vp);

  cuda_status = cudaMallocManaged(&vp, sizeof(Real) * N);
  assert(cuda_status == cudaSuccess);

  y_final = static_cast<Real*>(vp);
#endif

  // initialize systems
  for (size_t i = 0; i < N; i++) {
    y_initial[i] = 1.0;
  }

  // size of each vector and number of vector sets in a given shared memory structure
  const int nVectorSize = 128;   // use vectors of length 128
  const int nVectorSets = 2;     // each vectorset contains 2 component vectors

  // how many threads per block to use
  const int nThreads = 64;

  // each block operates on a number of elements M in the global array
  // M = nVectorSize * nVectorSets
  // so we need ceil(N/M) threadblocks
  const int nBlocks = static_cast<int>(ceil(((double) N)/(double) nVectorSize * nVectorSets));

  timer.start_wallclock();

#ifndef AMREX_USE_CUDA
  do_cpu_kernel<nVectorSets, N/nVectorSets>(y_initial, y_final, N);
#else
  do_gpu_kernel<nVectorSets, nVectorSize><<<nBlocks, nThreads>>>(y_initial, y_final, N);
  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);
#endif

  timer.stop_wallclock();

  std::cout << std::endl << "Final Vector -------------------" << std::endl;
  Real ysum = 0.0;
  for (size_t i = 0; i < N; i++) {
    std::cout << std::setprecision(std::numeric_limits<Real>::digits10 + 1);
    std::cout << "y_final[" << i << "] = " << y_final[i] << std::endl;
    ysum += y_final[i];
  }

  std::cout << "sum of y: " << ysum << std::endl;

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
