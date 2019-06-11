#include <iostream>
#include <iomanip>

#ifdef AMREX_USE_CUDA
#include <cuda_profiler_api.h>
#endif

#include <AMReX_REAL.H>

#include "VectorParallelUtil.H"
#include "VectorStorage.H"
#include "MathVectorSet.H"
#include "TaskQueue.H"
#include "WallTimer.H"

using namespace amrex;


#ifdef AMREX_USE_CUDA
template<int threads_per_block, size_t vector_set_length, size_t vector_length, class StorageType>
__global__
#else
template<int threads_per_block, size_t vector_set_length, size_t vector_length, class StorageType>
#endif
void index_test(Real* y_initial, Real* y_final, size_t array_comp_size,
                int array_chunk_index = -1)
{
  PARALLEL_SHARED MathVectorSet<Real, vector_set_length, vector_length, StorageType> x_initial;
  PARALLEL_SHARED MathVectorSet<Real, vector_set_length, vector_length, StorageType> x_final;

  PARALLEL_SHARED MathVectorSet<size_t, 2, PARALLEL_SIMT_SIZE,
                                StackCreate<size_t, PARALLEL_SIMT_SIZE>> simt_branch_indices;
  PARALLEL_SHARED MathVector<unsigned int, 2, StackCreate<unsigned int, 2>> map_branch_indices;

  const int threads_per_group = threads_per_block/2;

  PARALLEL_REGION
    {
      x_initial.map(y_initial, array_comp_size, array_chunk_index,
                    0, 0, vector_set_length);
      x_final.map(y_final, array_comp_size, array_chunk_index,
                  0, 0, vector_set_length);

      map_branch_indices.init();
      simt_branch_indices.init();

      WORKER_SYNC();

      x_final = -1.0;

      // even indexes are 1, odd indexes are 2
      simt_branch_indices = vector_length; // this indicates no mapping
      map_branch_indices = 1; // start at the beginning of the vector. Use 1-based index so we can check for the 0 stored by atomicInc.
      WORKER_SYNC();

      auto tblock = cg::this_thread_block();
      // x_final[0][tblock.thread_rank()] = tblock.thread_rank();

      // we will have 1 branch point with 2 branches so make 2 thread groups.
      cg::thread_block_tile<threads_per_group> thread_group = cg::tiled_partition<threads_per_group>(tblock);

      // figure out which branch to take with this group of threads
      int this_branch_flag = (tblock.thread_rank() < thread_group.size()) ? 0 : 1;

      int working_index = thread_group.thread_rank();

      x_final[0][tblock.thread_rank()] = tblock.thread_rank() * 10000 + thread_group.thread_rank() * 100 + this_branch_flag + 10000000;

      x_final.save(y_final, array_comp_size, array_chunk_index,
                   0, 0, vector_set_length);
    }
}


#ifdef AMREX_USE_CUDA
template<int threads_per_block, size_t vector_set_length, size_t vector_length, class StorageType>
__global__
#else
template<int threads_per_block, size_t vector_set_length, size_t vector_length, class StorageType>
#endif
void doit(Real* y_initial, Real* y_final, size_t array_comp_size,
          int array_chunk_index = -1)
{ 
  PARALLEL_SHARED MathVectorSet<Real, vector_set_length, vector_length, StorageType> x_initial;
  PARALLEL_SHARED MathVectorSet<Real, vector_set_length, vector_length, StorageType> x_final;

  const int threads_per_group = threads_per_block/2;
  PARALLEL_SHARED TaskQueue<2, threads_per_group> task_queue;

  PARALLEL_REGION
    {
      x_initial.map(y_initial, array_comp_size, array_chunk_index,
                    0, 0, vector_set_length);
      x_final.map(y_final, array_comp_size, array_chunk_index,
                  0, 0, vector_set_length);

      task_queue.init();

      WORKER_SYNC();

      auto branch_selector = [&](int& this_branch_flag, int& widx)->bool {
                               return ((x_initial[0][widx] == 1.0 && this_branch_flag == 0) ||
                                       (x_initial[0][widx] == 2.0 && this_branch_flag == 1));
                             };

      auto branch_selector_finished = [&](int& widx)->bool {
                                        return (widx >= vector_length);
                                      };

      auto branch_tasks = [&](int& this_branch_flag, size_t& sbi) {
                            if (this_branch_flag == 0)
                              x_initial[0][sbi] = 10.0;
                            else
                              x_initial[0][sbi] = 20.0;
                          };

      task_queue.execute<2>(branch_selector, branch_selector_finished, branch_tasks);

      x_final = x_initial;

      x_final.save(y_final, array_comp_size, array_chunk_index,
                   0, 0, vector_set_length);
    }
}

int main(int argc, char* argv[]) {

#ifdef AMREX_USE_CUDA
  cudaProfilerStart();
#endif

  const int N = 128;

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
  Real correct_ysum = 0.0;
  for (size_t i = 0; i < N; i++) {
    y_initial[i] = (i % 2 == 0) ? 1.0 : 2.0;
    correct_ysum += y_initial[i] * 10.0;
  }

#ifdef AMREX_USE_CUDA
  cuda_status = cudaMemcpy(y_initial_d, y_initial, sizeof(Real) * N,
                           cudaMemcpyHostToDevice);
  assert(cuda_status == cudaSuccess);
#endif

  // size of each vector and number of vector sets in a given shared memory structure
  const int nVectorSize = 64;   // use vectors of length 64
  const int nVectorSets = 1;    // each vectorset contains 1 component vector

  // how many threads per block to use for CUDA
  const int nThreads = 32;

  // each block operates on a number of elements M in the global array
  // M = nVectorSize * nVectorSets
  // so we need ceil(N/M) threadblocks
  const int nBlocks = static_cast<int>(ceil(((double) N)/((double) nVectorSize * nVectorSets)));
  //const int nBlocks = 1;

  //const size_t size_per_component = static_cast<size_t>(N/nVectorSets);
  const size_t size_per_component = N;

  timer.start_wallclock();

#ifdef AMREX_USE_CUDA
  doit<nThreads, nVectorSets, nVectorSize,
       //StackCreate<Real, nVectorSize>><<<nBlocks, nThreads>>>(y_initial_d, y_final_d, size_per_component);
       HeapWindow<Real>><<<nBlocks, nThreads>>>(y_initial_d, y_final_d, size_per_component);
  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);
#else
  for (int i = 0; i < nBlocks; i++) {
    doit<1,nVectorSets, nVectorSize,
         //StackCreate<Real, nVectorSize>>(y_initial, y_final, size_per_component, i);
         HeapWindow<Real>>(y_initial, y_final, size_per_component, i);
  }
#endif

  timer.stop_wallclock();

#ifdef AMREX_USE_CUDA
  cuda_status = cudaMemcpy(y_final, y_final_d, sizeof(Real) * N,
                           cudaMemcpyDeviceToHost);
  assert(cuda_status == cudaSuccess);
  cuda_status = cudaMemcpy(y_initial, y_initial_d, sizeof(Real) * N,
                           cudaMemcpyDeviceToHost);
  assert(cuda_status == cudaSuccess);  
#endif

  //  std::cout << std::endl << "Final Vector -------------------" << std::endl;
  for (size_t i = 0; i < N; i++) {
    std::cout << std::setprecision(std::numeric_limits<Real>::digits10 + 1);
    std::cout << "y_initial[" << i << "] = " << y_initial[i] << std::endl;
  }
  
  Real ysum = 0.0;
  for (size_t i = 0; i < N; i++) {
    std::cout << std::setprecision(std::numeric_limits<Real>::digits10 + 1);
    std::cout << "y_final[" << i << "] = " << y_final[i] << std::endl;
    ysum += y_final[i];
  }  

  std::cout << "sum of y: " << ysum << std::endl;

  if (ysum == correct_ysum)
    std::cout << "Success!" << std::endl;

#ifndef AMREX_USE_CUDA
  std::cout << "Finished execution on host CPU" << std::endl;
#else
  std::cout << "Finished execution on device" << std::endl;
#endif

  std::cout << std::endl << "walltime (s): " << timer.get_walltime() << std::endl;

  delete[] y_initial;
  delete[] y_final;

#ifdef AMREX_USE_CUDA  
  cuda_status = cudaFree(y_initial_d);
  assert(cuda_status == cudaSuccess);
  cuda_status = cudaFree(y_final_d);
  assert(cuda_status == cudaSuccess);

  cudaProfilerStop();
#endif

  return 0;
}
