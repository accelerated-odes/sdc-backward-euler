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
template<int threads_per_block, long vector_set_length, long vector_length, class StorageType>
__global__
#else
template<long vector_set_length, long vector_length, class StorageType>
#endif
void doit(Real* y_initial, Real* y_final, long array_comp_size,
          int array_chunk_index = -1)
{
  PARALLEL_SHARED MathVectorSet<Real, vector_set_length, vector_length, StorageType> x_initial;
  PARALLEL_SHARED MathVectorSet<Real, vector_set_length, vector_length, StorageType> x_final;

  const long number_branches = 3;

#ifdef AMREX_USE_CUDA
  const long threads_per_group = PARALLEL_SIMT_SIZE;
  PARALLEL_SHARED TaskQueue<number_branches, threads_per_group> task_queue;
#else
  PARALLEL_SHARED TaskQueue<number_branches> task_queue;
#endif

  PARALLEL_REGION
    {
      x_initial.map(y_initial, array_comp_size, array_chunk_index,
                    0, 0, vector_set_length);
      x_final.map(y_final, array_comp_size, array_chunk_index,
                  0, 0, vector_set_length);

      WORKER_SYNC();

      auto branch_selector = [&](const int& widx)->int {
                               if (x_initial[widx][0] == 1.0) return 0;
                               else if (x_initial[widx][0] == 2.0) return 1;
                               else if (x_initial[widx][0] == 3.0) return 2;
                               else return -1;
                             };

      auto branch_selector_finished = [&](const int& widx)->bool {
                                        return (widx >= vector_set_length);
                                      };

      auto branch_tasks = [&](cg::thread_block_tile<threads_per_group> thread_group,
                              MathVectorSet<long, number_branches, PARALLEL_SIMT_SIZE, StackCreate<long, PARALLEL_SIMT_SIZE>> & simt_indices,
                              const int& queue_fill_size, const int& this_branch_flag) {
                            if (this_branch_flag == 0) {
                                if (thread_group.thread_rank() == 0) {
                                    printf("Branch %d, starting VECTOR_SET_LAMBDA_CG with map_size = %d\n", this_branch_flag, queue_fill_size);
                                    for (int i = 0; i < queue_fill_size; i++)
                                        printf("simt_indices[%d][%d] = %d\n", this_branch_flag, i, simt_indices[this_branch_flag][i]);
                                }
                                thread_group.sync();

                                VECTOR_SET_LAMBDA_CG(thread_group, simt_indices[this_branch_flag], queue_fill_size, vector_length,
                                        [&](int isimt, int iset) {
                                        x_final[isimt][iset] += 10.0 * x_initial[isimt][iset];
                                        });
                            } else if (this_branch_flag == 1) {
                                if (thread_group.thread_rank() == 0) {
                                    printf("Branch %d, starting VECTOR_SET_LAMBDA_CG with map_size = %d\n", this_branch_flag, queue_fill_size);
                                    for (int i = 0; i < queue_fill_size; i++)
                                        printf("simt_indices[%d][%d] = %d\n", this_branch_flag, i, simt_indices[this_branch_flag][i]);
                                }
                                thread_group.sync();

                                VECTOR_SET_LAMBDA_CG(thread_group, simt_indices[this_branch_flag], queue_fill_size, vector_length,
                                        [&](int isimt, int iset) {
                                        x_final[isimt][iset] += 20.0 * x_initial[isimt][iset];
                                        });
                            } else if (this_branch_flag == 2) {
                                if (thread_group.thread_rank() == 0) {
                                    printf("Branch %d, starting VECTOR_SET_LAMBDA_CG with map_size = %d\n", this_branch_flag, queue_fill_size);
                                    for (int i = 0; i < queue_fill_size; i++)
                                        printf("simt_indices[%d][%d] = %d\n", this_branch_flag, i, simt_indices[this_branch_flag][i]);
                                }
                                thread_group.sync();

                                VECTOR_SET_LAMBDA_CG(thread_group, simt_indices[this_branch_flag], queue_fill_size, vector_length,
                                        [&](int isimt, int iset) {
                                        x_final[isimt][iset] += 30.0 * x_initial[isimt][iset];
                                        });
                            }
                          };

      task_queue.execute<number_branches, vector_set_length>(branch_selector, branch_selector_finished, branch_tasks);

      x_final.save(y_final, array_comp_size, array_chunk_index,
                   0, 0, vector_set_length);
    }
}

int main(int argc, char* argv[]) {

#ifdef AMREX_USE_CUDA
  cudaProfilerStart();
#endif

  const int N = 96;

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
  for (long i = 0; i < N; i++) {
      if (i < N/3)
        y_initial[i] = 1.0;
      else if (i < 2*N/3)
        y_initial[i] = 2.0;
      else
        y_initial[i] = 3.0;
    y_final[i] = 0.0;
    correct_ysum += y_final[i] + 10.0 * y_initial[i] * y_initial[i];
  }

#ifdef AMREX_USE_CUDA
  cuda_status = cudaMemcpy(y_initial_d, y_initial, sizeof(Real) * N,
                           cudaMemcpyHostToDevice);
  assert(cuda_status == cudaSuccess);
#endif

  // size of each vector and number of vector sets in a given shared memory structure
  const int nVectorSize = 32;   // use vectors of length 32
  const int nVectorSets = 3;    // each vectorset contains 3 components

  // how many threads per block to use for CUDA
  const int nThreads = 4 * 32;

  // each block operates on a number of elements M in the global array
  // M = nVectorSize * nVectorSets
  // so we need ceil(N/M) threadblocks
  const int nBlocks = static_cast<int>(ceil(((double) N)/((double) nVectorSize * nVectorSets)));

  //const long size_per_component = static_cast<long>(N/nVectorSets);
  const long size_per_component = N/3;

  timer.start_wallclock();

#ifdef AMREX_USE_CUDA
  doit<nThreads, nVectorSets, nVectorSize,
       //StackCreate<Real, nVectorSize>><<<nBlocks, nThreads>>>(y_initial_d, y_final_d, size_per_component);
       HeapWindow<Real>><<<nBlocks, nThreads>>>(y_initial_d, y_final_d, size_per_component);
  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);
#else
  for (int i = 0; i < nBlocks; i++) {
    doit<nVectorSets, nVectorSize,
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
  for (long i = 0; i < N; i++) {
    std::cout << std::setprecision(std::numeric_limits<Real>::digits10 + 1);
    std::cout << "y_initial[" << i << "] = " << y_initial[i] << std::endl;
  }

  Real ysum = 0.0;
  for (long i = 0; i < N; i++) {
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
