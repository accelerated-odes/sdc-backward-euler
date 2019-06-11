#ifndef TASK_QUEUE_H
#define TASK_QUEUE_H

#ifdef AMREX_USE_CUDA
#include <cooperative_groups.h>
#endif

#include "AMReX_GpuQualifiers.H"
#include "AMReX_Extension.H"

#include "MathVectorSet.H"
#include "VectorStorage.H"
#include "VectorParallelUtil.H"

namespace cg = cooperative_groups;

template<size_t Nq, size_t ThreadsPerGroup> class TaskQueue {
  MathVectorSet<size_t, 2, PARALLEL_SIMT_SIZE,
                StackCreate<size_t, PARALLEL_SIMT_SIZE>> simt_branch_indices;
  MathVector<unsigned int, 2, StackCreate<unsigned int, 2>> map_branch_indices;

public:

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  TaskQueue() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ~TaskQueue() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void init() {
    map_branch_indices.init();
    simt_branch_indices.init();
  }

  template<size_t NumberActiveQueues, typename AnyLambdaSelector, typename AnyLambdaSelectorFinished, typename AnyLambdaTask>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE  
  void execute(AnyLambdaSelector selector_lambda, AnyLambdaSelectorFinished selector_finished_lambda, AnyLambdaTask tasks_lambda) {
#ifdef __CUDA_ARCH__
    auto tblock = cg::this_thread_block();

    map_branch_indices = 1;

    tblock.sync();

    // make 2 thread groups.
    cg::thread_block_tile<ThreadsPerGroup> thread_group = cg::tiled_partition<ThreadsPerGroup>(tblock);

    // figure out which branch to take with this group of threads
    int this_branch_flag = tblock.thread_rank() / thread_group.size();

    int working_index = thread_group.thread_rank();

    auto check_loc = [&](int widx)->int {
                       if (selector_finished_lambda(widx)) return -2;

                       if (selector_lambda(this_branch_flag, widx)) {
                         if (map_branch_indices[this_branch_flag] == 0) return widx;
                         unsigned int imap = atomicInc(&map_branch_indices[this_branch_flag], PARALLEL_SIMT_SIZE+1);
                         imap--;
                         simt_branch_indices[this_branch_flag][imap] = widx;
                       }

                       // not finished filling the work queue and not at end of vector
                       return -1;
                     };

    /* // FOR DEBUGGING
       while(true) {
       int next_index = check_loc(working_index);
       if (next_index == -1)
       working_index += thread_group.size();
       else if (next_index == vector_length)
       break;
       }

       thread_group.sync();
       tblock.sync();

       x_initial = 0.0;
       x_final = 0.0;

       tblock.sync();

       x_initial[0][tblock.thread_rank()] = simt_branch_indices[this_branch_flag][thread_group.thread_rank()];
    */

    if (this_branch_flag < NumberActiveQueues) {
      while(true) {
        int next_index = check_loc(working_index);
        if (next_index == -1)
          working_index += thread_group.size();
        else {
          // sync
          thread_group.sync();

          // get the number of entries in the queue
          int queue_fill_size = map_branch_indices[this_branch_flag] - 1;
          if (queue_fill_size < 0) queue_fill_size = PARALLEL_SIMT_SIZE;

          if (queue_fill_size > 0) {
            // do work for queue_fill_size entries
            VECTOR_LAMBDA_CG(thread_group, queue_fill_size,
                             [&](size_t& ii) {
                               size_t sbi = simt_branch_indices[this_branch_flag][ii];
                               tasks_lambda(this_branch_flag, sbi);
                             });

            // reset the queue
            if (thread_group.thread_rank() == 0)
              map_branch_indices[this_branch_flag] = 1;

            // sync threads in this group
            thread_group.sync();
          } else {
            // we didn't have work to do so break
            break;
          }
        }
      }
    }

    tblock.sync();
#endif
  }
};
#endif
