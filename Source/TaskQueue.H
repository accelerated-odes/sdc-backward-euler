#ifndef TASK_QUEUE_H
#define TASK_QUEUE_H

#include "AMReX_GpuQualifiers.H"
#include "AMReX_Extension.H"

#include "MathVectorSet.H"
#include "VectorStorage.H"
#include "VectorParallelUtil.H"

#ifdef AMREX_USE_CUDA
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

typedef cg::thread_block_tile<PARALLEL_SIMT_SIZE> WarpType;
typedef MathVector<size_t, PARALLEL_SIMT_SIZE, StackCreate<size_t, PARALLEL_SIMT_SIZE>> BranchIndexType;
#endif

template<size_t Nq = 0, size_t ThreadsPerGroup = 0> class TaskQueue {

#ifdef __CUDA_ARCH__
  MathVectorSet<size_t, Nq, PARALLEL_SIMT_SIZE,
                StackCreate<size_t, PARALLEL_SIMT_SIZE>> simt_branch_indices;
  MathVector<unsigned int, Nq, StackCreate<unsigned int, Nq>> map_branch_indices;
#endif

public:

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  TaskQueue() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ~TaskQueue() {}

  template<size_t NumberActiveQueues, int MaximumIndex = -1, typename AnyLambdaSelector, typename AnyLambdaSelectorFinished, typename AnyLambdaTask>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void execute(AnyLambdaSelector selector_lambda, AnyLambdaSelectorFinished selector_finished_lambda, AnyLambdaTask tasks_lambda) {
#ifdef __CUDA_ARCH__
    auto thread_block = cg::this_thread_block();

    map_branch_indices = 1;

    thread_block.sync();

    cg::thread_block_tile<ThreadsPerGroup> thread_group = cg::tiled_partition<ThreadsPerGroup>(thread_block);

    // figure out which branch to take with this group of threads
    int this_branch_flag = thread_block.thread_rank() / thread_group.size();

    int working_index = thread_group.thread_rank();

    auto check_loc = [&](int widx)->int {
                       if (selector_finished_lambda(widx) ||
                           (MaximumIndex >= 0 && widx >= MaximumIndex)) return -2;

                       if (selector_lambda(widx) == this_branch_flag) {
                         if (map_branch_indices[this_branch_flag] == 0) return widx;
                         unsigned int imap = atomicInc(&map_branch_indices[this_branch_flag], PARALLEL_SIMT_SIZE+1);
                         imap--;
                         simt_branch_indices[this_branch_flag][imap] = widx;
                       }

                       // not finished filling the work queue and not at end of vector
                       return -1;
                     };

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
            tasks_lambda(thread_group, simt_branch_indices[this_branch_flag], queue_fill_size, this_branch_flag);

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

    thread_block.sync();
#else
    VECTOR_LAMBDA(MaximumIndex,
                  [&](size_t& widx) {
                    int this_branch_flag = selector_lambda(widx);
                    tasks_lambda(this_branch_flag, widx);
                  });
#endif
  }
};
#endif
