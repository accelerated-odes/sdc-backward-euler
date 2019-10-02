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
typedef MathVector<long, PARALLEL_SIMT_SIZE, StackCreate<long, PARALLEL_SIMT_SIZE>> BranchIndexType;
#else
typedef int WarpType;
typedef long BranchIndexType;
#endif

template<long Nq = 0, long ThreadsPerGroup = 0> class TaskQueue {

#ifdef __CUDA_ARCH__
  MathVectorSet<long, Nq, PARALLEL_SIMT_SIZE,
                StackCreate<long, PARALLEL_SIMT_SIZE>> simt_branch_indices;
  MathVector<unsigned int, Nq, StackCreate<unsigned int, Nq>> map_branch_indices;
#endif

public:

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  TaskQueue() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ~TaskQueue() {}

  template<long NumberActiveQueues, int NumberOfTasks = -1, typename AnyLambdaSelector, typename AnyLambdaSelectorFinished, typename AnyLambdaTask>
  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void execute(AnyLambdaSelector selector_lambda, AnyLambdaSelectorFinished selector_finished_lambda, AnyLambdaTask tasks_lambda) {
#ifdef __CUDA_ARCH__
    auto thread_block = cg::this_thread_block();

    thread_block.sync();

    cg::thread_block_tile<ThreadsPerGroup> thread_group = cg::tiled_partition<ThreadsPerGroup>(thread_block);

    // figure out which branch to take with this group of threads
    int this_branch_flag = thread_block.thread_rank() / thread_group.size();

    if (this_branch_flag < NumberActiveQueues) {

    /* if (thread_group.thread_rank() == 0) */
    /*     printf("in task queue, looking for branch flag = %d\n", this_branch_flag); */

    if (thread_group.thread_rank() == 0) {
        map_branch_indices[this_branch_flag] = 0;
    }

    for (int widx = 0; widx <= NumberOfTasks; widx++) {
        bool queue_run_ready = (map_branch_indices[this_branch_flag] == PARALLEL_SIMT_SIZE ||
                                (map_branch_indices[this_branch_flag] > 0 && widx == NumberOfTasks));
        /* if (thread_group.thread_rank() == 0) printf("%d: checking widx = %d\n", this_branch_flag, widx); */
        if (queue_run_ready) {
            __threadfence();

            /* if (thread_group.thread_rank() == 0) { */
            /*     printf("%d: Running tasks ...\n", this_branch_flag); */
            /*     for (int i = 0; i < map_branch_indices[this_branch_flag]; i++) { */
            /*         printf("%d: simt_branch_indices[%d][%d] = %d\n", this_branch_flag, this_branch_flag, i, simt_branch_indices[this_branch_flag][i]); */
            /*     } */
            /* } */

            tasks_lambda(thread_group, simt_branch_indices,
                         map_branch_indices[this_branch_flag], this_branch_flag);

            thread_group.sync();

            if (thread_group.thread_rank() == 0)
                map_branch_indices[this_branch_flag] = 0;
        }
        bool queue_add_ready = widx < NumberOfTasks;
        if (thread_group.thread_rank() == 0 && queue_add_ready) {
            /* if (thread_group.thread_rank() == 0) printf("%d: Checking if we can add index %d\n", this_branch_flag, widx); */
            if (selector_lambda(widx) == this_branch_flag) {
                /* if (thread_group.thread_rank() == 0) printf("%d: Adding index %d\n", this_branch_flag, widx); */
                simt_branch_indices[this_branch_flag][map_branch_indices[this_branch_flag]] = widx;
                map_branch_indices[this_branch_flag]++;
                /* if (thread_group.thread_rank() == 0) printf("%d: map_branch_indices[%d] = %d\n", this_branch_flag, this_branch_flag, map_branch_indices[this_branch_flag]); */
            }
        }
    }
    } else {
        /* if (thread_group.thread_rank() == 0) printf("Skipping branch number %d\n", this_branch_flag); */
    }

    thread_block.sync();

    /* if (thread_block.thread_rank() == 0) printf("Done with task queue execution.\n"); */
#else
    VECTOR_LAMBDA(NumberOfTasks,
                  [&](long& widx) {
                    const int this_branch_flag = selector_lambda(widx);
                    int c0 = 0;
                    const int c1 = 0;
                    tasks_lambda(c0, widx, c1, this_branch_flag);
                  });
#endif
  }
};
#endif
