#ifndef SDC_INTEGRATOR_TYPE_H
#define SDC_INTEGRATOR_TYPE_H
#include <iostream>
#include <iomanip>
#include <limits>
#include "math.h"
#include "AMReX_REAL.H"
#include "AMReX_GpuUtility.H"
#include "AMReX_Print.H"

#include "MathVector.H"
#include "MathVectorSet.H"
#include "RealSparseMatrixSet.H"
#include "VectorStorage.H"
#include "VectorParallelUtil.H"
#include "TaskQueue.H"


enum class SdcStatus {Completed, Failed, StartIntegrator, StartTimestep,
                      StartSDCIteration, StartNode, StartNewton};

template<typename EnumClass>
AMREX_GPU_DEVICE AMREX_INLINE
auto enum_class_integer(EnumClass const value) ->
typename std::underlying_type<EnumClass>::type {
  return static_cast<typename std::underlying_type<EnumClass>::type>(value);
};

template<typename SparseLinearSolver, typename SystemClass, long order, long vector_length,
         template<typename, long> typename InternalStorageType> class SdcIntegrator {
  static const long number_nodes = order - 1;

  const Real dt_control_S1 = 0.9;
  const Real dt_control_S2 = 4.0;

  // Scalar state data
  long maximum_newton, maximum_steps;
  bool fail_if_maximum_newton;
  bool use_adaptive_timestep;
  Real small, newton_tolerance;
  Real time_target;

  // Track system active status and indices
  unsigned int num_active_systems;
  MathVector<long, vector_length, InternalStorageType<long, vector_length>> active_systems;

  // Vector state data
  MathVector<SdcStatus, vector_length, InternalStorageType<SdcStatus, vector_length>> status;
  MathVector<long, vector_length, InternalStorageType<long, vector_length>> step_counter, sdc_counter, node_counter, newton_counter;

  MathVector<Real, vector_length, InternalStorageType<Real, vector_length>> time_previous, time_step, newton_error;
  MathVectorSet<Real, number_nodes, vector_length, InternalStorageType<Real, vector_length>> time_nodes;
  MathVectorSet<Real, vector_length, SystemClass::neqs, InternalStorageType<Real, SystemClass::neqs>> q_node, y_delta, implicit_rhs;

  RealSparseMatrixSet<SystemClass::neqs, vector_length, SystemClass::nnz, InternalStorageType> j_new;

  MathVectorSet<Real, vector_length, SystemClass::neqs, InternalStorageType<Real, SystemClass::neqs>> f_new[number_nodes];
  MathVectorSet<Real, vector_length, SystemClass::neqs, InternalStorageType<Real, SystemClass::neqs>> f_old[number_nodes];
  MathVectorSet<Real, vector_length, SystemClass::neqs, InternalStorageType<Real, SystemClass::neqs>> y_old[number_nodes];
  MathVectorSet<Real, vector_length, SystemClass::neqs, InternalStorageType<Real, SystemClass::neqs>> y_new[number_nodes];

  // The rule for using scratch space is that it should be treated as if it only
  // has local scope and does not retain memory across function calls.
  // It is useful for evaluating norms and otherwise getting results from
  // component-wise vectorized operations.
  MathVector<Real, vector_length, InternalStorageType<Real, vector_length>> scratch;

#ifdef AMREX_USE_CUDA
  TaskQueue<2, PARALLEL_SIMT_SIZE> task_queue;
#else
  TaskQueue<2> task_queue;
#endif

public:

  AMREX_GPU_DEVICE AMREX_INLINE
  SdcIntegrator() {}

  template<typename AnyStorageType>
  AMREX_GPU_DEVICE AMREX_INLINE
  void initialize(SystemClass& ode_system,
                  MathVectorSet<Real, SystemClass::neqs, vector_length, AnyStorageType>& y_init,
                  Real start_time, Real end_time, Real time_step_init = 0.0,
                  Real tolerance = 1.0e-6, long max_newton_iters=100,
                  bool fail_if_max_newton = true,
                  long max_steps = 1000000, Real epsilon = 2.5e-16,
                  bool adaptive_timestep = true,
                  long initial_num_systems = vector_length) {

    SINGLE_LAMBDA([&]() {
                    time_target = end_time;
                    use_adaptive_timestep = adaptive_timestep;

                    fail_if_maximum_newton = fail_if_max_newton;
                    maximum_newton = max_newton_iters;
                    newton_tolerance = tolerance;
                    maximum_steps = max_steps;
                    small = epsilon;

                    num_active_systems = initial_num_systems;

                    active_systems.resize();
                    status.resize();
                    step_counter.resize();
                    sdc_counter.resize();
                    node_counter.resize();
                    newton_counter.resize();
                    time_previous.resize();
                    time_step.resize();
                    newton_error.resize();
                    time_nodes.resize();
                    q_node.resize();
                    y_delta.resize();
                    implicit_rhs.resize();
                    j_new.resize();
                    scratch.resize();

                    for (int i = 0; i < number_nodes; i++) {
                      f_new[i].resize();
                      f_old[i].resize();
                      y_new[i].resize();
                      y_old[i].resize();
                    }
                  });

    WORKER_SYNC();

    status = SdcStatus::StartIntegrator;
    time_step = (time_step_init == 0.0) ? (end_time-start_time)/10.0 : time_step_init;

    VECTOR_LAMBDA(vector_length,
                  [&](const long& i) {
                    if (i < num_active_systems)
                      active_systems[i] = i;
                  });

    set_jacobian_layout(ode_system);

    time_previous = start_time;
    time_nodes = time_previous;

    // initialize y_new and y_old for all nodes. Not actually necessary, but makes valgrind happy for printing the full state...
    for (int i = 0; i < number_nodes; i++) {
        VECTOR_SET_LAMBDA(vector_length, SystemClass::neqs,
                         [&](const long& ivec, const long& ieq) {
                           y_new[i][ivec][ieq] = y_init[ieq][ivec];
                           y_old[i][ivec][ieq] = y_init[ieq][ivec];
                         });
    }

    VECTOR_SET_LAMBDA(vector_length, SystemClass::neqs,
                      [&](const long& ivec, const long& ieq) {
                        y_new[0][ivec][ieq] = y_init[ieq][ivec];
                        y_old[0][ivec][ieq] = y_init[ieq][ivec];
                      });

    step_counter = 0;
    sdc_counter = 1;
    node_counter = 0; // 0 for initialize_start, then we set to 1 to do the first node
    newton_counter = 0;

    for (long i = 0; i < number_nodes; i++) {
      f_old[i] = 0.0;
    }

    for (long i = 1; i < number_nodes; i++) {
      y_old[i] = 0.0;
    }

    WORKER_SYNC();
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  void integrate() {
    for (long i = 0; i < maximum_steps; i++) {
      prepare();
      solve();
      update();
      if (integrator_finished()) break;
    }
  }

  template<typename AnyStorageType>
  AMREX_GPU_DEVICE AMREX_INLINE
  void save_current_solution(MathVectorSet<Real, SystemClass::neqs, vector_length, AnyStorageType> ysave) {
    VECTOR_SET_LAMBDA(vector_length, SystemClass::neqs,
                      [&](const long& ivec, const long& ieq) {
                        auto ysol = y_new[node_counter[ivec]][ivec];

                        ysave[ieq][ivec] = ysol[ieq];
                      });

    WORKER_SYNC();
  }

  template<typename AnyStorageType>
  AMREX_GPU_DEVICE AMREX_INLINE
  void save_current_ydelta(MathVectorSet<Real, SystemClass::neqs, vector_length, AnyStorageType> ysave) {
    VECTOR_SET_LAMBDA(vector_length, SystemClass::neqs,
                      [&](const long& ivec, const long& ieq) {
                        ysave[ieq][ivec] = y_delta[ivec][ieq];
                      });

    WORKER_SYNC();
  }

  template<typename AnyStorageType>
  AMREX_GPU_DEVICE AMREX_INLINE
  void save_current_jac(MathVectorSet<Real, SystemClass::neqs, vector_length, AnyStorageType> ysave, const long ijac) {
    VECTOR_SET_LAMBDA(vector_length, SystemClass::neqs,
                      [&](const long& ivec, const long& ieq) {
                        ysave[ieq][ivec] = j_new[ivec][ijac];
                      });

    WORKER_SYNC();
  }

  template<typename AnyStorageType>
  AMREX_GPU_DEVICE AMREX_INLINE
  void save_current_scratch(MathVectorSet<Real, SystemClass::neqs, vector_length, AnyStorageType> ysave) {
    VECTOR_SET_LAMBDA(vector_length, SystemClass::neqs,
                      [&](const long& ivec, const long& ieq) {
                        ysave[ieq][ivec] = scratch[ivec];
                      });

    WORKER_SYNC();
  }

  template<typename AnyStorageType>
  AMREX_GPU_DEVICE AMREX_INLINE
  void save_current_status(MathVectorSet<Real, SystemClass::neqs, vector_length, AnyStorageType> ysave) {
    VECTOR_SET_LAMBDA(vector_length, SystemClass::neqs,
                      [&](const long& ivec, const long& ieq) {
                        ysave[ieq][ivec] = static_cast<amrex::Real>(enum_class_integer(status[ivec]));
                      });

    WORKER_SYNC();
  }

  template<typename AnyStorageType>
  AMREX_GPU_DEVICE AMREX_INLINE
  void save_current_node_counter(MathVectorSet<Real, SystemClass::neqs, vector_length, AnyStorageType> ysave) {
    VECTOR_SET_LAMBDA(vector_length, SystemClass::neqs,
                      [&](const long& ivec, const long& ieq) {
                        ysave[ieq][ivec] = static_cast<amrex::Real>(node_counter[ivec]);
                      });

    WORKER_SYNC();
  }

  template<typename AnyStorageType>
  AMREX_GPU_DEVICE AMREX_INLINE
  void save_current_node_times(MathVectorSet<Real, SystemClass::neqs, vector_length, AnyStorageType> ysave) {
    VECTOR_SET_LAMBDA(vector_length, SystemClass::neqs,
                      [&](const long& ivec, const long& ieq) {
                        ysave[ieq][ivec] = time_nodes[ieq][ivec];
                      });

    WORKER_SYNC();
  }

  template<typename AnyStorageType>
  AMREX_GPU_DEVICE AMREX_INLINE
  void save_current_rhs(MathVectorSet<Real, SystemClass::neqs, vector_length, AnyStorageType> ysave) {
    VECTOR_SET_LAMBDA(vector_length, SystemClass::neqs,
                      [&](const long& ivec, const long& ieq) {
                        auto rhs = f_new[node_counter[ivec]][ivec];

                        ysave[ieq][ivec] = rhs[ieq];
                      });

    WORKER_SYNC();
  }

  template<typename AnyStorageType>
  AMREX_GPU_DEVICE AMREX_INLINE
  void save_current_implicit_rhs(MathVectorSet<Real, SystemClass::neqs, vector_length, AnyStorageType> ysave) {
    VECTOR_SET_LAMBDA(vector_length, SystemClass::neqs,
                      [&](const long& ivec, const long& ieq) {
                        ysave[ieq][ivec] = implicit_rhs[ivec][ieq];
                      });

    WORKER_SYNC();
  }

  template<typename AnyStorageType>
  AMREX_GPU_DEVICE AMREX_INLINE
  void save_current_jac(MathVectorSet<Real, SystemClass::nnz, vector_length, AnyStorageType> ysave) {
    VECTOR_SET_LAMBDA(vector_length, SystemClass::nnz,
                      [&](const long& ivec, const long& inz) {
                        auto jac = j_new[ivec];

                        ysave[inz][ivec] = jac[inz];
                      });

    WORKER_SYNC();
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  void set_jacobian_layout(SystemClass& ode_system) {
    j_new.set_csr_layout(ode_system.csr_col_index, ode_system.csr_row_count);
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  ~SdcIntegrator() {}

  // auto check_todo = [&](const long& i)->bool {
  //                     if (i >= vector_length) return false;
  //                     else if (status[i] != SdcStatus::Completed &&
  //                              status[i] != SdcStatus::Failed) return true;
  //                     else return false;
  //                   };

  // auto node_map = [&](const long& i) constexpr -> long& { return node_counter[i]; };

  // auto active_map = [&](const long& i) constexpr -> long& { return active_indices[i]; };

  void print() {
    VECTOR_LAMBDA(vector_length,
                  [&](long& k) {
                    std::cout << "vector element index: " << k << std::endl;
                    std::cout << "status: " << enum_class_integer<SdcStatus>(status[k]);
                    std::cout << std::endl;

                    std::cout << "step_counter: " << step_counter[k] << std::endl;
                    std::cout << "sdc_counter: " << sdc_counter[k] << std::endl;
                    std::cout << "node_counter: " << node_counter[k] << std::endl;
                    std::cout << "newton_counter: " << newton_counter[k] << std::endl;

                    std::cout << "time target: " << time_target << std::endl;

                    std::cout << "time nodes:" << std::endl;
                    for (long i = 0; i < number_nodes; i++)
                        std::cout << time_nodes[i][k] << " ";

                    std::cout << std::endl << "y_old:" << std::endl;
                    for (long i = 0; i < number_nodes; i++) {
                      std::cout << i << ": ";
                      for (long j = 0; j < SystemClass::neqs; j++)
                        std::cout << y_old[i][k][j] << " ";
                      std::cout << std::endl;
                    }

                    std::cout << "f_old:" << std::endl;
                    for (long i = 0; i < number_nodes; i++) {
                      std::cout << i << ": ";
                      for (long j = 0; j < SystemClass::neqs; j++)
                        std::cout << f_old[i][k][j] << " ";
                      std::cout << std::endl;
                    }

                    std::cout << "y_new:" << std::endl;
                    for (long i = 0; i < number_nodes; i++) {
                      std::cout << i << ": ";
                      for (long j = 0; j < SystemClass::neqs; j++)
                        std::cout << y_new[i][k][j] << " ";
                      std::cout << std::endl;
                    }
                  });
  }

  // We store SoA format, but each vector will have a different sdc.node_counter
  // so there is no single object we can return ...
  // ... commented until I can think of a solution ...
  //
  // AMREX_GPU_DEVICE AMREX_INLINE
  // RealVectorSet<SystemClass::neqs, vector_length>& get_previous_solution() {
  //   return y_new[sdc.node_counter-1];
  // }

  // AMREX_GPU_DEVICE AMREX_INLINE
  // RealVector<vector_length>& get_previous_time() {
  //   return time_nodes[node_counter-1];
  // }

  // don -- need to fix this ...
  // AMREX_GPU_DEVICE AMREX_INLINE
  // RealVectorSet<SystemClass::neqs, vector_length>& get_current_solution() {
  //   return y_new[node_counter];
  // }

  // AMREX_GPU_DEVICE AMREX_INLINE
  // RealVector<vector_length>& get_current_time() {
  //   return time_nodes[node_counter];
  // }

  AMREX_GPU_DEVICE AMREX_INLINE
  void prepare() {
    // Set up the solve

    if (status[0] == SdcStatus::StartIntegrator) {
      initialize_start();

      node_counter = 1;
      WORKER_SYNC();

      initialize_all_nodes();
    }

    {
      auto branch_selector = [&](const int& i)->int {
                               if (status[i] == SdcStatus::StartTimestep) return 0;
                               else if (status[i] == SdcStatus::StartSDCIteration) return 1;
                               else return -1;
                             };

      auto branch_finished = [&](const int& i)->bool {
                               return (i >= vector_length);
                             };

      auto branch_tasks = [&](WarpType thread_group, BranchIndexType& map,
                              const int& map_size, const int& branch) {
                            if (branch == 0) {
                              initialize_step(thread_group, map, map_size);
                              initialize_nodes(thread_group, map, map_size);
                            } else if (branch == 1) {
                              save_sdc_iteration(thread_group, map, map_size);
                            }
                          };

      task_queue.execute<2, vector_length>(branch_selector, branch_finished, branch_tasks);
    }

    {
      auto branch_selector = [&](const int& i)->int {
                               if (status[i] == SdcStatus::StartIntegrator ||
                                   status[i] == SdcStatus::StartTimestep ||
                                   status[i] == SdcStatus::StartSDCIteration ||
                                   status[i] == SdcStatus::StartNode) return 0;
                               else return -1;
                             };

      auto branch_finished = [&](const int& i)->bool {
                               return (i >= vector_length);
                             };

      auto branch_tasks = [&](WarpType thread_group, BranchIndexType& map,
                              const int& map_size, const int& branch) {
                            if (branch == 0) {
                              evaluate_quadrature(thread_group, map, map_size);
                              guess_new_solution(thread_group, map, map_size);

                              VECTOR_LAMBDA_CG(thread_group, map, map_size,
                                               [&](const long& k) {
                                                 newton_error[k] = 1.0e30;
                                               });
                            }
                          };

      task_queue.execute<1, vector_length>(branch_selector, branch_finished, branch_tasks);
    }

    evaluate_system();
    VECTOR_LAMBDA(num_active_systems,
            [&](const long& iactive) {
                const long k = active_systems[iactive];
                if (status[k] != SdcStatus::Completed &&
                    status[k] != SdcStatus::Failed) {
                    scratch[k] = (time_nodes[node_counter[k]-1][k] - time_nodes[node_counter[k]][k]);
                    for (long inz = 0; inz < SystemClass::nnz; inz++)
                        j_new[k][inz] = 0.0;
                        /* j_new[k][inz] = j_new[k][inz] * scratch[k]; */
                }
            });

    WORKER_SYNC();
    setup_linear_system();
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  void solve() {
    SparseLinearSolver::solve(j_new, y_delta, implicit_rhs, active_systems, num_active_systems);
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  void update() {
      update_solution();
      update_status();
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  void update_solution() {
    // Use the results of the previous solve to check status
    // and update integration state & status

    // We update y_new and update corrector error.
    // generically, newton_error = frobenius_norm(y_delta)/(max_norm(y_new) + small)
    {
      auto branch_selector = [&](const int& i)->int {
                               if (status[i] != SdcStatus::Completed &&
                                   status[i] != SdcStatus::Failed) return 0;
                               else return -1;
                             };

      auto branch_finished = [&](const int& i)->bool {
                               return (i >= vector_length);
                             };

      auto branch_tasks = [&](WarpType thread_group, BranchIndexType& map,
                              const int& map_size, const int& branch) {
                            if (branch == 0) {
                              VECTOR_SET_LAMBDA_CG(thread_group, map, map_size, SystemClass::neqs,
                                                   [&](const long& k, const long& ieq) {
                                                     y_new[node_counter[k]][k][ieq] += y_delta[k][ieq];
                                               });
                            }
                          };

      task_queue.execute<1, vector_length>(branch_selector, branch_finished, branch_tasks);
    }

    for (long k = 0; k < vector_length; k++) {
      if (status[k] != SdcStatus::Completed &&
          status[k] != SdcStatus::Failed) {
        y_new[node_counter[k]][k].max_norm(scratch[k]);
        y_delta[k].frobenius_norm(newton_error[k]);
      }
    }

    {
      auto branch_selector = [&](const int& i)->int {
                               if (status[i] != SdcStatus::Completed &&
                                   status[i] != SdcStatus::Failed) return 0;
                               else return -1;
                             };

      auto branch_finished = [&](const int& i)->bool {
                               return (i >= vector_length);
                             };

      auto branch_tasks = [&](WarpType thread_group, BranchIndexType& map,
                              const int& map_size, const int& branch) {
                            if (branch == 0) {
                              VECTOR_LAMBDA_CG(thread_group, map, map_size,
                                               [&](const long& k) {
                                                 scratch[k] += small;
                                                 newton_error[k] /= scratch[k];
                                               });
                            }
                          };

      task_queue.execute<1, vector_length>(branch_selector, branch_finished, branch_tasks);
    }
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  void update_status() {

    auto branch_selector = [&](const int& i)->int {
                             if (status[i] != SdcStatus::Completed &&
                                 status[i] != SdcStatus::Failed) return 0;
                             else return -1;
                           };

    auto branch_finished = [&](const int& i)->bool {
                             return (i >= vector_length);
                           };

    auto branch_tasks = [&](WarpType thread_group, BranchIndexType& map,
                            const int& map_size, const int& branch) {
                          if (branch == 0) {

                            VECTOR_LAMBDA_CG(thread_group, map, map_size,
                                             [&](const long& k) {
                                               // Now check for Newton stopping conditions and evaluate convergence
                                               if (newton_error[k] < newton_tolerance ||
                                                   (!fail_if_maximum_newton && newton_counter[k] == maximum_newton)) {
                                                 // Newton iteration has finished successfully
                                                 // Check if we are finished with all nodes in this step
                                                 if (node_counter[k] == number_nodes-1) {
                                                   // Check if we are finished with all SDC iterations
                                                   if (sdc_counter[k] == order) {
                                                     // Check if we have reached the stop time or max timesteps
                                                     if (fabs(time_nodes[number_nodes-1][k] - time_target) < 2.0*small) {
                                                       // We have reached the stop time successfully
                                                       status[k] = SdcStatus::Completed;
                                                     } else {
                                                       if (step_counter[k] == maximum_steps)
                                                         status[k] = SdcStatus::Failed;
                                                       else
                                                         status[k] = SdcStatus::StartTimestep;
                                                     }
                                                   } else {
                                                     status[k] = SdcStatus::StartSDCIteration;
                                                   }
                                                 } else {
                                                   status[k] = SdcStatus::StartNode;
                                                 }
                                               } else {
                                                 if (newton_counter[k] == maximum_newton)
                                                   // Newton iteration has failed
                                                   status[k] = SdcStatus::Failed;
                                                 else
                                                   // Keep iterating the Newton solver
                                                   status[k] = SdcStatus::StartNewton;
                                               }

                                               // update counters for the new integration status
                                               if (!is_finished(k)) {

                                                 if (status[k] == SdcStatus::StartTimestep) {
                                                   // reset SDC, Node, Newton counters
                                                   sdc_counter[k] = 1;
                                                   node_counter[k] = 1;
                                                   newton_counter[k] = 0;

                                                   // increment timestep counter
                                                   step_counter[k]++;
                                                 } else if (status[k] == SdcStatus::StartSDCIteration) {
                                                   // reset Node, Newton counters
                                                   node_counter[k] = 1;
                                                   newton_counter[k] = 0;

                                                   // increment SDC iteration counter
                                                   sdc_counter[k]++;
                                                 } else if (status[k] == SdcStatus::StartNode) {
                                                   // reset Newton counter
                                                   newton_counter[k] = 0;

                                                   // increment node counter
                                                   node_counter[k]++;
                                                 } else {
                                                   // increment newton counter for the newton step we just did
                                                   newton_counter[k]++;
                                                 }

                                               }
                                             });
                          }
                        };

    task_queue.execute<1, vector_length>(branch_selector, branch_finished, branch_tasks);
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  void evaluate_system() {
    // Evaluate F and Jac at y_new to set f_new at the current node and j_new
    SystemClass::evaluate(time_nodes, y_new, f_new, j_new, node_counter, active_systems, num_active_systems);
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  void setup_linear_system() {
    // Set up the linear system A*x = b to solve.
    // Puts A into j_new in-place and puts b into implicit_rhs.

    WORKER_SYNC();

    // Form the matrices A and vectors b
    auto branch_selector = [&](const int& i)->int {
                             if (status[i] != SdcStatus::Completed &&
                                 status[i] != SdcStatus::Failed) return 0;
                             else return -1;
                           };

    auto branch_finished = [&](const int& i)->bool {
                             return (i >= vector_length);
                           };

    auto branch_tasks = [&](WarpType thread_group, BranchIndexType& map,
                            const int& map_size, const int& branch) {
                          if (branch == 0) {
                            /* VECTOR_LAMBDA_CG(thread_group, map, map_size, */
                            /*                  [&](const long& k) { */
                            /*                    scratch[k] = (time_nodes[node_counter[k]-1][k] - time_nodes[node_counter[k]][k]); */
                            /*                  }); */

                            /* WORKER_CG_SYNC(thread_group); */

                            /* VECTOR_LAMBDA_CG(thread_group, map, map_size, */
                            /*             [&](const long& k) { */
                            /*             for (long inz = 0; inz < SystemClass::nnz; inz++) { */
                            /*                        /1* amrex::Real jn = j_new[k][inz]; *1/ */
                            /*                        /1* amrex::Real sk = scratch[k]; *1/ */
                            /*                        /1* j_new[k][inz] = j_new[k][inz] * scratch[k]; *1/ */
                            /*                        /1* j_new[k][inz] = j_new[k][inz]; *1/ */
                            /*                        /1* j_new[k][inz] = scratch[k]; *1/ */
                            /*                        /1* j_new[k][inz] = 1.0; *1/ */
                            /*                        /1* j_new[k][inz] = sk; *1/ */
                            /*                        /1* j_new[k][inz] = jn; *1/ */
                            /*                        /1* j_new[k][inz] = jn * sk; *1/ */
                            /*                        /1* j_new[k][inz] = thread_group.thread_rank() + 1000*threadIdx.x; *1/ */
                            /*                        j_new[k][inz] += 1; */
                            /*                        } */
                                                 /* }); */

                              if (thread_group.thread_rank() == 0)
                                printf("Branch 0, starting VECTOR_SET_LAMBDA_CG with map_size = %d, map=%d %d %d %d\n", map_size, map[0], map[1], map[2], map[3]);

                            VECTOR_SET_LAMBDA_CG(thread_group, map, map_size, SystemClass::nnz,
                                                 [&](int k, int inz) {
                                                   /* amrex::Real jn = j_new[k][inz]; */
                                                   /* amrex::Real sk = scratch[k]; */
                                                   /* j_new[k][inz] = j_new[k][inz] * scratch[k]; */
                                                   /* j_new[k][inz] = j_new[k][inz]; */
                                                   /* j_new[k][inz] = scratch[k]; */
                                                   /* j_new[k][inz] = 1.0; */
                                                   /* j_new[k][inz] = sk; */
                                                   /* j_new[k][inz] = jn; */
                                                   /* j_new[k][inz] = jn * sk; */
                                                   /* j_new[k][inz] = thread_group.thread_rank() + 1000*threadIdx.x; */
                                                   j_new[k][inz] += 1;
                                                 });

                            /* WORKER_CG_SYNC(thread_group); */

                            /* VECTOR_SET_LAMBDA_CG(thread_group, map, map_size, SystemClass::neqs, */
                            /*                      [&](const long& k, const long& ieq) { */
                            /*                        /1* j_new(ieq, ieq, k) += 1.0; *1/ */

                            /*                        // Form the vectors b: */
                            /*                        implicit_rhs[k][ieq] = f_new[node_counter[k]][k][ieq]; */
                            /*                        implicit_rhs[k][ieq] -= f_old[node_counter[k]][k][ieq]; */
                            /*                        implicit_rhs[k][ieq] *= -scratch[k]; */
                            /*                        implicit_rhs[k][ieq] += q_node[k][ieq]; */
                            /*                        implicit_rhs[k][ieq] += y_new[node_counter[k]-1][k][ieq]; */
                            /*                        implicit_rhs[k][ieq] -= y_new[node_counter[k]][k][ieq]; */
                            /*                      }); */
                          }
                        };

    task_queue.execute<1, vector_length>(branch_selector, branch_finished, branch_tasks);
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  bool integrator_finished() {
    // return finished status for all systems
    SINGLE_LAMBDA([&]() {
                    num_active_systems = 0;
                  });

    WORKER_SYNC();

    // warning: not OMP threadsafe!!!
    VECTOR_LAMBDA(vector_length,
                  [&](const long& i) {
                    if (!is_finished(i)) {
                      unsigned int isys = Gpu::Atomic::Inc(&num_active_systems,
                                                            static_cast<unsigned int>(vector_length+1));
                      active_systems[isys] = i;
                    }
                  });

    WORKER_SYNC();

    return (num_active_systems == 0);
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  bool is_finished(const long k) {
    // return finished status for vector index k
    return (status[k] == SdcStatus::Completed || status[k] == SdcStatus::Failed);
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  bool is_started(long k) {
    return (step_counter[k] > 0 || sdc_counter[k] > 1 ||
            node_counter[k] > 1 || newton_counter[k] > 0);
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  void evaluate_quadrature(WarpType thread_group, BranchIndexType& map, const int& map_size) {
    // Use high order quadrature rule to set q_node

    // Right now this is hardcoded for simpson integration
    // using evenly spaced nodes.
    VECTOR_SET_LAMBDA_CG(thread_group, map, map_size, SystemClass::neqs,
                         [&](const long& k, const long& ieq) {
                           if (node_counter[k] == 1) {
                             // integral from m = 0 to m = 1
                             q_node[k][ieq] = (time_nodes[1][k] - time_nodes[0][k])/12.0;
                             q_node[k][ieq] *= (5.0e0*f_old[0][k][ieq] +
                                                8.0e0*f_old[1][k][ieq] -
                                                f_old[2][k][ieq]);
                           } else {
                             // integral from m = 1 to m = 2
                             q_node[k][ieq] = (time_nodes[1][k] - time_nodes[0][k])/12.0;
                             q_node[k][ieq] *= (-f_old[0][k][ieq] +
                                                8.0e0*f_old[1][k][ieq] +
                                                5.0e0*f_old[2][k][ieq]);
                           }
                         });

    WORKER_CG_SYNC(thread_group);
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  void initialize_step(WarpType thread_group, BranchIndexType& map, const int& map_size) {
    // If we are advancing the step, then move y_old forward
    // along with time_previous

    VECTOR_LAMBDA_CG(thread_group, map, map_size,
                     [&](const long& k) {
                       time_previous[k] = time_nodes[number_nodes-1][k];
                     });

    VECTOR_SET_LAMBDA_CG(thread_group, map, map_size, SystemClass::neqs,
                         [&](const long& k, const long& ieq) {
                           y_old[0][k][ieq] = y_new[number_nodes-1][k][ieq];
                           f_old[0][k][ieq] = f_new[number_nodes-1][k][ieq];
                           y_new[0][k][ieq] = y_old[0][k][ieq];
                           f_new[0][k][ieq] = f_old[0][k][ieq];
                         });

    VECTOR_LAMBDA_CG(thread_group, map, map_size,
                     [&](const long& k) {
                       Real tnow = time_nodes[number_nodes-1][k];
                       if (use_adaptive_timestep) {
                         // Set new timestep for 4th order error control
                         // Uses adaptive timestepping from Garcia, Eqs. 3.30, 3.31
                         Real dt_est = time_step[k] * pow(fabs(newton_tolerance/newton_error[k]), 0.2);
                         time_step[k] = amrex::min(amrex::min(amrex::max(dt_control_S1 * dt_est,
                                                    time_step[k]/dt_control_S2),
                                                time_step[k] * dt_control_S2),
                                            time_target - tnow);
                       } else {
                         time_step[k] = amrex::min(time_step[k], time_target - tnow);
                       }
                     });

    WORKER_CG_SYNC(thread_group);
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  void initialize_start() {
    // Evaluate the system at y_old and set y_old and f_old for nodes 0
    SystemClass::evaluate(time_nodes, y_old, f_old, node_counter, active_systems, num_active_systems);
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  void initialize_all_nodes() {
    // Set old y, f to their old values at node 0.
    for (long i = 1; i < number_nodes; i++) {
      VECTOR_SET_LAMBDA(vector_length, SystemClass::neqs,
                        [&](const long& k, const long& ieq) {
                          y_old[i][k][ieq] = y_old[0][k][ieq];
                          f_old[i][k][ieq] = f_old[0][k][ieq];
                        });
    }

    // Set evenly spaced node times
    for (long i = 0; i < number_nodes; i++) {
      VECTOR_LAMBDA(vector_length,
                    [&](const long& k) {
                      time_nodes[i][k] = time_step[k];
                      time_nodes[i][k] *= static_cast<Real>(i)/(number_nodes-1);
                      time_nodes[i][k] += time_previous[k];
                    });
    }

    WORKER_SYNC();
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  void initialize_nodes(WarpType thread_group, BranchIndexType& map, const int& map_size) {
    // Set old y, f to their old values at node 0.
    for (long i = 1; i < number_nodes; i++) {
      VECTOR_SET_LAMBDA_CG(thread_group, map, map_size, SystemClass::neqs,
                           [&](const long& k, const long& ieq) {
                             y_old[i][k][ieq] = y_old[0][k][ieq];
                             f_old[i][k][ieq] = f_old[0][k][ieq];
                           });
    }

    // Set evenly spaced node times
    for (long i = 0; i < number_nodes; i++) {
      VECTOR_LAMBDA_CG(thread_group, map, map_size,
                       [&](const long& k) {
                         time_nodes[i][k] = time_step[k];
                         time_nodes[i][k] *= static_cast<Real>(i)/(number_nodes-1);
                         time_nodes[i][k] += time_previous[k];
                       });
    }

    WORKER_CG_SYNC(thread_group);
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  void save_sdc_iteration(WarpType thread_group, BranchIndexType& map, const int& map_size) {
    for (long i = 1; i < number_nodes; i++) {
      VECTOR_SET_LAMBDA_CG(thread_group, map, map_size, SystemClass::neqs,
                           [&](const long& k, const long& ieq) {
                             y_old[i][k][ieq] = y_new[i][k][ieq];
                             f_old[i][k][ieq] = f_new[i][k][ieq];
                           });
    }

    WORKER_CG_SYNC(thread_group);
  }

  AMREX_GPU_DEVICE AMREX_INLINE
  void guess_new_solution(WarpType thread_group, BranchIndexType& map, const int& map_size) {
    VECTOR_SET_LAMBDA_CG(thread_group, map, map_size, SystemClass::neqs,
                         [&](const long& k, const long& ieq) {
                           if (sdc_counter[k] > 1)
                             y_new[node_counter[k]][k][ieq] = y_old[node_counter[k]][k][ieq];
                           else
                             y_new[node_counter[k]][k][ieq] = y_new[node_counter[k]-1][k][ieq];
                         });

    WORKER_CG_SYNC(thread_group);
  }
};
#endif
