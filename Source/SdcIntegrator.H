#ifndef SDC_INTEGRATOR_TYPE_H
#define SDC_INTEGRATOR_TYPE_H
#include <iostream>
#include <iomanip>
#include <limits>
#include "math.h"
#include "AMReX_REAL.H"
#include "RealVector.H"
#include "RealVectorSet.H"
#include "RealSparseMatrix.H"
#include "VectorGpuMacros.H"

#ifndef AMREX_USE_CUDA
using std::min;
using std::max;
#endif

enum class SdcStatus {Completed, Failed, StartTimestep, StartSDCIteration, StartNode, StartNewton};

template<typename EnumClass> auto enum_class_integer(EnumClass const value) ->
  typename std::underlying_type<EnumClass>::type {
  return static_cast<typename std::underlying_type<EnumClass>::type>(value);
};

template<class SparseLinearSolver, class SystemClass, size_t order, size_t vector_length> class SdcIntegrator {
  static const size_t number_nodes = order - 1;

  const Real dt_control_S1 = 0.9;
  const Real dt_control_S2 = 4.0;

  // Scalar state data
  size_t maximum_newton, maximum_steps;
  bool fail_if_maximum_newton;
  bool use_adaptive_timestep;
  Real small, newton_tolerance;
  Real time_target;

  // Vector state data
  SdcStatus status[vector_length];
  MathVector<size_t, vector_length> step_counter, sdc_counter, node_counter, newton_counter;

  RealVector<vector_length> time_previous, time_step, newton_error;
  RealVectorSet<number_nodes, vector_length> time_nodes;
  RealVectorSet<SystemClass::neqs, vector_length> q_node, y_delta, implicit_rhs;

  RealSparseMatrix<SystemClass::neqs, SystemClass::nnz, vector_length> j_new;

  RealVectorSet<SystemClass::neqs, vector_length> f_new[number_nodes];
  RealVectorSet<SystemClass::neqs, vector_length> f_old[number_nodes];
  RealVectorSet<SystemClass::neqs, vector_length> y_old[number_nodes];
  RealVectorSet<SystemClass::neqs, vector_length> y_new[number_nodes];

  // The rule for using scratch space is that it should be treated as if it only
  // has local scope and does not remain valid across function calls.
  // It is useful for evaluating norms and otherwise getting results from
  // component-wise vectorized operations.
  RealVector<vector_length> scratch;

public:

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  SdcIntegrator() {}

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void initialize(RealVectorSet<SystemClass::neqs, vector_length>& y_init,
                  Real start_time, Real end_time, Real time_step_init = 0.0,
                  Real tolerance = 1.0e-6, size_t max_newton_iters=100,
                  bool fail_if_max_newton = true,
                  size_t max_steps = 1000000, Real epsilon = 2.5e-16,
                  bool use_adaptive_timestep = true) {

    SINGLE_LAMBDA([&]() {
      time_previous = start_time;
      time_nodes = time_previous;
      time_target = end_time;
      time_step = (time_step_init == 0.0) ? (end_time-start_time)/10.0 : time_step_init;
      use_adaptive_timestep = use_adaptive_timestep;

      fail_if_maximum_newton = fail_if_max_newton;
      maximum_newton = max_newton_iters;
      newton_tolerance = tolerance;
      maximum_steps = max_steps;
      small = epsilon;

      status = SdcStatus::StartTimestep;
                  });

    y_new[0] = y_init;
    y_old[0] = y_init;

    step_counter = 0;
    sdc_counter = 1;
    node_counter = 1;
    newton_counter = 0;

    for (size_t i = 0; i < number_nodes; i++) {
      f_old[i] = 0.0;
      if (i > 0)
        y_old[i] = 0.0;
    }

    WORKER_SYNC();
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void set_jacobian_layout(SystemClass& ode_system) {
    j_new.set_csr_layout(ode_system.csr_col_index, ode_system.csr_row_count);
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  ~SdcIntegrator() {}

  void print() {
    VECTOR_LAMBDA(vector_length, [&](size_t& k) {
      std::cout << "vector element index: " << k << std::endl;
      std::cout << "status: " << enum_class_integer<SdcStatus>(status[k]) << std::endl;

      std::cout << "step_counter: " << step_counter[k] << std::endl;
      std::cout << "sdc_counter: " << sdc_counter[k] << std::endl;
      std::cout << "node_counter: " << node_counter[k] << std::endl;
      std::cout << "newton_counter: " << newton_counter[k] << std::endl;

      std::cout << "time target: " << time_target << std::endl;

      std::cout << "time nodes:" << std::endl;
      time_nodes[k].print();

      std::cout << std::endl << "y_old:" << std::endl;
      for (size_t i = 0; i < number_nodes; i++) {
        std::cout << i << ": "; y_old[i][k].print(); std::cout << std::endl;
      }

      std::cout << "f_old:" << std::endl;
      for (size_t i = 0; i < number_nodes; i++) {
        std::cout << i << ": "; f_old[i][k].print(); std::cout << std::endl;
      }

      std::cout << "y_new:" << std::endl;
      for (size_t i = 0; i < number_nodes; i++) {
        std::cout << i << ": "; y_new[i][k].print(); std::cout << std::endl;
      }
                                 });
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  RealVectorSet<SystemClass::neqs, vector_length>& get_previous_solution() {
    return y_new[sdc.node_counter-1];
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  RealVector<vector_length>& get_previous_time() {
    return time_nodes[node_counter-1];
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  RealVectorSet<SystemClass::neqs, vector_length>& get_current_solution() {
    return y_new[node_counter];
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  RealVector<vector_length>& get_current_time() {
    return time_nodes[node_counter];
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void prepare() {
    // Set up the solve
    // don -- this is a branch point
    if (status == SdcStatus::StartTimestep) {
      initialize_nodes();
    }

    if (status == SdcStatus::StartSDCIteration) {
      save_sdc_iteration();
    }

    if (status == SdcStatus::StartTimestep ||
        status == SdcStatus::StartSDCIteration ||
        status == SdcStatus::StartNode) {
      evaluate_quadrature();
      guess_new_solution();
      newton_error = 1.0e30;
    }

    evaluate_system();
    setup_linear_system();
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void solve() {
    // don -- need to vectorize this
      SparseLinearSolver::solve(j_new.data, y_delta.data, implicit_rhs.data);
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void update() {
      update_solution();
      update_status();
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void update_solution() {
    // Use the results of the previous solve to check status
    // and update integration state & status

    // We update y_new and update corrector error.
    // newton_error = frobenius_norm(y_delta)/(max_norm(y_new) + small)
    y_new[node_counter] += y_delta;
    y_new[node_counter].max_norm(scratch);
    scratch += small;
    y_delta.frobenius_norm(newton_error);
    newton_error /= scratch;
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void update_status() {

    VECTOR_LAMBDA(vector_length, [&](size_t& k) {
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

    WORKER_SYNC();
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void evaluate_system() {
    // Evaluate F and Jac at y_new to set f_new and j_new
    // don -- vectorize this
    SystemClass::evaluate(time_nodes[node_counter], y_new[node_counter],
                          f_new[node_counter], j_new);
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void setup_linear_system() {
    // Set up the linear system A*x = b to solve.
    // Puts A into j_new in-place and puts b into implicit_rhs.

    // Form the matrix A:
    scratch = time_nodes[node_counter-1];
    scratch -= time_nodes[node_counter];
    j_new *= scratch;
    for (size_t i = 0; i < SystemClass::neqs; i++) {
      j_new(i,i) += 1.0;
    }

    // Form the vector b:
    implicit_rhs = f_new[node_counter];
    implicit_rhs -= f_old[node_counter];
    implicit_rhs *= scratch.negate();
    implicit_rhs += q_node;
    implicit_rhs += y_new[node_counter-1];
    implicit_rhs -= y_new[node_counter];

    WORKER_SYNC();
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  bool is_finished(size_t k) {
    return (status[k] == SdcStatus::Completed || status[k] == SdcStatus::Failed);
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  bool is_started(size_t k) {
    return (step_counter[k] > 0 || sdc_counter[k] > 1 ||
            node_counter[k] > 1 || newton_counter[k] > 0);
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void evaluate_quadrature() {
    // Use high order quadrature rule to set q_node

    // Right now this is hardcoded for simpson integration
    // using evenly spaced nodes.

    if (node_counter == 1)
      // integral from m = 0 to m = 1
      for (size_t i = 0; i < SystemClass::neqs; i++) {
        VECTOR_LAMBDA(vector_length, [&](size_t& k) {
          q_node[i][k] = (time_nodes[1][k] - time_nodes[0][k])/12.0;
          q_node[i][k] *= (5.0e0*f_old[0][i][k] + 8.0e0*f_old[1][i][k] - f_old[2][i][k]);
                                     });
        VECTOR_SYNC();
      }
    else
      // integral from m = 1 to m = 2
      for (size_t i = 0; i < SystemClass::neqs; i++) {
        VECTOR_LAMBDA(vector_length, [&](size_t& k) {
          q_node[i][k] = (time_nodes[1][k] - time_nodes[0][k])/12.0;
          q_node[i][k] *= (-f_old[0][i][k] + 8.0e0*f_old[1][i][k] + 5.0e0*f_old[2][i][k]);
                                     });
        VECTOR_SYNC();
      }
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void initialize_nodes() {
    // If we are advancing the step, then move y_old forward
    // along with time_previous

    // don - actually this is a branch point ... need to fix ...
    if (step_counter > 0) {
        time_previous = time_nodes[number_nodes-1];
        y_old[0] = y_new[number_nodes-1];
        f_old[0] = f_new[number_nodes-1];
        y_new[0] = y_old[0];
        f_new[0] = f_old[0];

        VECTOR_LAMBDA(vector_length, [&](size_t& k) {
          Real tnow = time_nodes[number_nodes-1][k];
          if (use_adaptive_timestep) {
            // Set new timestep for 4th order error control
            // Uses adaptive timestepping from Garcia, Eqs. 3.30, 3.31
            Real dt_est = time_step[k] * pow(fabs(newton_tolerance/newton_error[k]), 0.2);
            time_step[k] = min(min(max(dt_control_S1 * dt_est,
                                       time_step[k]/dt_control_S2),
                                   time_step[k] * dt_control_S2),
                               time_target - tnow);
          } else {
            time_step[k] = min(time_step[k], time_target - tnow);
          }
                                     });
    } else {
      // Evaluate the system at y_old and set y_old and f_old for nodes
      SystemClass::evaluate(time_nodes[0], y_old[0], f_old[0]);
    }

    WORKER_SYNC();

    for (size_t i = 1; i < number_nodes; i++) {
        y_old[i] = y_old[0];
        f_old[i] = f_old[0];
    }

    // Set evenly spaced node times
    for (size_t i = 0; i < number_nodes; i++) {
      time_nodes[i] = time_step;
      time_nodes[i] *= static_cast<Real>(i)/(number_nodes-1);
      time_nodes[i] += time_previous;
    }

    WORKER_SYNC();
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void save_sdc_iteration() {
    for (size_t i = 1; i < number_nodes; i++) {
        y_old[i] = y_new[i];
        f_old[i] = f_new[i];
    }

    WORKER_SYNC();
  }

  AMREX_GPU_HOST_DEVICE AMREX_INLINE
  void guess_new_solution() {
    VECTOR_LAMBDA(vector_length, [&](size_t& k) {
      if (sdc_counter[k] > 1)
        y_new[node_counter[k]][k] = y_old[node_counter[k]][k];
      else
        y_new[node_counter[k]][k] = y_new[node_counter[k]-1][k];
                                 });

    WORKER_SYNC();
  }
};
#endif
