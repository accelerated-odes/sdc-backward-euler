#ifndef SDC_INTEGRATOR_TYPE_H
#define SDC_INTEGRATOR_TYPE_H
#include <iostream>
#include <iomanip>
#include <limits>
#include "math.h"
#include "AMReX_REAL.H"
#include "RealVector.H"
#include "RealSparseMatrixLayout.H"
#include "unified.H"

enum class SdcStatus {Completed, Failed, StartTimestep, StartSDCIteration, StartNode, StartNewton};

template<typename EnumClass> auto enum_class_integer(EnumClass const value) ->
  typename std::underlying_type<EnumClass>::type {
  return static_cast<typename std::underlying_type<EnumClass>::type>(value);
}

template<class SystemClass, size_t order> class SdcIntegratorType {
  static const size_t number_nodes = order - 1;
  static const size_t maximum_steps = 1000000;

  SdcStatus status;
  size_t step_counter, sdc_counter, node_counter, newton_counter;
  size_t maximum_newton;
  bool fail_if_maximum_newton;

  Real time_previous, time_target, time_step, newton_error, newton_tolerance, small;
  RealVector<number_nodes> time_nodes;

  RealVector<SystemClass::neqs> q_node;

  Real* y_delta;
  Real* implicit_rhs;
  RealSparseMatrixLayout<SystemClass::neqs, SystemClass::nnz> j_new;

  RealVector<SystemClass::neqs> f_new[number_nodes];
  RealVector<SystemClass::neqs> f_old[number_nodes];
  RealVector<SystemClass::neqs> y_old[number_nodes];
  RealVector<SystemClass::neqs> y_new[number_nodes];

public:

  SdcIntegratorType() {}

  SdcIntegratorType(SystemClass& ode_system) {
    set_jacobian_layout(ode_system);
  }

  SdcIntegratorType(SystemClass& ode_system, RealVector<SystemClass::neqs>& y_init,
                    Real start_time, Real end_time, Real time_step_init,
                    Real tolerance = 1.0e-6, size_t max_newton_iters=100) {
    set_jacobian_layout(ode_system);
    initialize(y_init, start_time, end_time, time_step_init,
	       tolerance, max_newton_iters);
  }

  void initialize(RealVector<SystemClass::neqs>& y_init,
		  Real start_time, Real end_time, Real time_step_init = 0.0,
		  Real tolerance = 1.0e-6, size_t max_newton_iters=100,
		  bool fail_if_max_newton = true) {
    y_new[0] = y_init;
    y_old[0] = y_init;

    time_previous = start_time;
    time_nodes = time_previous;
    time_target = end_time;
    time_step = (time_step_init == 0.0) ? (end_time-start_time)/10.0 : time_step_init;

    step_counter = 0;
    sdc_counter = 1;
    node_counter = 1;
    newton_counter = 0;

    fail_if_maximum_newton = fail_if_max_newton;
    maximum_newton = max_newton_iters;
    newton_tolerance = tolerance;
    small = std::numeric_limits<Real>::epsilon();

    for (size_t i = 0; i < number_nodes; i++) {
      f_old[i] = 0.0;
      if (i > 0) y_old[i] = 0.0;
    }

    status = SdcStatus::StartTimestep;
  }

  void set_jacobian_layout(SystemClass& ode_system) {
    j_new.set_csr_layout(ode_system.csr_col_index, ode_system.csr_row_count);
  }

  ~SdcIntegratorType() {}

  void print() {
    std::cout << "status: " << enum_class_integer<SdcStatus>(status) << std::endl;

    std::cout << "step_counter: " << step_counter << std::endl;
    std::cout << "sdc_counter: " << sdc_counter << std::endl;
    std::cout << "node_counter: " << node_counter << std::endl;
    std::cout << "newton_counter: " << newton_counter << std::endl;

    std::cout << "time target: " << time_target << std::endl;

    std::cout << "time nodes:" << std::endl;
    time_nodes.print();

    std::cout << std::endl << "y_old:" << std::endl;
    for (size_t i = 0; i < number_nodes; i++) {
      std::cout << i << ": "; y_old[i].print(); std::cout << std::endl;
    }

    std::cout << "f_old:" << std::endl;
    for (size_t i = 0; i < number_nodes; i++) {
      std::cout << i << ": "; f_old[i].print(); std::cout << std::endl;
    }

    std::cout << "y_new:" << std::endl;
    for (size_t i = 0; i < number_nodes; i++) {
      std::cout << i << ": "; y_new[i].print(); std::cout << std::endl;
    }
  }

  __host__ __device__
  RealVector<SystemClass::neqs> get_previous_solution() {
    return y_new[node_counter-1];
  }

  __host__ __device__
  Real get_previous_time() {
    return time_nodes[node_counter-1];
  }

  __host__ __device__
  RealVector<SystemClass::neqs> get_current_solution() {
    return y_new[node_counter];
  }

  __host__ __device__
  Real get_current_time() {
    return time_nodes[node_counter];
  }

  __host__ __device__
  void prepare() {
    // Set up the solve
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

  __host__ __device__
  void update() {
    update_solution();
    update_status();
  }

  __host__ __device__
  void update_solution() {
    // Use the results of the previous solve to check status
    // and update integration state & status

    // We update y_new and update corrector error.
    Real delta_norm = zero;
    for (size_t i = 0; i < SystemClass::neqs; i++) {
      y_new[node_counter][i] += y_delta[i];
      delta_norm += y_delta[i] * y_delta[i];
    }
    delta_norm = sqrt(delta_norm);
    
    newton_error = delta_norm/(y_new[node_counter].max_norm() + small);
  }

  __host__ __device__
  void update_status() {
    SdcStatus new_status;

    // Now check for Newton stopping conditions and evaluate convergence

    if (newton_error < newton_tolerance ||
	(!fail_if_maximum_newton && newton_counter == maximum_newton)) {
      // Newton iteration has finished successfully
      // Check if we are finished with all nodes in this step
      if (node_counter == number_nodes-1) {
	// Check if we are finished with all SDC iterations
	if (sdc_counter == order) {
	  // Check if we have reached the stop time or max timesteps
	  if (fabs(time_nodes[number_nodes-1] - time_target) < 2.0*small) {
	    // We have reached the stop time successfully
	    new_status = SdcStatus::Completed;
	  } else {
	    if (step_counter == maximum_steps)
	      new_status = SdcStatus::Failed;
	    else
	      new_status = SdcStatus::StartTimestep;
	  }
	} else {
	  new_status = SdcStatus::StartSDCIteration;
	}
      } else {
	new_status = SdcStatus::StartNode;
      }
    } else {
      if (newton_counter == maximum_newton)
	// Newton iteration has failed
	new_status = SdcStatus::Failed;
      else
	// Keep iterating the Newton solver
	new_status = SdcStatus::StartNewton;
    }

    // update counters for the new integration status
    set_status(new_status);
  }

  __host__ __device__
  void evaluate_system() {
    // Evaluate F and Jac at y_new to set f_new and j_new
    SystemClass::evaluate(time_nodes[node_counter], y_new[node_counter],
			  f_new[node_counter], j_new);
  }

  __host__ __device__
  void setup_linear_system() {
    // Set up the linear system A*x = b to solve.
    // Puts A into j_new in-place and puts b into implicit_rhs.

    // Form the matrix A:
    j_new * (time_nodes[node_counter-1] - time_nodes[node_counter]);
    for (size_t i = 0; i < SystemClass::neqs; i++) {
      j_new.ij(i,i) += 1.0;
    }

    // Form the vector b:
    for (size_t i = 0; i < SystemClass::neqs; i++) {
      implicit_rhs[i] = y_new[node_counter-1][i] - y_new[node_counter][i];
      implicit_rhs[i] += (time_nodes[node_counter] - time_nodes[node_counter-1]) * (f_new[node_counter][i] - f_old[node_counter][i]);
      implicit_rhs[i] += q_node[i];
    }
  }

  __host__ __device__
  bool is_finished() {
    return (status == SdcStatus::Completed || status == SdcStatus::Failed);
  }

  __host__ __device__
  bool is_started() {
    return (step_counter > 0 || sdc_counter > 1 || node_counter > 1 || newton_counter > 0);
  }

  __host__ __device__
  void evaluate_quadrature() {
    // Use high order quadrature rule to set q_node

    // Right now this is hardcoded for simpson integration
    // using evenly spaced nodes.

    if (node_counter == 1)
      // integral from m = 0 to m = 1
      for (size_t i = 0; i < SystemClass::neqs; i++) {
	q_node[i] = (time_nodes[1] - time_nodes[0])/12.0;
	q_node[i] *= (5.0e0*f_old[0][i] + 8.0e0*f_old[1][i] - f_old[2][i]);
      }
    else
      // integral from m = 1 to m = 2
      for (size_t i = 0; i < SystemClass::neqs; i++) {
	q_node[i] = (time_nodes[1] - time_nodes[0])/12.0;
	q_node[i] *= (-f_old[0][i] + 8.0e0*f_old[1][i] + 5.0e0*f_old[2][i]);
      }
  }

  __host__ __device__
  void set_status(SdcStatus new_status) {
    status = new_status;
    if (status == SdcStatus::StartTimestep) {
      // reset SDC, Node, Newton counters
      sdc_counter = 1;
      node_counter = 1;
      newton_counter = 0;

      // increment timestep counter
      step_counter++;
    } else if (status == SdcStatus::StartSDCIteration) {
      // reset Node, Newton counters
      node_counter = 1;
      newton_counter = 0;

      // increment SDC iteration counter
      sdc_counter++;
    } else if (status == SdcStatus::StartNode) {
      // reset Newton counter
      newton_counter = 0;

      // increment node counter
      node_counter++;
    } else {
      // increment newton counter for the newton step we just did
      newton_counter++;
    }
  }

  __host__ __device__
  void initialize_nodes() {
    // If we are advancing the step, then move y_old forward
    // along with time_previous
    if (step_counter > 0) {
      time_previous = time_nodes[number_nodes-1];
      y_old[0] = y_new[number_nodes-1];
      f_old[0] = f_new[number_nodes-1];
      y_new[0] = y_old[0];
      f_new[0] = f_old[0];
    } else {
      // Evaluate the system at y_old and set y_old and f_old for nodes
      SystemClass::evaluate(time_nodes[0], y_old[0], f_old[0]);
    }

    for (size_t i = 1; i < number_nodes; i++) {
      y_old[i] = y_old[0];
      f_old[i] = f_old[0];
    }

    // Set evenly spaced node times
    Real dt_node = time_step/(number_nodes-1);
    for (size_t i = 0; i < number_nodes; i++) {
      time_nodes[i] = time_previous + i * dt_node;
    }
  }

  __host__ __device__
  void save_sdc_iteration() {
    for (size_t i = 1; i < number_nodes; i++) {
      y_old[i] = y_new[i];
      f_old[i] = f_new[i];
    }
  }

  __host__ __device__
  void guess_new_solution() {
    if (sdc_counter > 1)
      y_new[node_counter] = y_old[node_counter];
    else
      y_new[node_counter] = y_new[node_counter-1];
  }

  __host__ __device__
  void map_linear_system(Real* A, Real* x, Real* b) {
    j_new.map(A);
    y_delta = x;
    implicit_rhs = b;
  }
};
#endif
