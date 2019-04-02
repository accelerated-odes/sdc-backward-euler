#ifndef SDC_INTEGRATOR_TYPE_H
#define SDC_INTEGRATOR_TYPE_H
#include <iostream>
#include <iomanip>
#include <limits>
#include "math.h"
#include "AMReX_REAL.H"
#include "RealVector.H"
#include "RealSparseMatrix.H"

enum class SdcStatus {Completed, Failed, StartTimestep, StartSDCIteration, StartNode, StartNewton};

template<typename EnumClass> auto enum_class_integer(EnumClass const value) ->
  typename std::underlying_type<EnumClass>::type {
  return static_cast<typename std::underlying_type<EnumClass>::type>(value);
};

template<class SparseLinearSolver, class SystemClass, size_t order> class SdcIntegrator {
  typedef SdcIntegrator<SparseLinearSolver, SystemClass, order> SdcIntClass;
  typedef RealVector<SystemClass::neqs> NReals;
  typedef RealSparseMatrix<SystemClass::neqs, SystemClass::nnz> SMat;
    
  static const size_t number_nodes = order - 1;

  SdcStatus status;
  size_t step_counter, sdc_counter, node_counter, newton_counter;
  size_t maximum_newton, maximum_steps;
  bool fail_if_maximum_newton;

  Real time_previous, time_target, time_step, newton_error, newton_tolerance, small;
  RealVector<number_nodes> time_nodes;

  RealVector<SystemClass::neqs> q_node;

  RealVector<SystemClass::neqs> y_delta;
  RealVector<SystemClass::neqs> implicit_rhs;
  RealSparseMatrix<SystemClass::neqs, SystemClass::nnz> j_new;

  RealVector<SystemClass::neqs> f_new[number_nodes];
  RealVector<SystemClass::neqs> f_old[number_nodes];
  RealVector<SystemClass::neqs> y_old[number_nodes];
  RealVector<SystemClass::neqs> y_new[number_nodes];

public:

  __host__ __device__
  SdcIntegrator() {}

    __host__ __device__    
    static void initialize(SdcIntClass& sdc,
                           RealVector<SystemClass::neqs>& y_init,
                           Real start_time, Real end_time, Real time_step_init = 0.0,
                           Real tolerance = 1.0e-6, size_t max_newton_iters=100,
                           bool fail_if_max_newton = true, 
			   size_t max_steps = 1000000, Real epsilon = 2.5e-16) {
        sdc.y_new[0] = y_init;
        sdc.y_old[0] = y_init;

        sdc.time_previous = start_time;
        RealVector<number_nodes>::equals(sdc.time_nodes, sdc.time_previous);
        sdc.time_target = end_time;
        sdc.time_step = (time_step_init == 0.0) ? (end_time-start_time)/10.0 : time_step_init;

        sdc.step_counter = 0;
        sdc.sdc_counter = 1;
        sdc.node_counter = 1;
        sdc.newton_counter = 0;

        sdc.fail_if_maximum_newton = fail_if_max_newton;
        sdc.maximum_newton = max_newton_iters;
        sdc.newton_tolerance = tolerance;
	sdc.maximum_steps = max_steps;
        sdc.small = epsilon;

        for (size_t i = 0; i < sdc.number_nodes; i++) {
	  NReals::equals(sdc.f_old[i], 0.0);
	  if (i > 0) NReals::equals(sdc.y_old[i], 0.0);
        }

        sdc.status = SdcStatus::StartTimestep;
    }

  __host__ __device__    
  static void set_jacobian_layout(SdcIntClass& sdc, SystemClass& ode_system) {
      RealSparseMatrix<SystemClass::neqs, SystemClass::nnz>::set_csr_layout(sdc.j_new,
                                                                            ode_system.csr_col_index,
                                                                            ode_system.csr_row_count);
  }

  __host__ __device__    
  ~SdcIntegrator() {}

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
  static RealVector<SystemClass::neqs>& get_previous_solution(SdcIntClass& sdc) {
    return sdc.y_new[sdc.node_counter-1];
  }

  __host__ __device__
  static Real get_previous_time(SdcIntClass& sdc) {
    return sdc.time_nodes.data[sdc.node_counter-1];
  }

  __host__ __device__
  static RealVector<SystemClass::neqs>& get_current_solution(SdcIntClass& sdc) {
    return sdc.y_new[sdc.node_counter];
  }

  __host__ __device__
  static Real get_current_time(SdcIntClass& sdc) {
    return sdc.time_nodes.data[sdc.node_counter];
  }

  __host__ __device__
  static void prepare(SdcIntClass& sdc) {
    // Set up the solve
    if (sdc.status == SdcStatus::StartTimestep) {
        SdcIntClass::initialize_nodes(sdc);
    }

    if (sdc.status == SdcStatus::StartSDCIteration) {
        SdcIntClass::save_sdc_iteration(sdc);
    }

    if (sdc.status == SdcStatus::StartTimestep ||
        sdc.status == SdcStatus::StartSDCIteration ||
        sdc.status == SdcStatus::StartNode) {
        SdcIntClass::evaluate_quadrature(sdc);
        SdcIntClass::guess_new_solution(sdc);
        sdc.newton_error = 1.0e30;
    }

    SdcIntClass::evaluate_system(sdc);
    SdcIntClass::setup_linear_system(sdc);
  }

  __host__ __device__
  static void solve(SdcIntClass& sdc) {
      SparseLinearSolver::solve(sdc.j_new.data, sdc.y_delta.data, sdc.implicit_rhs.data);
  }

  __host__ __device__
  static void update(SdcIntClass& sdc) {
      SdcIntClass::update_solution(sdc);
      SdcIntClass::update_status(sdc);
  }

  __host__ __device__
  static void update_solution(SdcIntClass& sdc) {
    // Use the results of the previous solve to check status
    // and update integration state & status

    // We update y_new and update corrector error.
    NReals::add(sdc.y_new[sdc.node_counter], sdc.y_delta);
    Real delta_norm = NReals::frobenius_norm(sdc.y_delta);
    
    sdc.newton_error = delta_norm/(NReals::max_norm(sdc.y_new[sdc.node_counter]) + sdc.small);
  }

  __host__ __device__
  static void update_status(SdcIntClass& sdc) {
    SdcStatus new_status;

    // Now check for Newton stopping conditions and evaluate convergence

    if (sdc.newton_error < sdc.newton_tolerance ||
	(!sdc.fail_if_maximum_newton && sdc.newton_counter == sdc.maximum_newton)) {
      // Newton iteration has finished successfully
      // Check if we are finished with all nodes in this step
      if (sdc.node_counter == sdc.number_nodes-1) {
	// Check if we are finished with all SDC iterations
	if (sdc.sdc_counter == order) {
	  // Check if we have reached the stop time or max timesteps
	  if (fabs(sdc.time_nodes.data[sdc.number_nodes-1] - sdc.time_target) < 2.0*sdc.small) {
	    // We have reached the stop time successfully
          new_status = SdcStatus::Completed;
	  } else {
	    if (sdc.step_counter == sdc.maximum_steps)
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
      if (sdc.newton_counter == sdc.maximum_newton)
	// Newton iteration has failed
	new_status = SdcStatus::Failed;
      else
	// Keep iterating the Newton solver
	new_status = SdcStatus::StartNewton;
    }

    // update counters for the new integration status
    SdcIntClass::set_status(sdc, new_status);
  }

  __host__ __device__
  static void evaluate_system(SdcIntClass& sdc) {
    // Evaluate F and Jac at y_new to set f_new and j_new
    SystemClass::evaluate(sdc.time_nodes.data[sdc.node_counter], sdc.y_new[sdc.node_counter],
                          sdc.f_new[sdc.node_counter], sdc.j_new);
  }

  __host__ __device__
  static void setup_linear_system(SdcIntClass& sdc) {
    // Set up the linear system A*x = b to solve.
    // Puts A into j_new in-place and puts b into implicit_rhs.

    // Form the matrix A:
    SMat::mul(sdc.j_new, sdc.time_nodes.data[sdc.node_counter-1] - sdc.time_nodes.data[sdc.node_counter]);
    for (size_t i = 0; i < SystemClass::neqs; i++) {
        SMat::ij(sdc.j_new, i, i) += 1.0;
    }

    // Form the vector b:
    for (size_t i = 0; i < SystemClass::neqs; i++) {
      sdc.implicit_rhs.data[i] = sdc.y_new[sdc.node_counter-1].data[i] - sdc.y_new[sdc.node_counter].data[i];
      sdc.implicit_rhs.data[i] += (sdc.time_nodes.data[sdc.node_counter] - sdc.time_nodes.data[sdc.node_counter-1]) * (sdc.f_new[sdc.node_counter].data[i] - sdc.f_old[sdc.node_counter].data[i]);
      sdc.implicit_rhs.data[i] += sdc.q_node.data[i];
    }
  }

  __host__ __device__
  static bool is_finished(SdcIntClass& sdc) {
    return (sdc.status == SdcStatus::Completed || sdc.status == SdcStatus::Failed);
  }

  __host__ __device__
  static bool is_started(SdcIntClass& sdc) {
    return (sdc.step_counter > 0 || sdc.sdc_counter > 1 || sdc.node_counter > 1 || sdc.newton_counter > 0);
  }

  __host__ __device__
  static void evaluate_quadrature(SdcIntClass& sdc) {
    // Use high order quadrature rule to set q_node

    // Right now this is hardcoded for simpson integration
    // using evenly spaced nodes.

    if (sdc.node_counter == 1)
      // integral from m = 0 to m = 1
      for (size_t i = 0; i < SystemClass::neqs; i++) {
          sdc.q_node.data[i] = (sdc.time_nodes.data[1] - sdc.time_nodes.data[0])/12.0;
          sdc.q_node.data[i] *= (5.0e0*sdc.f_old[0].data[i] + 8.0e0*sdc.f_old[1].data[i] - sdc.f_old[2].data[i]);
      }
    else
      // integral from m = 1 to m = 2
      for (size_t i = 0; i < SystemClass::neqs; i++) {
          sdc.q_node.data[i] = (sdc.time_nodes.data[1] - sdc.time_nodes.data[0])/12.0;
          sdc.q_node.data[i] *= (-sdc.f_old[0].data[i] + 8.0e0*sdc.f_old[1].data[i] + 5.0e0*sdc.f_old[2].data[i]);
      }
  }

    __host__ __device__
    static void set_status(SdcIntClass& sdc, SdcStatus new_status) {
        sdc.status = new_status;
        if (SdcIntClass::is_finished(sdc)) return;
        
        if (sdc.status == SdcStatus::StartTimestep) {
            // reset SDC, Node, Newton counters
            sdc.sdc_counter = 1;
            sdc.node_counter = 1;
            sdc.newton_counter = 0;

            // increment timestep counter
            sdc.step_counter++;
        } else if (sdc.status == SdcStatus::StartSDCIteration) {
            // reset Node, Newton counters
            sdc.node_counter = 1;
            sdc.newton_counter = 0;

            // increment SDC iteration counter
            sdc.sdc_counter++;
        } else if (sdc.status == SdcStatus::StartNode) {
            // reset Newton counter
            sdc.newton_counter = 0;

            // increment node counter
            sdc.node_counter++;
        } else {
            // increment newton counter for the newton step we just did
            sdc.newton_counter++;
        }
    }

  __host__ __device__
  static void initialize_nodes(SdcIntClass& sdc) {
    // If we are advancing the step, then move y_old forward
    // along with time_previous
    if (sdc.step_counter > 0) {
        sdc.time_previous = sdc.time_nodes.data[sdc.number_nodes-1];
        NReals::equals(sdc.y_old[0], sdc.y_new[sdc.number_nodes-1]);
        NReals::equals(sdc.f_old[0], sdc.f_new[number_nodes-1]);
        NReals::equals(sdc.y_new[0], sdc.y_old[0]);
        NReals::equals(sdc.f_new[0], sdc.f_old[0]);
    } else {
      // Evaluate the system at y_old and set y_old and f_old for nodes
      SystemClass::evaluate(sdc.time_nodes.data[0], sdc.y_old[0], sdc.f_old[0]);
    }

    for (size_t i = 1; i < number_nodes; i++) {
        NReals::equals(sdc.y_old[i], sdc.y_old[0]);
        NReals::equals(sdc.f_old[i], sdc.f_old[0]);
    }

    // Set evenly spaced node times
    Real dt_node = sdc.time_step/(sdc.number_nodes-1);
    for (size_t i = 0; i < sdc.number_nodes; i++) {
      sdc.time_nodes.data[i] = sdc.time_previous + i * dt_node;
    }
  }

  __host__ __device__
  static void save_sdc_iteration(SdcIntClass& sdc) {
    for (size_t i = 1; i < sdc.number_nodes; i++) {
        NReals::equals(sdc.y_old[i], sdc.y_new[i]);
        NReals::equals(sdc.f_old[i], sdc.f_new[i]);
    }
  }

  __host__ __device__
  static void guess_new_solution(SdcIntClass& sdc) {
    if (sdc.sdc_counter > 1)
        NReals::equals(sdc.y_new[sdc.node_counter], sdc.y_old[sdc.node_counter]);
    else
        NReals::equals(sdc.y_new[sdc.node_counter], sdc.y_new[sdc.node_counter-1]);
  }
};
#endif
