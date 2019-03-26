#ifndef SDC_INTEGRATOR_TYPE_H
#define SDC_INTEGRATOR_TYPE_H

#include "AMReX_REAL.H"
#include "RealVector.H"
#include "RealSparseMatrix.H"
#include "unified.H"

enum class SdcStatus {Completed, Failed, StartTimestep, StartSDCIteration, StartNode, StartNewton};

template<class SystemClass, size_t order> class SdcIntegratorType : public UnifiedMemoryClass {
    static const size_t number_nodes = order - 1;
    static const size_t maximum_steps = 1000000;
    static const size_t maximum_newton = 100;
    
    SdcStatus status;
    size_t step_counter, sdc_counter, node_counter, newton_counter;
    
    Real time_previous, time_target, time_step, newton_error, newton_tolerance, small;
    RealVector<number_nodes> time_nodes;
    
    RealVector<SystemClass::neqs> f_new, q_node;
    RealVector<SystemClass::neqs> f_old[number_nodes];
    RealVector<SystemClass::neqs> y_old[number_nodes];
    RealVector<SystemClass::neqs> y_new[number_nodes];
    RealSparseMatrix<SystemClass::neqs, SystemClass::nnz> j_new;

public:

    SdcIntegratorType() {}

    SdcIntegratorType(RealVector<SystemClass::neqs>& y_init, Real time, Real time_step_init, SystemClass& ode_system, Real tolerance = 1.0e-6) {
        y_new[0] = y_init;

        time_previous = 0.0;
        time_nodes = time_previous;
        time_target = time;
        time_step = time_step_init;

        step_counter = 0;        
        sdc_counter = 1;
        node_counter = 1;
        newton_counter = 0;

        newton_tolerance = tolerance;
        small = 1.0e-100;

        j_new.set_csr_layout(ode_system.csr_col_index,
                             ode_system.csr_row_count);
        j_new = 0.0;

        for (size_t i = 0; i < number_nodes; i++) {
            f_old[i] = 0.0;
            y_old[i] = 0.0;
        }

        status = SdcStatus::StartTimestep;
    }    

    ~SdcIntegratorType() {}

    RealVector<SystemClass::neqs> get_current_solution() {
        return y_new[node_counter];
    }

    Real get_current_time() {
        return time_nodes[node_counter];
    }

    void prepare() {
        // Set up the solve
        if (status == SdcStatus::StartTimestep) {
            initialize_nodes();
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

    void update() {
        update_solution();
        update_status();
    }

    void update_solution() {
        // Use the results of the previous solve to check status
        // and update integration state & status

        // The linear solve will have stored y_delta in f_new
        // to save memory so we update y_new and update corrector error.

        y_new[node_counter] + f_new;
        newton_error = f_new.rms_norm()/(y_new[node_counter].max_norm() + small);
    }

    void update_status() {
        SdcStatus new_status;

        // Now check for Newton stopping conditions and evaluate convergence

        if (newton_error < newton_tolerance) {
            // Newton iteration has finished successfully
            // Check if we are finished with all nodes in this step
            if (node_counter == number_nodes-1) {
                // Check if we are finished with all SDC iterations
                if (sdc_counter == order) {
                    // Check if we have reached the stop time or max timesteps
                    if ((time_nodes[number_nodes-1] - time_target) < small) {
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

    void evaluate_system() {
        // Evaluate F and Jac at y_new to set f_new and j_new
        SystemClass::evaluate(time_nodes[node_counter], y_new[node_counter], f_new, j_new);
    }

    void setup_linear_system() {
        // Set up the linear system A*x = b to solve.
        // Puts A into j_new and puts b into f_new.

        // Form the matrix A:
        j_new * (time_nodes[node_counter-1] - time_nodes[node_counter]);
        for (size_t i = 0; i < SystemClass::neqs; i++) {
            j_new.ij(i,i) += 1.0;
        }

        // Form the vector b:
        for (size_t i = 0; i < SystemClass::neqs; i++) {
            f_new[i] = (y_new[node_counter-1][i] - y_new[node_counter][i] +
                        (time_nodes[node_counter] - time_nodes[node_counter-1]) *
                        (f_new[i] - f_old[node_counter][i]) +
                        q_node[i]);
        }
    }

    bool is_finished() {
        return (status == SdcStatus::Completed || status == SdcStatus::Failed);
    }

    void evaluate_quadrature() {
        // Use high order quadrature rule to set q_node

        // Right now this is hardcoded for simpson integration
        // using evenly spaced nodes.

        if (node_counter == 1)
            // integral from m = 0 to m = 1
            for (size_t i = 0; i < SystemClass::neqs; i++) {
                q_node[i] = (time_nodes[1] - time_nodes[0])/12.0;
                q_node[i] *= 5.0*f_old[0][i] + 8.0*f_old[1][i] - f_old[2][i];
            }
        else
            // integral from m = 1 to m = 2
            for (size_t i = 0; i < SystemClass::neqs; i++) {
                q_node[i] = (time_nodes[1] - time_nodes[0])/12.0;
                q_node[i] *= -f_old[0][i] + 8.0*f_old[1][i] + 5.0*f_old[2][i];
            }            
    }

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
        } else if (status == SdcStatus::StartNewton) {
            // increment newton counter
            newton_counter++;
        }
    }

    void initialize_nodes() {
        // If we are advancing the step, then move y_old forward
        // along with time_previous
        if (step_counter > 0) {
            time_previous = time_nodes[number_nodes-1];
            y_old[0] = y_new[number_nodes-1];
        }

        // Evaluate the system at y_old and set y_old and f_old for nodes
        SystemClass::evaluate(time_nodes[0], y_old[0], f_old[0]);

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

    void guess_new_solution() {
        if (sdc_counter > 1)
            y_new[node_counter] = y_old[node_counter];
        else
            y_new[node_counter] = y_new[node_counter-1];
    }

    // Get the pointers to the linear system array A and vector b.
    // System is Ax = b.
    Real* get_matrix() {
        return j_new.dataPtr();
    }

    Real* get_vector() {
        return f_new.dataPtr();
    }
};
#endif
