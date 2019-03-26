#ifndef SDC_INTEGRATOR_H
#define SDC_INTEGRATOR_H

#include <iostream>
#include "SparseLinearSolver.H"
#include "AMReX_REAL.H"
#include "unified.H"
#include "unified_vector.H"
#include "SdcIntegratorType.H"


__global__
void states_to_solver_kernel(Real* solver_matrix,
                             Real** batched_matrices,
                             Real* solver_vector,
                             Real** batched_vectors,
                             size_t size, size_t neqs, size_t nnz) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
      Real* system_matrix = batched_matrices[tid];
      for (size_t i = 0; i < nnz; i++) {
          solver_matrix[tid * nnz + i] = system_matrix[i];
      }

      Real* system_vector = batched_vectors[tid];
      for (size_t i = 0; i < neqs; i++) {
          solver_vector[tid * neqs + i] = system_vector[i];
      }
  }
}

__global__
void solver_to_states_kernel(Real* solver_solution,
                             Real** batched_solutions,
                             size_t size, size_t neqs) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
      Real* system_solution = batched_solutions[tid];
      for (size_t i = 0; i < neqs; i++) {
          system_solution[i] = solver_solution[tid * neqs + i];
      }
  }
}

template<class SystemClass> class SdcIntegrator : public UnifiedMemoryClass {
    size_t number_systems;
    static const size_t order = 4;
    UnifiedVector<SdcIntegratorType<SystemClass, order>*> states;

    SystemClass ode_system;

    UnifiedVector<Real*> batched_matrices;
    UnifiedVector<Real*> batched_vectors;
    UnifiedVector<size_t> batched_indices;
    
    Real* solver_matrix;
    Real* solver_vector;
    Real* solver_solution;

private:

    void states_to_solver() {
        cudaError_t cuda_status = cudaSuccess;

        int size = batched_indices.size();
        int nThreads = min(32, size);
        int nBlocks = static_cast<int>(ceil(((double) size)/
                                              ((double) nThreads)));
        
        states_to_solver_kernel<<<nBlocks, nThreads>>>(solver_matrix,
                                                       batched_matrices.dataPtr(),
                                                       solver_vector,
                                                       batched_vectors.dataPtr(),
                                                       size,
                                                       SystemClass::neqs,
                                                       SystemClass::nnz);
    
        cuda_status = cudaDeviceSynchronize();
        assert(cuda_status == cudaSuccess);
    }

    void solver_to_states() {
        cudaError_t cuda_status = cudaSuccess;

        int size = number_systems;
        int nThreads = min(32, size);
        int nBlocks = static_cast<int>(ceil(((double) size)/
                                              ((double) nThreads)));

        solver_to_states_kernel<<<nBlocks, nThreads>>>(solver_solution,
                                                       batched_vectors.dataPtr(),
                                                       size,
                                                       SystemClass::neqs);

        cuda_status = cudaDeviceSynchronize();
        assert(cuda_status == cudaSuccess);
    }

    void initialize() {
        RealVector<SystemClass::neqs> y_initial;
        y_initial = 0.0;
        y_initial[0] = 1.0;
        Real end_time = 1.0e-6;
        Real dt_init = end_time/10.0;

        for (size_t i = 0; i < number_systems; i++) {
            SdcIntegratorType<SystemClass, order>* sdc_state;
            sdc_state = new SdcIntegratorType<SystemClass, order>(y_initial,
                                                                  end_time, dt_init,
                                                                  ode_system);
            states.push_back(sdc_state);
        }
    }

    void linear_solve() {
        // Construct and solve the linear system A*x = b
        // for each unfinished state. Put the answer x into f_new.
        size_t number_unfinished = batched_indices.size();

        SparseLinearSolver<SystemClass> linear_solver(number_unfinished, ode_system);

        solver_matrix = static_cast<Real*>(linear_solver.get_device_matrix());
        solver_vector = static_cast<Real*>(linear_solver.get_device_vector());
        solver_solution = static_cast<Real*>(linear_solver.get_device_solution());

        states_to_solver();
        linear_solver.solve();
        solver_to_states();
    }    

public:

    SdcIntegrator(size_t nsystems) {
        number_systems = nsystems;
        states.resize(0);
        batched_matrices.resize(0);
        batched_vectors.resize(0);
        batched_indices.resize(0);
        initialize();
        std::cout << "sdc integrator initialized" << std::endl;
    }

    ~SdcIntegrator() {}

    void integrate() {
        bool some_unfinished = true;
        
        batched_matrices.resize(0);
        batched_vectors.resize(0);
        batched_indices.resize(0);
        
        while (some_unfinished) {
            some_unfinished = false;

            // Prepare
            size_t istate = 0;
            for (auto& sdc_state : states) {
                if (!sdc_state->is_finished()) {
                    sdc_state->prepare();
                    some_unfinished = true;
                    batched_matrices.push_back(sdc_state->get_matrix());
                    batched_vectors.push_back(sdc_state->get_vector());
                    batched_indices.push_back(istate);
                }
                istate++;
            }

            if (!some_unfinished) break;

            // Solve linear system
            linear_solve();

            // Update
            for (auto& sdc_state : states) {
                if (!sdc_state->is_finished()) {
                    sdc_state->update();
                }
            }
        }
    }

    void print_state() {
        size_t istate = 0;
        std::cout << "in print_state" << std::endl;
        for (const auto& sdc_state : states) {
            std::cout << "System no. " << istate << std::endl;
            std::cout << "time = " << sdc_state->get_current_time() << std::endl;
            std::cout << "solution = "; sdc_state->get_current_solution().print(); std::cout << std::endl;
            istate++;
        }
        std::cout << "leaving print_state" << std::endl;
    }
};
#endif
