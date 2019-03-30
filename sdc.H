#ifndef SDC_INTEGRATOR_H
#define SDC_INTEGRATOR_H

#include <iostream>
#include <vector>
#include <thrust/device_vector.h>
#include "SparseLinearSolver.H"
#include "AMReX_REAL.H"
#include "AMReX_GpuError.H"
#include "unified.H"
#include "unified_vector.H"
#include "WallTimer.H"
#include "SdcIntegratorType.H"

template<class SystemClass, size_t order>
__global__
void prepare_states_kernel(SdcIntegratorType<SystemClass, order>** states, size_t size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    states[tid]->prepare();
  }
}

template<class SystemClass, size_t order>
__global__
void sdc_state_map_linear_system(SdcIntegratorType<SystemClass, order>** s,
				 Real** A, Real** x, Real** b, size_t size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size)
    s[tid]->map_linear_system(A[tid], x[tid], b[tid]);
}

template<class SystemClass, size_t order>
__global__
void update_states_kernel(SdcIntegratorType<SystemClass, order>** states, size_t size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    states[tid]->update();
  }
}

template<class SystemClass> class SdcIntegrator : public WallTimer {
  size_t number_systems;
  static const size_t order = 4;

  std::vector<SdcIntegratorType<SystemClass, order>*> states;
  SdcIntegratorType<SystemClass, order>* state_memory;

  SystemClass ode_system;

  UnifiedVector<SdcIntegratorType<SystemClass, order>*> batched_states;

  void initialize() {
    void* vp;

    AMREX_GPU_SAFE_CALL(cudaMallocManaged(&vp, number_systems * sizeof(SdcIntegratorType<SystemClass, order>)))
    state_memory = static_cast<SdcIntegratorType<SystemClass, order>*>(vp);

    for (size_t i = 0; i < number_systems; i++) {
      SdcIntegratorType<SystemClass, order>* sdc_state = &state_memory[i];
      new (sdc_state) SdcIntegratorType<SystemClass, order>(ode_system);
      states.push_back(sdc_state);
    }
  }

public:

  SdcIntegrator(size_t nsystems) {
    number_systems = nsystems;
    states.resize(0);
    batched_states.resize(0);
    initialize();
    std::cout << "sdc integrator initialized" << std::endl;
  }

  ~SdcIntegrator() {}

  decltype(states.begin()) begin() {
    return states.begin();
  }

  decltype(states.begin()) end() {
    return states.end();
  }

  void integrate() {
    start_wallclock();

    bool some_unfinished = true;

    while (some_unfinished) {
      some_unfinished = false;

      batched_states.resize(0);

      // Figure out which have not finished
      size_t istate = 0;
      for (auto& sdc_state : states) {
	if (!sdc_state->is_finished()) {
	  some_unfinished = true;
	  batched_states.push_back(sdc_state);
	}
	istate++;
      }

      int batched_size = istate;

      if (!some_unfinished) break;

      {
	SparseLinearSolver<SystemClass> linear_solver(batched_size, ode_system);

	Real* solver_matrix = static_cast<Real*>(linear_solver.get_device_matrix());
	Real* solver_solution = static_cast<Real*>(linear_solver.get_device_solution());
	Real* solver_vector = static_cast<Real*>(linear_solver.get_device_vector());

	std::vector<Real*> solver_matrix_map(batched_size);
	std::vector<Real*> solver_solution_map(batched_size);
	std::vector<Real*> solver_vector_map(batched_size);

	for (size_t i = 0; i < batched_size; i++) {
	  solver_matrix_map[i] = &solver_matrix[i * SystemClass::nnz];
	  solver_solution_map[i] = &solver_solution[i * SystemClass::neqs];
	  solver_vector_map[i] = &solver_vector[i * SystemClass::neqs];
	}

	thrust::device_vector<Real*> solver_matrix_map_d = solver_matrix_map;
	thrust::device_vector<Real*> solver_solution_map_d = solver_solution_map;
	thrust::device_vector<Real*> solver_vector_map_d = solver_vector_map;

	int size = batched_size;
	int nThreads = min(32, size);
	int nBlocks = static_cast<int>(ceil(((double) size)/
					    ((double) nThreads)));

	SdcIntegratorType<SystemClass,order>** s_pointers = thrust::raw_pointer_cast(&batched_states[0]);
	Real** A_pointers = thrust::raw_pointer_cast(&solver_matrix_map_d[0]);
	Real** x_pointers = thrust::raw_pointer_cast(&solver_solution_map_d[0]);
	Real** b_pointers = thrust::raw_pointer_cast(&solver_vector_map_d[0]);

	sdc_state_map_linear_system<SystemClass, order><<<nBlocks, nThreads>>>(s_pointers,
									       A_pointers,
									       x_pointers,
									       b_pointers,
									       batched_size);
	cudaError_t cuda_status = cudaDeviceSynchronize();
	assert(cuda_status == cudaSuccess);

	prepare_states();
	linear_solver.solve();
	update_states();
      }

    }

    stop_wallclock();
  }

  void prepare_states() {
    cudaError_t cuda_status = cudaSuccess;

    int size = batched_states.size();
    int nThreads = min(32, size);
    int nBlocks = static_cast<int>(ceil(((double) size)/
					((double) nThreads)));

    prepare_states_kernel<SystemClass, order><<<nBlocks, nThreads>>>(batched_states.dataPtr(), size);

    cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);
  }

  void update_states() {
    cudaError_t cuda_status = cudaSuccess;

    int size = batched_states.size();
    int nThreads = min(32, size);
    int nBlocks = static_cast<int>(ceil(((double) size)/
					((double) nThreads)));

    update_states_kernel<SystemClass, order><<<nBlocks, nThreads>>>(batched_states.dataPtr(), size);

    cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);
  }

  void print_states() {
    size_t istate = 0;
    for (const auto& sdc_state : states) {
      std::cout << "System no. " << istate << std::endl;
      if (sdc_state->is_started()) {
	std::cout << "time = " << sdc_state->get_current_time() << std::endl;
	std::cout << "solution = "; sdc_state->get_current_solution().print(); std::cout << std::endl;
      } else {
	std::cout << "time = " << sdc_state->get_previous_time() << std::endl;
	std::cout << "solution = "; sdc_state->get_previous_solution().print(); std::cout << std::endl;
      }
      istate++;
    }
  }
};
#endif
