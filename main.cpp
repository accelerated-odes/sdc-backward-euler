#include <iostream>

#ifdef AMREX_USE_CUDA
#include <cuda_profiler_api.h>
#endif

#include "SdcIntegrator.H"
#include "SparseGaussJordan.H"
#include "vode_system.H"
#include "RealVector.H"
#include "WallTimer.H"

#ifdef AMREX_USE_CUDA
template<class SparseLinearSolver, class SystemClass, size_t order>
__global__
void do_sdc_kernel(Real* y_initial, Real* y_final, 
		   Real start_time, Real end_time, Real start_timestep,
                   Real tolerance, size_t maximum_newton_iters, 
		   bool fail_if_maximum_newton, Real maximum_steps,
		   Real epsilon, size_t size) {

  typedef SdcIntegrator<SparseLinearSolver,SystemClass,order> SdcIntClass;

  const size_t WarpBatchSize = 128;
  const size_t WarpSize = 32;
  size_t warp_batch_id = blockIdx.x * WarpBatchSize;
  size_t global_index, local_index;

  SystemClass ode_system;
  global_index = threadIdx.x + warp_batch_id;

  if (global_index >= size) return;

  for (local_index = threadIdx.x; local_index < WarpBatchSize && global_index < size; local_index += WarpSize) {
    global_index = local_index + warp_batch_id;

    SdcIntClass sdc;
    RealVector<SystemClass::neqs> y_ini;

    for (size_t i = 0; i < SystemClass::neqs; i++) {
      y_ini.data[i] = y_initial[global_index * SystemClass::neqs + i];
    }

    SdcIntClass::set_jacobian_layout(sdc, ode_system);
    SdcIntClass::initialize(sdc, y_ini, 
			    start_time, end_time, start_timestep,
			    tolerance, maximum_newton_iters, 
			    fail_if_maximum_newton, maximum_steps,
			    epsilon);

    for (size_t i = 0; i < maximum_steps; i++) {
      SdcIntClass::prepare(sdc);
      SdcIntClass::solve(sdc);
      SdcIntClass::update(sdc);
      if (SdcIntClass::is_finished(sdc)) break;
    }

    RealVector<SystemClass::neqs>& y_fin = SdcIntClass::get_current_solution(sdc);
    for (size_t i = 0; i < SystemClass::neqs; i++) {
      y_final[global_index * SystemClass::neqs + i] = y_fin.data[i];
    }
  }
}
#endif

template<class SparseLinearSolver, class SystemClass, size_t order>
void do_sdc_host(Real* y_initial, Real* y_final, 
		 Real start_time, Real end_time, Real start_timestep,
		 Real tolerance, size_t maximum_newton_iters, 
		 bool fail_if_maximum_newton, Real maximum_steps,
		 Real epsilon, size_t size) {

  typedef SdcIntegrator<SparseLinearSolver,SystemClass,order> SdcIntClass;

  SystemClass ode_system;

  for (size_t global_index = 0; global_index < size; global_index++) {
    SdcIntClass sdc;
    RealVector<SystemClass::neqs> y_ini;

    for (size_t i = 0; i < SystemClass::neqs; i++) {
      y_ini.data[i] = y_initial[global_index * SystemClass::neqs + i];
    }

    SdcIntClass::set_jacobian_layout(sdc, ode_system);
    SdcIntClass::initialize(sdc, y_ini, 
			    start_time, end_time, start_timestep,
			    tolerance, maximum_newton_iters, 
			    fail_if_maximum_newton, maximum_steps,
			    epsilon);

    for (size_t i = 0; i < maximum_steps; i++) {
      SdcIntClass::prepare(sdc);
      SdcIntClass::solve(sdc);
      SdcIntClass::update(sdc);
      if (SdcIntClass::is_finished(sdc)) break;
    }

    RealVector<SystemClass::neqs>& y_fin = SdcIntClass::get_current_solution(sdc);
    for (size_t i = 0; i < SystemClass::neqs; i++) {
      y_final[global_index * SystemClass::neqs + i] = y_fin.data[i];
    }
  }
}


int main(int argc, char* argv[]) {

#ifdef AMREX_USE_CUDA
  cudaProfilerStart();
#endif

  size_t num_systems = 2100000;
  const size_t order = 4;

  WallTimer timer;

  Real* y_initial;
  Real* y_final;

#ifndef AMREX_USE_CUDA
  y_initial = new Real[VodeSystem::neqs * num_systems];
  y_final = new Real[VodeSystem::neqs * num_systems];
#else
  cudaError_t cuda_status = cudaSuccess;
  void* vp;
  cuda_status = cudaMallocManaged(&vp, sizeof(Real) * VodeSystem::neqs * num_systems);
  assert(cuda_status == cudaSuccess);

  y_initial = static_cast<Real*>(vp);

  cuda_status = cudaMallocManaged(&vp, sizeof(Real) * VodeSystem::neqs * num_systems);
  assert(cuda_status == cudaSuccess);

  y_final = static_cast<Real*>(vp);
#endif

  // initialize systems
  for (size_t i = 0; i < num_systems; i += VodeSystem::neqs) {
    y_initial[i] = 1.0;
    y_initial[i+1] = 0.0;
    y_initial[i+2] = 0.0;
  }

  Real start_time = 0.0;
  Real end_time = 1.0;
  Real start_timestep = (end_time - start_time)/10.0;
  Real tolerance = 1.0e-6;
  size_t maximum_newton_iters = 100;
  size_t maximum_steps = 1000000;
  bool fail_if_maximum_newton = true;
  Real epsilon = std::numeric_limits<Real>::epsilon();

  const int nThreads = 32;
  const size_t WarpBatchSize = 128;
  const int nBlocks = static_cast<int>(ceil(((double) num_systems)/(double) WarpBatchSize));

  std::cout << "Starting integration ..." << std::endl;

  timer.start_wallclock();

#ifndef AMREX_USE_CUDA
  do_sdc_host<SparseGaussJordan, VodeSystem, order>(y_initial, y_final,
						    start_time, end_time, start_timestep,
						    tolerance, maximum_newton_iters,
						    fail_if_maximum_newton, maximum_steps,
						    epsilon, num_systems);
#else
  do_sdc_kernel<SparseGaussJordan, 
		VodeSystem, 
		order><<<nBlocks, nThreads>>>(y_initial, y_final,
					      start_time, end_time, start_timestep,
					      tolerance, maximum_newton_iters,
					      fail_if_maximum_newton, maximum_steps,
					      epsilon, num_systems);

  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);
#endif

  timer.stop_wallclock();

  std::cout << std::endl << "Final Integration States -------------------" << std::endl;
  for (size_t i = 0; i < num_systems; i += VodeSystem::neqs) {
    std::cout << "y_final[" << i << "]: " << std::endl;
    std::cout << " ";
    for (size_t j = 0; j < VodeSystem::neqs; j++) {
      std::cout << y_final[i + j] << " ";
    }
    std::cout << std::endl;
  }

#ifndef AMREX_USE_CUDA
  std::cout << "Finished execution on host CPU" << std::endl;
#else
  std::cout << "Finished execution on device" << std::endl;
#endif
  
  std::cout << std::endl << "Integration walltime (s): " << timer.get_walltime() << std::endl;

#ifndef AMREX_USE_CUDA
  delete[] y_initial;
  delete[] y_final;
#else
  cuda_status = cudaFree(y_initial);
  assert(cuda_status == cudaSuccess);
  cuda_status = cudaFree(y_final);
  assert(cuda_status == cudaSuccess);

  cudaProfilerStop();
#endif

  return 0;
}
