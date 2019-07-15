#include <iostream>

#ifdef AMREX_USE_CUDA
#include <cuda_profiler_api.h>
#endif

#include "SdcIntegrator.H"
#include "SparseGaussJordan.H"
#include "vode_system.H"
#include "MathVector.H"
#include "MathVectorSet.H"
#include "VectorStorage.H"
#include "VectorParallelUtil.H"
#include "WallTimer.H"

#ifdef AMREX_USE_CUDA
template<class SparseLinearSolver, class SystemClass, size_t order, size_t vector_length>
__global__
void do_sdc_kernel(Real* y_initial, Real* y_final,
		   Real start_time, Real end_time, Real start_timestep,
                   Real tolerance, size_t maximum_newton_iters,
		   bool fail_if_maximum_newton, Real maximum_steps,
		   Real epsilon, size_t size, bool use_adaptive_timestep)
{
  PARALLEL_SHARED SystemClass ode_system;
  PARALLEL_SHARED SdcIntegrator<SparseLinearSolver, SystemClass, order, vector_length, StackCreate> sdc;
  PARALLEL_SHARED MathVectorSet<Real, SystemClass::neqs, vector_length, HeapWindow<Real, vector_length>> y_ini, y_fin;

  PARALLEL_REGION
    {
      // map y_ini and y_fin to y_initial and y_final
      y_ini.map(y_initial, size, -1, 0, 0, SystemClass::neqs);
      y_fin.map(y_final, size, -1, 0, 0, SystemClass::neqs);

      // initialize system and integrator
      ode_system.initialize();
      sdc.initialize(ode_system, y_ini,
                     start_time, end_time, start_timestep,
                     tolerance, maximum_newton_iters,
                     fail_if_maximum_newton, maximum_steps,
                     epsilon, use_adaptive_timestep);

      // integrate
      sdc.integrate();

      // save results to y_fin
      sdc.save_current_solution(y_fin);
    }
}
#endif


int main(int argc, char* argv[]) {

#ifdef AMREX_USE_CUDA
  cudaProfilerStart();
#endif

  size_t grid_size = 32;

  size_t num_systems = grid_size * grid_size * grid_size;

  const size_t order = 4;

  WallTimer timer;

  Real* y_initial;
  Real* y_final;

#ifndef AMREX_USE_CUDA
  y_initial = new Real[VodeSystem::neqs * num_systems];
  y_final = new Real[VodeSystem::neqs * num_systems];
#else
  cudaError_t cuda_status;
  void* vp;
  cuda_status = cudaMallocManaged(&vp, sizeof(Real) * VodeSystem::neqs * num_systems);
  assert(cuda_status == cudaSuccess);

  y_initial = static_cast<Real*>(vp);

  cuda_status = cudaMallocManaged(&vp, sizeof(Real) * VodeSystem::neqs * num_systems);
  assert(cuda_status == cudaSuccess);

  y_final = static_cast<Real*>(vp);
#endif

  // initialize systems with a Fab data layout
  for (size_t i = 0; i < num_systems * VodeSystem::neqs; i++) {
    if (i/num_systems == 0)
      y_initial[i] = 1.0;
    else if (i/num_systems == 1)
      y_initial[i] = 0.0;
    else if (i/num_systems == 2)
      y_initial[i] = 0.0;
  }

  Real start_time = 0.0;
  Real end_time = 1.0;
  Real start_timestep = (end_time - start_time)/10.0;
  Real tolerance = 1.0e-12;
  size_t maximum_newton_iters = 1000;
  size_t maximum_steps = 1000000;
  bool fail_if_maximum_newton = true;
  Real epsilon = std::numeric_limits<Real>::epsilon();
  bool use_adaptive_timestep = false;

  const int kernel_vector_length = 32 * 32;
  const int nBlocks = static_cast<int>(ceil(((double) num_systems)/(double) kernel_vector_length));
  const int nThreads = 128;

  std::cout << "Starting integration ..." << std::endl;

  timer.start_wallclock();

// #ifndef AMREX_USE_CUDA
//   do_sdc_host<SparseGaussJordan, VodeSystem, order>(y_initial, y_final,
// 						    start_time, end_time, start_timestep,
// 						    tolerance, maximum_newton_iters,
// 						    fail_if_maximum_newton, maximum_steps,
// 						    epsilon, num_systems, use_adaptive_timestep);
// #else
  do_sdc_kernel<SparseGaussJordan<VodeSystem, kernel_vector_length>,
		VodeSystem,
		order, kernel_vector_length><<<nBlocks, nThreads>>>(y_initial, y_final,
                                                                    start_time, end_time, start_timestep,
                                                                    tolerance, maximum_newton_iters,
                                                                    fail_if_maximum_newton, maximum_steps,
                                                                    epsilon, num_systems, use_adaptive_timestep);

  cuda_status = cudaDeviceSynchronize();
  assert(cuda_status == cudaSuccess);
// #endif

  timer.stop_wallclock();

  std::cout << std::endl << "Final Integration States -------------------" << std::endl;
  for (size_t i = 0; i < num_systems * VodeSystem::neqs; i++) {
    std::cout << std::setprecision(std::numeric_limits<Real>::digits10 + 1);
    std::cout << "y_final[" << i << "]: " << y_final[i] << std::endl;
  }
  std::cout << std::endl;

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
