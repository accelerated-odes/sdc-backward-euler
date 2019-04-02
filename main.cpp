#include <iostream>
#include <cuda_profiler_api.h>
#include "SdcIntegrator.H"
#include "SparseGaussJordan.H"
#include "vode_system.H"
#include "RealVector.H"
#include "WallTimer.H"

template<class SparseLinearSolver, class SystemClass, size_t order>
__global__
void do_sdc_kernel(Real* y_initial, Real* y_final, Real start_time, Real end_time, Real start_timestep,
                   Real tolerance, size_t maximum_newton_iters, bool fail_if_maximum_newton,
                   size_t size) {
    typedef SdcIntegrator<SparseLinearSolver,SystemClass,order> SdcIntClass;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= size) return;

    SystemClass ode_system;
    SdcIntClass sdc;

    RealVector<SystemClass::neqs> y_ini;
    for (size_t i = 0; i < SystemClass::neqs; i++) {
        y_ini.data[i] = y_initial[tid * SystemClass::neqs + i];
    }

    SdcIntClass::set_jacobian_layout(sdc, ode_system);
    SdcIntClass::initialize(sdc, y_ini, start_time, end_time, start_timestep,
                            tolerance, maximum_newton_iters, fail_if_maximum_newton);

    for (size_t i = 0; i < SdcIntClass::maximum_steps; i++) {
        SdcIntClass::prepare(sdc);
        SdcIntClass::solve(sdc);
        SdcIntClass::update(sdc);
        if (SdcIntClass::is_finished(sdc)) break;
    }

    RealVector<SystemClass::neqs>& y_fin = SdcIntClass::get_current_solution(sdc);
    for (size_t i = 0; i < SystemClass::neqs; i++) {
        y_final[tid * SystemClass::neqs + i] = y_fin.data[i];
    }
}

int main(int argc, char* argv[]) {

    cudaProfilerStart();

    size_t num_systems = 1000000;
    const size_t order = 4;

    WallTimer timer;

    Real* y_initial;
    Real* y_final;
    
    cudaError_t cuda_status = cudaSuccess;
    void* vp;
    cuda_status = cudaMallocManaged(&vp, sizeof(Real) * VodeSystem::neqs * num_systems);
    assert(cuda_status == cudaSuccess);

    y_initial = static_cast<Real*>(vp);

    cuda_status = cudaMallocManaged(&vp, sizeof(Real) * VodeSystem::neqs * num_systems);
    assert(cuda_status == cudaSuccess);

    y_final = static_cast<Real*>(vp);

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
    bool fail_if_maximum_newton = true;

    const int nThreads = 32;
    const int WarpBatchSize = 32;
    const int nBlocks = static_cast<int>(ceil(((double) num_systems)/(double) WarpBatchSize));

    std::cout << "Starting integration ..." << std::endl;

    timer.start_wallclock();

    do_sdc_kernel<SparseGaussJordan, VodeSystem, order><<<nBlocks, nThreads>>>(y_initial, y_final,
                                                                               start_time, end_time,
                                                                               start_timestep,
                                                                               tolerance,
                                                                               maximum_newton_iters,
                                                                               fail_if_maximum_newton,
                                                                               num_systems);

    cuda_status = cudaDeviceSynchronize();
    assert(cuda_status == cudaSuccess);

    timer.stop_wallclock();

    std::cout << std::endl << "Final Integration States -------------------" << std::endl;
    for (size_t i = 0; i < num_systems; i += VodeSystem::neqs) {
        std::cout << "y_final[" << i << "]: " << std::endl;
        std::cout << " ";
        for (size_t j = 0; j < VodeSystem::neqs; j++) {
            std::cout << y_final[i + j];
        }
        std::cout << std::endl;
    }

    std::cout << std::endl << "Integration walltime (s): " << timer.get_walltime() << std::endl;

    cuda_status = cudaFree(y_initial);
    assert(cuda_status == cudaSuccess);
    cuda_status = cudaFree(y_final);
    assert(cuda_status == cudaSuccess);

    cudaProfilerStop();

    return 0;
}
