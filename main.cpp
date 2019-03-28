#include <iostream>
#include <cuda_profiler_api.h>
#include "sdc.H"
#include "vode_system.H"
#include "RealVector.H"
#include "WallTimer.H"

int main(int argc, char* argv[]) {

    cudaProfilerStart();

    size_t num_systems = 1000000;

    WallTimer timer;

    timer.start_wallclock();
    SdcIntegrator<VodeSystem> sdc_integrator(num_systems);
    timer.stop_wallclock();

    std::cout << "Initialized SdcIntegrator driver in " << timer.get_walltime() << " s." << std::endl;

    RealVector<VodeSystem::neqs> y_initial;
    y_initial = 0.0;
    y_initial[0] = 1.0;
    Real start_time = 0.0;
    Real end_time = 1.0;
    Real start_timestep = (end_time - start_time)/10.0;
    Real tolerance = 1.0e-6;
    size_t maximum_newton_iters = 100;
    bool fail_if_maximum_newton = true;

    timer.start_wallclock();

    for (auto& sdc_state : sdc_integrator) {
      sdc_state->initialize(y_initial, start_time, end_time, start_timestep,
			    tolerance, maximum_newton_iters, fail_if_maximum_newton);
    }

    timer.stop_wallclock();

    std::cout << "Initialized SdcIntegrator states in " << timer.get_walltime() << " s." << std::endl;

    // std::cout << std::endl << "Initial Conditions -----------------" << std::endl;
    // sdc_integrator.print_states();

    std::cout << "Starting integration ..." << std::endl;

    sdc_integrator.integrate();

    std::cout << std::endl << "Final Conditions -------------------" << std::endl;
    sdc_integrator.print_states();

    // std::cout << std::endl << "Final Integration States -------------------" << std::endl;
    // size_t i = 0;
    // for (auto& sdc_state : sdc_integrator) {
    //   std::cout << "system " << i << std::endl;
    //   sdc_state->print();
    //   i++;
    // }

    std::cout << std::endl << "Integration walltime (s): " << sdc_integrator.get_walltime() << std::endl;

    cudaProfilerStop();

    return 0;
}
