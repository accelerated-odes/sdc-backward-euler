#include <iostream>
#include "sdc.H"
#include "vode_system.H"
#include "RealVector.H"

int main(int argc, char* argv[]) {

    size_t num_systems = 1;

    SdcIntegrator<VodeSystem> sdc_integrator(num_systems);

    RealVector<VodeSystem::neqs> y_initial;
    y_initial = 0.0;
    y_initial[0] = 1.0;
    Real start_time = 0.0;
    Real end_time = 1.0;
    Real start_timestep = (end_time - start_time)/10.0;
    Real tolerance = 1.0e-6;
    size_t maximum_newton_iters = 100;
    bool fail_if_maximum_newton = true;

    for (auto& sdc_state : sdc_integrator) {
      sdc_state->initialize(y_initial, start_time, end_time, start_timestep,
			    tolerance, maximum_newton_iters, fail_if_maximum_newton);
    }

    std::cout << "Initial Conditions -----------------" << std::endl;
    sdc_integrator.print_states();

    sdc_integrator.integrate();

    std::cout << "Final Conditions -------------------" << std::endl;
    sdc_integrator.print_states();

    size_t i = 0;
    for (auto& sdc_state : sdc_integrator) {
      std::cout << "system " << i << std::endl;
      sdc_state->print();
      i++;
    }

    return 0;
}
