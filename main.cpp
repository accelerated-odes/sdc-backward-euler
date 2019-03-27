#include <iostream>
#include "sdc.H"
#include "vode_system.H"
#include "RealVector.H"

int main(int argc, char* argv[]) {

    size_t num_systems = 2;

    SdcIntegrator<VodeSystem> sdc_integrator(num_systems);

    RealVector<VodeSystem::neqs> y_initial;
    y_initial = 0.0;
    y_initial[0] = 1.0;
    Real start_time = 0.0;
    Real end_time = 1.0e8;
    Real start_timestep = (end_time - start_time)/10.0;

    for (auto& sdc_state : sdc_integrator) {
      sdc_state->initialize(y_initial, start_time, end_time, start_timestep);
    }

    std::cout << "Initial Conditions -----------------" << std::endl;
    sdc_integrator.print_state();
    
    sdc_integrator.integrate();

    std::cout << "Final Conditions -------------------" << std::endl;
    sdc_integrator.print_state();

    size_t i = 0;
    for (auto& sdc_state : sdc_integrator) {
      std::cout << "system " << i << std::endl;
      sdc_state->print();
      i++;
    }
    
    return 0;
}
