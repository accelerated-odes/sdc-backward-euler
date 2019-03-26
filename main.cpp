#include <iostream>
#include "sdc.H"
#include "vode_system.H"

int main(int argc, char* argv[]) {

    size_t num_systems = 2;

    SdcIntegrator<VodeSystem> sdc_integrator(num_systems);

    sdc_integrator.print_state();
    
    sdc_integrator.integrate();

    sdc_integrator.print_state();
    
    return 0;
}
