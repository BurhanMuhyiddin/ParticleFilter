#include "standard_particle_filter.h"
#include <iostream>

StandardParticleFilter::StandardParticleFilter(const unsigned long int num_particles) :
                                            ParticleFilter(num_particles) {
    set_is_initialised(false);
}

void StandardParticleFilter::init(const VectorXd &initial_state, const std::vector<double> &std, const ParticleFilter::KernelInit &kernel, void* args) {
    // call the kernel function with the initial states and std
    kernel(get_particles(), initial_state, std, args);
    set_is_initialised(true);
}

void StandardParticleFilter::predict(const void* u, const double dt, const std::vector<double> &std, const ParticleFilter::KernelPredict &kernel, void* args) {
    // call the kernel function with the input u and dt
    kernel(get_is_initialised(), get_particles(), u, dt, std, args);
}

void StandardParticleFilter::update(const void* z, const std::vector<double> &std, const ParticleFilter::KernelUpdate &kernel, void* args) {
    // call the kernel function with the observations z
    kernel(get_is_initialised(), get_particles(), get_weights(), z, std, args);
}

Eigen::VectorXd StandardParticleFilter::integrate(const ParticleFilter::KernelIntegrate &kernel, void* args) {
    return kernel(get_particles(), args);
}

void StandardParticleFilter::resample() {
    auto weights = get_weights();
    std::discrete_distribution<> dist_particles((*weights).begin(), (*weights).end());

    auto particles = *(get_particles());
    std::vector<Particle> new_particles(get_num_particles());
    for (size_t i = 0; i < get_num_particles(); ++i) {
        new_particles[i] = particles[dist_particles(ParticleFilter::rng)];
    }
    set_particles(new_particles);
}

std::default_random_engine ParticleFilter::rng;