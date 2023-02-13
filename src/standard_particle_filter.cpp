#include "standard_particle_filter.h"
#include <iostream>
#include "utils.h"

StandardParticleFilter::StandardParticleFilter(const unsigned long int num_particles) :
                                            ParticleFilter(num_particles) {
    set_is_initialised(false);
}

void StandardParticleFilter::init(const VectorXd &state, const std::vector<double> &std) {
    // get reference to particle array
    auto particles = get_particles();

    // get number of particles
    auto num_particles = get_num_particles();

    // create vector of normal distributions with mean of state and std of std
    std::vector<std::normal_distribution<double>> state_rngs(state.rows());
    for (size_t i = 0; i < state.rows(); ++i) {
        state_rngs[i] = std::normal_distribution<double>(state[i], std[i]);
    }

    // set initial value of particle states and their ids and weights
    double init_weight = 1.0 / (1.0 * num_particles);
    for (size_t i = 0; i < num_particles; ++i) {
        VectorXd temp(state.rows());
        for (size_t j = 0; j < state.rows(); ++j) {
            if (j == 2) {
                temp(j) = wrapAngle(state_rngs[j](rng));
            } else {
                temp(j) = state_rngs[j](rng);
            }
            
        }
        (*particles)[i].set_id(i);
        (*particles)[i].set_state(temp);
        (*particles)[i].set_weight(init_weight);
    }
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