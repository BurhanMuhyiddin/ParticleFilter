#ifndef STANDARD_PARTICLE_FILTER_H_
#define STANDARD_PARTICLE_FILTER_H_

#include "particle_filter.h"

class StandardParticleFilter : public ParticleFilter {
public:
    StandardParticleFilter() = default;
    StandardParticleFilter(const unsigned long int num_particles);

    void init(const VectorXd &state, const std::vector<double> &std) override;
    void predict(const void* u, const double dt, const std::vector<double> &std, const ParticleFilter::KernelPredict &kernel, void* args) override;
    void update(const void* z, const std::vector<double> &std, const ParticleFilter::KernelUpdate &kernel, void* args) override;
    Eigen::VectorXd integrate(const ParticleFilter::KernelIntegrate &kernel, void* args) override;

    void resample();
private:
};

#endif