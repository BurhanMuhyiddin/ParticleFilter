#ifndef PARTICLE_FILTER_H_
#define PARTICLE_FILTER_H_

#include <vector>
#include <random>
#include <Eigen/Core>

using namespace Eigen;

class Particle {
public:
    unsigned long int get_id() const { return id; }
    void set_id(const unsigned long int id) { this->id = id; }

    double get_weight() const { return weight; }
    void set_weight(const double weight) { this->weight = weight; }

    VectorXd* get_state() { return &state; }
    void set_state(const VectorXd &state) { this->state = state; }

    size_t get_state_dim() const { return state.rows(); }

private:
    unsigned long int id;
    VectorXd state;
    double weight;
};

class ParticleFilter {
public:
    typedef std::function<void(std::vector<Particle>* particles, const Eigen::VectorXd &initial_state, const std::vector<double> &std, void* args)> KernelInit;
    typedef std::function<void(bool is_initialised, std::vector<Particle>* particles, const void* u, const double dt, const std::vector<double> &std, void* args)> KernelPredict;
    typedef std::function<void(bool is_initialised, std::vector<Particle>* particles, std::vector<double>* weights, const void* z, const std::vector<double> &std, void* args)> KernelUpdate;
    typedef std::function<Eigen::VectorXd(std::vector<Particle>* particles, void* args)> KernelIntegrate;

    ParticleFilter(const unsigned long int num_particles) {
        this->num_particles = num_particles;
        particles.resize(num_particles);
        weights.resize(num_particles);

        resampling_threshold = 0.5 * num_particles;
    }

    unsigned long int get_num_particles() const { return num_particles; }
    void set_num_particles() { this->num_particles = num_particles; }

    std::vector<Particle>* get_particles() { return &this->particles; }
    void set_particles(const std::vector<Particle> &particles) { this->particles = std::move(particles); }

    std::vector<double>* get_weights() { return &weights; }
    void set_weights(const std::vector<double> &weights) { this->weights = std::move(weights); }

    bool get_is_initialised() const { return is_initialised; }
    void set_is_initialised(const bool is_initialised) { this->is_initialised = is_initialised; }

    double get_resampling_threshold() const { return resampling_threshold; }
    void set_resampling_threshold(const double resampling_threshold) { this->resampling_threshold = resampling_threshold; }

    virtual void init(const VectorXd &initial_state, const std::vector<double> &std, const ParticleFilter::KernelInit &kernel, void* args) = 0;
    virtual void predict(const void* u, const double dt, const std::vector<double> &std, const ParticleFilter::KernelPredict &kernel, void* args) = 0;
    virtual void update(const void* z, const std::vector<double> &std, const ParticleFilter::KernelUpdate &kernel, void* args) = 0;
    virtual Eigen::VectorXd integrate(const ParticleFilter::KernelIntegrate &kernel, void* args) = 0;

    void reset() {
        is_initialised = false;

        particles.clear();
        particles.resize(num_particles);

        weights.clear();
        weights.resize(num_particles);
    }

    double getESS() {
        long double sum = 0.0;
        long double sumsq = 0.0;

        auto particles = *get_particles();

        for (size_t i = 0; i < get_num_particles(); ++i)
            sum += particles[i].get_weight();

        for (size_t i = 0; i < get_num_particles(); ++i) {
            auto log_weight = log(particles[i].get_weight());
            sumsq += expl(2.0*(log_weight));
        }

        return expl(-log(sumsq) + 2.0*log(sum));
    }

public:
    static std::default_random_engine rng;
private:
    unsigned long int num_particles;
    std::vector<Particle> particles;
    std::vector<double> weights;
    bool is_initialised;
    double resampling_threshold;
};

#endif