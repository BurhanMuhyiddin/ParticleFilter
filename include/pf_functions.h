#ifndef PF_FUNCTIONS_H_
#define PF_FUNCTIONS_H_

#include "particle_filter.h"
// #include "utils.h"
// #include "sensors.h"
// #include "beacons.h"

void handle_gyro(bool is_initialised, std::vector<Particle>* particles, const void* u, const double dt, const std::vector<double> &std, void* args) {
    auto inp = (GyroMeasurement*)u;

    if (is_initialised) {
        auto psi_dot = inp->psi_dot;
        auto psi_dot_dt = psi_dot * dt;

        for (size_t i = 0; i < particles->size(); ++i) {
            auto state = (*particles)[i].get_state();

            double x = (*state)(0);
            double y = (*state)(1);
            double psi = (*state)(2);
            double V = (*state)(3);

            // Update State
            double x_new = x + dt * V * cos(psi);
            double y_new = y + dt * V * sin(psi);
            double psi_new = wrapAngle(psi + psi_dot_dt);
            double V_new = V;

            x_new += std::normal_distribution<double>(0.0, std[0])(ParticleFilter::rng);
            y_new += std::normal_distribution<double>(0.0, std[1])(ParticleFilter::rng);
            psi_new += std::normal_distribution<double>(0.0, std[2])(ParticleFilter::rng);
            V_new += std::normal_distribution<double>(0.0, std[3])(ParticleFilter::rng);
            (*state) << x_new,y_new,wrapAngle(psi_new),V_new;
        }
    }
}

void handle_gps(bool is_initialised, std::vector<Particle>* particles, std::vector<double>* weights, const void* z, const std::vector<double> &std, void* args) {
    auto meas = (GPSMeasurement*)z;

    auto previous_meas = (GPSMeasurement*)args;
    
    if (is_initialised) {
        const auto gps_var = std[0] * std[0];
        const auto heading_var = std[1] * std[1];

        // const double k = 2 * M_PI * gps_var;

        double sum_w = 0.0;
        for (size_t i = 0; i < particles->size(); ++i) {
            auto state = (*particles)[i].get_state();

            double weight_no_exp = 0.0;
            double dx = meas->x - (*state)(0);
            double dy = meas->y - (*state)(1);
            double h_dx = meas->x - previous_meas->x;
            double h_dy = meas->y - previous_meas->y;
            double heading = atan2(h_dy, h_dx);
            double d_heading = wrapAngle(heading - (*state)(2));
            weight_no_exp += dx * dx / gps_var + dy * dy / gps_var + d_heading * d_heading / heading_var; // dx^T * R^-1 * dx
            (*particles)[i].set_weight((*particles)[i].get_weight() * exp(-0.5*weight_no_exp));
            sum_w += (*particles)[i].get_weight();
        }
        for (size_t i = 0; i < particles->size(); ++i) {
            (*particles)[i].set_weight((*particles)[i].get_weight() / (sum_w /** k*/));
            (*weights)[i] = (*particles)[i].get_weight();
        }
    }
}

void handle_lidar(bool is_initialised, std::vector<Particle>* particles, std::vector<double>* weights, const void* z, const std::vector<double> &std, void* args) {
    auto dataset = (std::vector<LidarMeasurement>*)z;

    BeaconMap* map = (BeaconMap*)args;

    if (is_initialised) {
        const auto range_var = std[0] * std[0];
        const auto theta_var = std[1] * std[1];

        double sum_w = 0.0;
        for (size_t i = 0; i < particles->size(); ++i) {
            auto state = (*particles)[i].get_state();

            double weight_no_exp = 0.0;
            for(const auto& meas : *dataset) {
                BeaconData map_beacon = map->getBeaconWithId(meas.id);

                if (meas.id != -1 && map_beacon.id != -1) {
                    double delta_x = map_beacon.x - (*state)(0);
                    double delta_y = map_beacon.y - (*state)(1);
                    double zhat_range = sqrt(delta_x*delta_x + delta_y*delta_y);
                    double zhat_theta = wrapAngle(atan2(delta_y,delta_x) - (*state)(2));
                    double d_range = meas.range - zhat_range;
                    double d_theta = wrapAngle(meas.theta - zhat_theta);
                    weight_no_exp += d_range * d_range / range_var + d_theta * d_theta / theta_var;
                }
            }
            (*particles)[i].set_weight((*particles)[i].get_weight() * exp(-0.5*weight_no_exp));
            sum_w += (*particles)[i].get_weight();
        }
        for (size_t i = 0; i < particles->size(); ++i) {
            (*particles)[i].set_weight((*particles)[i].get_weight() / (sum_w /** k*/));
            (*weights)[i] = (*particles)[i].get_weight();
        }
    }
}

Eigen::VectorXd integrand_mean(std::vector<Particle>* particles, void* args) {
    double sum_w = 0.0;
    Eigen::VectorXd temp = Eigen::VectorXd::Zero((*particles)[0].get_state_dim());

    for (size_t i = 0; i < particles->size(); ++i) {
        temp += (*particles)[i].get_weight() * (*(*particles)[i].get_state());
        sum_w += (*particles)[i].get_weight();
    }

    return temp / sum_w;
}

Eigen::VectorXd integrand_variance(std::vector<Particle>* particles, void* args) {
    auto mean = *((Eigen::VectorXd*)args);

    double sum_w = 0.0;
    Eigen::VectorXd temp = Eigen::VectorXd::Zero((*particles)[0].get_state_dim());

    for (size_t i = 0; i < particles->size(); ++i) {
        auto diff = mean - (*(*particles)[i].get_state());
        temp += (*particles)[i].get_weight() * diff * diff;
        sum_w += (*particles)[i].get_weight();
    }

    return temp / sum_w;
}

#endif