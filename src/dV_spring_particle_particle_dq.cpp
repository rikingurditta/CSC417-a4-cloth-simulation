#include <dV_spring_particle_particle_dq.h>

void dV_spring_particle_particle_dq(Eigen::Ref<Eigen::Vector6d> f, Eigen::Ref<const Eigen::Vector3d> q0,
                                    Eigen::Ref<const Eigen::Vector3d> q1, double l0, double stiffness) {
    // dV = -k (|q1-q0| - l0) * (q1 - q0) / |q1 - q0|
    f.head(3) = -stiffness * ((q1 - q0).norm() - l0) * (q1 - q0).normalized();
    f.tail(3) = -f.head(3);
}