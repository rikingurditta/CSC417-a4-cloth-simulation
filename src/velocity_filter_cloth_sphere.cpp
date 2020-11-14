#include <velocity_filter_cloth_sphere.h>

void velocity_filter_cloth_sphere(Eigen::VectorXd &qdot, const std::vector<unsigned int> &indices, 
                                  const std::vector<Eigen::Vector3d> &normals) {
    for (int i = 0; i < indices.size(); i++) {
        Eigen::Vector3d curr_qdot = qdot.segment(i * 3, 3);
        qdot.segment(i * 3, 3) = curr_qdot - normals[i] * normals[i].transpose() * curr_qdot;
    }
}