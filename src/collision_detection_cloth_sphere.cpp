#include <collision_detection_cloth_sphere.h>
#include <iostream>

void collision_detection_cloth_sphere(std::vector<unsigned int> &cloth_index, std::vector<Eigen::Vector3d> &normals,
                                      Eigen::Ref<const Eigen::VectorXd> q, Eigen::Ref<const Eigen::Vector3d> center,
                                      double radius) {
    cloth_index.clear();
    normals.clear();
    for (int i = 0; i * 3 < q.rows(); i++) {
        Eigen::Vector3d d = q.segment(i * 3, 3) - center;
        if (d.squaredNorm() <= radius * radius) {
            cloth_index.emplace_back(i);
            normals.emplace_back(d.normalized());
        }
    }
}