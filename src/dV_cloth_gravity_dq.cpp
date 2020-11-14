#include <dV_cloth_gravity_dq.h>

void dV_cloth_gravity_dq(Eigen::VectorXd &fg, Eigen::SparseMatrixd &M, Eigen::Ref<const Eigen::Vector3d> g) {
    Eigen::VectorXd global_g(M.rows());
    for (int i = 0; i * 3 < M.rows(); i++) {
        global_g.segment(i * 3, 3) = g;
    }
    fg = -M * global_g;
}
