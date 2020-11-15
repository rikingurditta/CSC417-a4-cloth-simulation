#include <assemble_forces.h>
#include <iostream>

void assemble_forces(Eigen::VectorXd &f, Eigen::Ref<const Eigen::VectorXd> q, Eigen::Ref<const Eigen::MatrixXd> qdot,
                     Eigen::Ref<const Eigen::MatrixXd> dX, Eigen::Ref<const Eigen::MatrixXd> V,
                     Eigen::Ref<const Eigen::MatrixXi> F, Eigen::Ref<const Eigen::VectorXd> a0,
                     double mu, double lambda) {
    f = Eigen::VectorXd::Zero(q.rows());
    for (int tri = 0; tri < F.rows(); tri++) {
        // get -force for current triangle
        Eigen::Vector9d dV_tri = Eigen::Vector9d::Zero();
        Eigen::Matrix3d dphi = Eigen::Map<Eigen::Matrix3d>(((Eigen::Vector9d)dX.row(tri)).data());
        dV_membrane_corotational_dq(dV_tri, q, dphi, V, F.row(tri), a0(tri), mu, lambda);
        // distribute to global force vector
        f.segment(F(tri, 0) * 3, 3) -= dV_tri.segment(0, 3);
        f.segment(F(tri, 1) * 3, 3) -= dV_tri.segment(3, 3);
        f.segment(F(tri, 2) * 3, 3) -= dV_tri.segment(6, 3);
    }
}