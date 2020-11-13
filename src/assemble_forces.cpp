#include <assemble_forces.h>
#include <iostream>

void assemble_forces(Eigen::VectorXd &f, Eigen::Ref<const Eigen::VectorXd> q, Eigen::Ref<const Eigen::MatrixXd> qdot,
                     Eigen::Ref<const Eigen::MatrixXd> dX, Eigen::Ref<const Eigen::MatrixXd> V,
                     Eigen::Ref<const Eigen::MatrixXi> F, Eigen::Ref<const Eigen::VectorXd> a0,
                     double mu, double lambda) {
    f.setZero();
    for (int tri = 0; tri < F.rows(); tri++) {
        // get -force for current triangle
        Eigen::Vector9d dV_tri = Eigen::Vector9d::Zero();
        dV_membrane_corotational_dq(dV_tri, q, dX, V, F.row(tri), a0(tri), mu, lambda);
        // distribute to global force vector
        f.segment(F(tri, 0), 3) -= dV_tri.segment(0, 3);
        f.segment(F(tri, 1), 3) -= dV_tri.segment(6, 3);
        f.segment(F(tri, 2), 3) -= dV_tri.segment(9, 3);
    }
};
