#include <assemble_stiffness.h>
#include <iostream>

void
assemble_stiffness(Eigen::SparseMatrixd &K, Eigen::Ref<const Eigen::VectorXd> q, Eigen::Ref<const Eigen::VectorXd> qdot,
                   Eigen::Ref<const Eigen::MatrixXd> dX, Eigen::Ref<const Eigen::MatrixXd> V,
                   Eigen::Ref<const Eigen::MatrixXi> F, Eigen::Ref<const Eigen::VectorXd> a0,
                   double mu, double lambda) {
    K.setZero();
    K.resize(q.rows(), q.rows());
    std::vector<Eigen::Triplet<double>> tl;
    tl.reserve(81 * F.rows());
    for (int tri = 0; tri < F.rows(); tri++) {
        // get -K for current triangle
        Eigen::Matrix99d d2V_tri = Eigen::Matrix99d::Zero();
        d2V_membrane_corotational_dq2(d2V_tri, q, dX, V, F.row(tri), a0(tri), mu, lambda);
        // distribute to global K
        for (int tri_i = 0; tri_i < 3; tri_i++) {
            for (int tri_j = 0; tri_j < 3; tri_j++) {
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        tl.emplace_back(F(tri, tri_i) * 3 + i, F(tri, tri_j) * 3 + j, -d2V_tri(i, j));
                    }
                }
            }
        }
    }
    K.setFromTriplets(tl.begin(), tl.end());
};
