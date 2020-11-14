#include <d2V_membrane_corotational_dq2.h>
#include "dphi_cloth_triangle_dX.h"
#include <iostream>

Eigen::Matrix3d cross_product_matrix_2(const Eigen::Ref<const Eigen::Vector3d> &v);

void d2V_membrane_corotational_dq2(Eigen::Matrix99d &H, Eigen::Ref<const Eigen::VectorXd> q,
                                   Eigen::Ref<const Eigen::Matrix3d> dX, Eigen::Ref<const Eigen::MatrixXd> V,
                                   Eigen::Ref<const Eigen::RowVectorXi> element,
                                   double area, double mu, double lambda) {
    // vertices of element
    int x0 = element(0);
    int x1 = element(1);
    int x2 = element(2);
    // calculate deformation gradient F
    Eigen::Matrix34d xs;
    xs.col(0) = q.segment(x0 * 3, 3);
    xs.col(1) = q.segment(x1 * 3, 3);
    xs.col(2) = q.segment(x2 * 3, 3);
    Eigen::Vector3d delta_x1 = xs.col(1) - xs.col(0);
    Eigen::Vector3d delta_x2 = xs.col(2) - xs.col(0);
    Eigen::Vector3d ntilde = delta_x1.cross(delta_x2);
    Eigen::Vector3d n = ntilde.normalized();
    xs.col(3) = n;
    Eigen::Matrix43d r;
    Eigen::Matrix3d dphi;
    Eigen::Vector3d X;  // doesn't actually matter, gradient is constant over triangle
    dphi_cloth_triangle_dX(dphi, V, element, X);
    r.block<3, 3>(0, 0) = dphi;
    Eigen::Vector3d delta_X1 = V.row(x1) - V.row(x0);
    Eigen::Vector3d delta_X2 = V.row(x2) - V.row(x0);
    Eigen::Vector3d N = delta_X1.cross(delta_X2).normalized();
    r.block<1, 3>(3, 0) = N;
    Eigen::Matrix3d F = xs * r;

    //SVD = USW^T
    Eigen::Matrix3d U;
    Eigen::Vector3d S;
    Eigen::Matrix3d W;

    double tol = 1e-5;

    // get singular values of F
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U = svd.matrixU();
    W = svd.matrixV();
    S = svd.singularValues();
    Eigen::Tensor3333d dU, dW;
    Eigen::Tensor333d dS;
    dsvd(dU, dS, dW, F);

    //deal with singularity in the svd gradient
    if (std::fabs(S[0] - S[1]) < tol || std::fabs(S[1] - S[2]) < tol || std::fabs(S[0] - S[2]) < tol) {
        F += Eigen::Matrix3d::Random() * tol;
        Eigen::JacobiSVD<Eigen::Matrix3d> svd2(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
        U = svd2.matrixU();
        W = svd2.matrixV();
        S = svd2.singularValues();
    }

    //Fix for inverted elements (thanks to Danny Kaufman)
    double det = S[0] * S[1];

    if (det <= -1e-10) {
        if (S[0] < 0) S[0] *= -1;
        if (S[1] < 0) S[1] *= -1;
        if (S[2] < 0) S[2] *= -1;
    }

    if (U.determinant() <= 0) {
        U(0, 2) *= -1;
        U(1, 2) *= -1;
        U(2, 2) *= -1;
    }

    if (W.determinant() <= 0) {
        W(0, 2) *= -1;
        W(1, 2) *= -1;
        W(2, 2) *= -1;
    }
    // calculate derivatives
    double s0 = S(0), s1 = S(1), s2 = S(2);
    Eigen::Matrix3d dpsi_ds = Eigen::Matrix3d::Zero();
    dpsi_ds(0, 0) = lambda * (s0 + s1 + s2 - 3.) + mu * 2 * (s0 - 1.);
    dpsi_ds(1, 1) = lambda * (s0 + s1 + s2 - 3.) + mu * 2 * (s1 - 1.);
    dpsi_ds(2, 2) = lambda * (s0 + s1 + s2 - 3.) + mu * 2 * (s2 - 1.);
    Eigen::Matrix3d d2psi_ds2 = Eigen::Matrix3d::Zero();
    d2psi_ds2(0, 0) = lambda + mu * 2.;
    d2psi_ds2(0, 1) = lambda;
    d2psi_ds2(0, 2) = lambda;
    d2psi_ds2(1, 0) = lambda;
    d2psi_ds2(1, 1) = lambda + mu * 2.;
    d2psi_ds2(1, 2) = lambda;
    d2psi_ds2(2, 0) = lambda;
    d2psi_ds2(2, 1) = lambda;
    d2psi_ds2(2, 2) = lambda + mu * 2.;
    // compute d2psi_dF2 based on formula from lecture
    Eigen::Matrix99d d2psi_dF2 = Eigen::Matrix99d::Zero();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Eigen::Vector3d delta_s_non_diag = d2psi_ds2 * dS[i][j];
            d2psi_dF2.block<3, 3>(i * 3, j * 3) = dU[i][j] * dpsi_ds * W.transpose()
                                                  + U * delta_s_non_diag.asDiagonal() * W.transpose()
                                                  + U * dpsi_ds * dW[i][j].transpose();
        }
    }

    // create matrix based on derivative of shape functions
    Eigen::Matrix99d B = Eigen::Matrix99d::Zero();
    B.block<3, 1>(0, 0) = dphi.row(0);
    B.block<3, 1>(3, 1) = dphi.row(0);
    B.block<3, 1>(6, 2) = dphi.row(0);
    B.block<3, 1>(0, 3) = dphi.row(1);
    B.block<3, 1>(3, 4) = dphi.row(1);
    B.block<3, 1>(6, 5) = dphi.row(1);
    B.block<3, 1>(0, 6) = dphi.row(2);
    B.block<3, 1>(3, 7) = dphi.row(2);
    B.block<3, 1>(3, 8) = dphi.row(2);
    // create matrix of N as in lecture
    Eigen::Matrix93d N_matrix = Eigen::Matrix93d::Zero();
    N_matrix.block<3, 1>(0, 0) = N;
    N_matrix.block<3, 1>(3, 1) = N;
    N_matrix.block<3, 1>(6, 2) = N;
    // create Nu matrix as in lecture
    Eigen::Matrix39d c1 = Eigen::Matrix39d::Zero();
    c1.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    c1.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
    Eigen::Matrix39d c2 = Eigen::Matrix39d::Zero();
    c2.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    c2.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    Eigen::Matrix39d Nu = 1 / ntilde.norm() * (Eigen::Matrix3d::Identity() - n * n.transpose())
                          * (cross_product_matrix_2(delta_x1) * c1 - cross_product_matrix_2(delta_x2) * c2);
    Eigen::Matrix99d dF_dq = B + N_matrix * Nu;
    // including thickness factor 1 * as a reminder that our model is volumetric
    H = 1 * area * dF_dq.transpose() * d2psi_dF2 * dF_dq;

    //fix errant eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix99d> es(H);

    Eigen::MatrixXd DiagEval = es.eigenvalues().real().asDiagonal();
    Eigen::MatrixXd Evec = es.eigenvectors().real();

    for (int i = 0; i < 9; ++i) {
        if (es.eigenvalues()[i] < 1e-6) {
            DiagEval(i, i) = 1e-3;
        }
    }

    H = Evec * DiagEval * Evec.transpose();

}

// TODO: figure out function naming business
// return a cross product matrix for a given vector, i.e. given v return [v] so that [v] * w = v.cross(w)
Eigen::Matrix3d cross_product_matrix_2(const Eigen::Ref<const Eigen::Vector3d> &v) {
    Eigen::Matrix3d out = Eigen::Matrix3d::Zero();
    out(0, 1) = -v(2);
    out(1, 0) = v(2);
    out(0, 2) = v(1);
    out(2, 0) = -v(1);
    out(1, 2) = -v(0);
    out(2, 1) = v(0);
    return out;
}