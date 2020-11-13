#include <dV_membrane_corotational_dq.h>
#include "dphi_cloth_triangle_dX.h"
#include <iostream>

void dV_membrane_corotational_dq(Eigen::Vector9d &dV, Eigen::Ref<const Eigen::VectorXd> q,
                                 Eigen::Ref<const Eigen::Matrix3d> dX,
                                 Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::RowVectorXi> element,
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
    xs.col(3) = (xs.col(1) - xs.col(0)).cross(xs.col(2) - xs.col(0));
    Eigen::Matrix43d r;
    Eigen::Matrix3d dphi;
    Eigen::Vector3d X;  // doesn't actually matter, gradient is constant over triangle
    dphi_cloth_triangle_dX(dphi, V, element, X);
    r.block<3, 3>(0, 0) = dphi;
    Eigen::Vector3d delta_x1 = V.row(x1) - V.row(x0);
    Eigen::Vector3d delta_x2 = V.row(x2) - V.row(x0);
    r.block<1, 3>(3, 0) = (delta_x1).cross(delta_x2);
    Eigen::Matrix3d F = xs * r;

    Eigen::Matrix3d dx; //deformed tangent matrix 
    Eigen::Matrix3d U;
    Eigen::Vector3d S;
    Eigen::Matrix3d W;

    // get singular values of F
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    U = svd.matrixU();
    W = svd.matrixV();
    S = svd.singularValues();

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

    // calculate dpsi_dF using singular value decomposition derivative formula
    double s0 = S(0), s1 = S(1), s2 = S(2);
    Eigen::Matrix3d dS = Eigen::Matrix3d::Zero();
    dS(0, 0) = lambda * (s0 + s1 + s2 - 3.) + mu * 2 * (s0 - 1.);
    dS(1, 1) = lambda * (s0 + s1 + s2 - 3.) + mu * 2 * (s1 - 1.);
    dS(2, 2) = lambda * (s0 + s1 + s2 - 3.) + mu * 2 * (s2 - 1.);
    Eigen::Matrix3d dpsi_dF = U * dS * V.transpose();
    Eigen::Vector9d dpsi_vector;
    dpsi_vector.segment(0, 3) = dpsi_dF.block<3, 1>(0, 0);
    dpsi_vector.segment(3, 3) = dpsi_dF.block<3, 1>(1, 0);
    dpsi_vector.segment(6, 3) = dpsi_dF.block<3, 1>(2, 0);

    // B = dF/dq (taken from last assignment)
    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(9, 12);
    B.block<3, 1>(0, 0) = dphi.block<1, 3>(0, 0);
    B.block<3, 1>(3, 1) = dphi.block<1, 3>(0, 0);
    B.block<3, 1>(6, 2) = dphi.block<1, 3>(0, 0);
    B.block<3, 1>(0, 3) = dphi.block<1, 3>(1, 0);
    B.block<3, 1>(3, 4) = dphi.block<1, 3>(1, 0);
    B.block<3, 1>(6, 5) = dphi.block<1, 3>(1, 0);
    B.block<3, 1>(0, 6) = dphi.block<1, 3>(2, 0);
    B.block<3, 1>(3, 7) = dphi.block<1, 3>(2, 0);
    B.block<3, 1>(6, 8) = dphi.block<1, 3>(2, 0);
    B.block<3, 1>(0, 9) = dphi.block<1, 3>(3, 0);
    B.block<3, 1>(3, 10) = dphi.block<1, 3>(3, 0);
    B.block<3, 1>(6, 11) = dphi.block<1, 3>(3, 0);
    // including thickness factor 1 * as a reminder that our model is volumetric
    dV = 1 * area * B.transpose() * dpsi_vector;
}
