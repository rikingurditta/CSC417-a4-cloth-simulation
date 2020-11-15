#include <dV_membrane_corotational_dq.h>
#include "dphi_cloth_triangle_dX.h"
#include <iostream>

Eigen::Matrix3d cross_product_matrix(const Eigen::Ref<const Eigen::Vector3d> &v);

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
    Eigen::Vector3d delta_x1 = xs.col(1) - xs.col(0);
    Eigen::Vector3d delta_x2 = xs.col(2) - xs.col(0);
    Eigen::Vector3d ntilde = delta_x1.cross(delta_x2);
    Eigen::Vector3d n = ntilde.normalized();
    xs.col(3) = n;
    Eigen::Matrix43d r;
    r.block<3, 3>(0, 0) = dX;
    Eigen::Vector3d delta_X1 = V.row(x1) - V.row(x0);
    Eigen::Vector3d delta_X2 = V.row(x2) - V.row(x0);
    Eigen::Vector3d N = delta_X1.cross(delta_X2).normalized();
    r.row(3) = N;
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
    // cloth has constant thickness, so principle stretch s2 (in normal direction) is 1
    double s0 = S(0), s1 = S(1), s2 = 1.;
    Eigen::Matrix3d dpsi_ds = Eigen::Matrix3d::Zero();
    dpsi_ds(0, 0) = lambda * (s0 + s1 + s2 - 3.) + mu * 2 * (s0 - 1.);
    dpsi_ds(1, 1) = lambda * (s0 + s1 + s2 - 3.) + mu * 2 * (s1 - 1.);
    dpsi_ds(2, 2) = 0;  // since s2 is constant, its derivative is 0
    Eigen::Matrix3d dpsi_dF = U * dpsi_ds * W.transpose();
    // flatten dpsi_dF row wise
    Eigen::Vector9d dpsi_vector;
    dpsi_vector.segment(0, 3) = dpsi_dF.row(0);
    dpsi_vector.segment(3, 3) = dpsi_dF.row(1);
    dpsi_vector.segment(6, 3) = dpsi_dF.row(2);

    // create matrix based on derivative of shape functions
    Eigen::Matrix99d B = Eigen::Matrix99d::Zero();
    B.block<3, 1>(0, 0) = dX.row(0);
    B.block<3, 1>(3, 1) = dX.row(0);
    B.block<3, 1>(6, 2) = dX.row(0);
    B.block<3, 1>(0, 3) = dX.row(1);
    B.block<3, 1>(3, 4) = dX.row(1);
    B.block<3, 1>(6, 5) = dX.row(1);
    B.block<3, 1>(0, 6) = dX.row(2);
    B.block<3, 1>(3, 7) = dX.row(2);
    B.block<3, 1>(6, 8) = dX.row(2);
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
    Eigen::Matrix39d Nu = 1. / ntilde.norm() * (Eigen::Matrix3d::Identity() - n * n.transpose())
                          * (cross_product_matrix(delta_x1) * c1 - cross_product_matrix(delta_x2) * c2);

    // including thickness factor 1 * as a reminder that our model is volumetric
    dV = 1. * area * (B + N_matrix * Nu).transpose() * dpsi_vector;
}

// return a cross product matrix for a given vector, i.e. given v return [v] so that [v] * w = v.cross(w)
Eigen::Matrix3d cross_product_matrix(const Eigen::Ref<const Eigen::Vector3d> &v) {
    Eigen::Matrix3d out = Eigen::Matrix3d::Zero();
    out(0, 1) = -v(2);
    out(1, 0) = v(2);
    out(0, 2) = v(1);
    out(2, 0) = -v(1);
    out(1, 2) = -v(0);
    out(2, 1) = v(0);
    return out;
}