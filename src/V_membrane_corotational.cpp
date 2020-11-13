#include <V_membrane_corotational.h>
#include "dphi_cloth_triangle_dX.h"

//Allowed to use libigl SVD or Eigen SVD for this part
void V_membrane_corotational(double &energy, Eigen::Ref<const Eigen::VectorXd> q, Eigen::Ref<const Eigen::Matrix3d> dX,
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
    // get singular values of F
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    double s0 = svd.singularValues()(0);
    double s1 = svd.singularValues()(1);
    double s2 = svd.singularValues()(2);
    // calculate corotational linear elasticity energy
    // including thickness factor 1 * to remind us that our cloth model is volumetric
    energy = 1 * area * (mu * (pow(s0, 2) + pow(s1, 2) + pow(s2, 2))
                         + lambda / 2 * pow(s0 + s1 + s2 - 3, 2));
}
