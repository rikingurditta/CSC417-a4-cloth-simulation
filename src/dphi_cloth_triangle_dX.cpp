#include <dphi_cloth_triangle_dX.h>

//compute 3x3 deformation gradient 
void dphi_cloth_triangle_dX(Eigen::Matrix3d &dphi, Eigen::Ref<const Eigen::MatrixXd> V,
                            Eigen::Ref<const Eigen::RowVectorXi> element, Eigen::Ref<const Eigen::Vector3d> X) {
    Eigen::Matrix32d T;
    T.col(0) = V.row(element(1)) - V.row(element(0));
    T.col(1) = V.row(element(2)) - V.row(element(0));
    // TTT = (T^T T)^-1 T^T
    Eigen::MatrixXd TTT = (T.transpose() * T).inverse() * T.transpose();
    // first row of dphi is -1^T TTT so that shape functions add to 1
    dphi.block<1, 3>(0, 0) = -Eigen::RowVector2d::Ones() * TTT;
    // second and third rows are just TTT
    dphi.block<2, 3>(1, 0) = TTT;
}