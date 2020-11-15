#include <mass_matrix_mesh.h>

void mass_matrix_mesh(Eigen::SparseMatrixd &M, Eigen::Ref<const Eigen::VectorXd> q,
                      Eigen::Ref<const Eigen::MatrixXd> V, Eigen::Ref<const Eigen::MatrixXi> F,
                      double density, Eigen::Ref<const Eigen::VectorXd> areas) {
    std::vector<Eigen::Triplet<double>> tl;
    tl.reserve(27 * F.rows());
    for (int tri = 0; tri < F.rows(); tri++) {
        // build mass matrix for current triangle
        Eigen::Matrix99d M_curr = Eigen::Matrix99d::Zero();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                // * 1 to remind us that our model is actually volumetric with a thickness of 1
                M_curr.block<3, 3>(i * 3, j * 3)
                        = Eigen::Matrix3d::Identity() * density * areas(tri) * 1. / 12.;
            }
        }
        // every entry is density * area / 12, except diagonal which is density * area / 6
        M_curr.diagonal() *= 2;
        for (int tri_row = 0; tri_row < 3; tri_row++) {
            for (int tri_col = 0; tri_col < 3; tri_col++) {
                // i loop: iterate over diagonal of 3x3 block for tri_row, tri_col in triangle mass matrix
                // only need to iterate over diagonal because each 3x3 block is diagonal
                for (int i = 0; i < 3; i++) {
                    tl.emplace_back(F(tri, tri_row) * 3 + i, F(tri, tri_col) * 3 + i,
                                    M_curr(tri_row * 3 + i, tri_col * 3 + i));
                }
            }
        }
    }
    M.resize(3 * V.rows(), 3 * V.rows());
    M.setFromTriplets(tl.begin(), tl.end());
}
 