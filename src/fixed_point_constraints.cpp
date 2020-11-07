#include <fixed_point_constraints.h>
#include <algorithm>
#include <unordered_set>

void fixed_point_constraints(Eigen::SparseMatrixd &P, unsigned int q_size, const std::vector<unsigned int> indices) {
    // P has 3 entries for each non-fixed vertex, aka 3 * total vertices - 3 * fixed vertices
    P.resize(q_size - indices.size() * 3, q_size);
    // change fixed vertices to zeros
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(q_size - indices.size() * 3);
    // make set of vertices that shouldn't be included
    std::unordered_set<int> indices_set(indices.begin(), indices.end());
    int row = 0;
    for (int i = 0; i < q_size / 3; i++) {
        if (indices_set.find(i) == indices_set.end()) {
            // keep vertex if not in set of exclusions
            int vi = i * 3;
            tripletList.emplace_back(row, vi, 1.);
            tripletList.emplace_back(row + 1, vi + 1, 1.);
            tripletList.emplace_back(row + 2, vi + 2, 1.);
            row += 3;
        }
    }
    P.setFromTriplets(tripletList.begin(), tripletList.end());
}