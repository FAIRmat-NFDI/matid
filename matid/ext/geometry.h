/*Copyright 2019 DScribe developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <vector>
#include <stdexcept>
#include <cstdint>
#include <pybind11/numpy.h>
#include "celllist.h"

namespace py = pybind11;
using namespace std;

struct ExtendedSystem {
    py::array_t<double> positions;
    py::array_t<int> atomic_numbers;
    py::array_t<int> indices;
    py::array_t<double> factors;
};

struct SparseDistances {
    py::array_t<int64_t> row_ptr;     // shape (n_atoms + 1,): CSR row offsets
    py::array_t<int> col;             // shape (nnz,), ascending within each row
    py::array_t<double> distance;     // shape (nnz,)
    py::array_t<double> displacement; // shape (nnz, 3): pos_row - pos_col (min image)
    py::array_t<int> factor;          // shape (nnz, 3)
};

inline vector<double> cross(const vector<double>& a, const vector<double>& b);
inline double dot(const vector<double>& a, const vector<double>& b);
inline double norm(const vector<double>& a);

/**
 * Used to periodically extend an atomic system in order to take into account
 * periodic copies beyond the given unit cell.
 *
 * @param positions Cartesian positions of the original system.
 * @param atomic_numbers Atomic numbers of the original system.
 * @param cell Unit cell of the original system.
 * @param pbc Periodic boundary conditions (array of three booleans) of the original system.
 * @param cutoff Radial cutoff value for determining extension size.
 *
 * @return Instance of ExtendedSystem.
 */
ExtendedSystem extend_system(
    py::array_t<double> positions,
    py::array_t<int> atomic_numbers,
    py::array_t<double> cell,
    py::array_t<bool> pbc,
    double cutoff
);

/**
 * Returns a CellList instance for the given atomic system.
 *
 * Note that you should control how large extension should be performed to take
 * into account the periodic boundary conditions, as the CellList internally
 * works with plain cartesian coordinates.
 */
CellList get_cell_list(
    py::array_t<double> positions,
    py::array_t<double> cell,
    py::array_t<bool> pbc,
    double extension,
    double cutoff
);

/**
 * Calculates a pairwise displacement tensor (distance vectors) with a given
 * cutoff.
 */
void get_displacement_tensor(
    py::array_t<double> displacements,
    py::array_t<double> distances,
    py::array_t<double> factors,
    py::array_t<double> positions,
    py::array_t<double> cell,
    py::array_t<bool> pbc,
    double cutoff,
    bool return_factors,
    bool return_distances
);

/**
 * Calculates a sparse minimum-image neighbour list (CSR format) for all pairs
 * within the given finite cutoff. Both pair directions are included. Much cheaper than the dense displacement
 * tensor when only local distances are needed.
 */
SparseDistances get_displacement_list(
    py::array_t<double> positions,
    py::array_t<double> cell,
    py::array_t<bool> pbc,
    double cutoff
);

#endif
