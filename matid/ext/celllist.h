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

#ifndef CELLLIST_H
#define CELLLIST_H

#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;
using namespace std;

struct CellListResult {
    vector<int> indices;
    vector<double> distances;
    vector<double> distances_squared;
    vector<vector<double>> displacements;
    vector<int> indices_original;
    vector<vector<double>> factors;
};

/**
 * For calculating pairwise distances using a cell lists:
 * https://en.wikipedia.org/wiki/Cell_lists.
 */
class CellList {
    public:
        /**
         * Constructor
         *
         * @param positions Atomic positions in cartesian coordinates.
         * @param atomicNumbers Atomic numbers.
         */
        CellList(py::array_t<double> positions, py::array_t<int> indices, py::array_t<double> factors, double cutoff);
        /**
         * Get the indices of atoms within the radial cutoff distance from the
         * given position.
         *
         * @param x Cartesian x-coordinate.
         * @param y Cartesian y-coordinate.
         * @param z Cartesian z-coordinate.
         */
        CellListResult get_neighbours_for_position(const double x, const double y, const double z);
        /**
         * Get the indices of atoms within the radial cutoff distance from the
         * given atomic index. The given index is not included in the returned
         * values.
         *
         * @param i Index of the atom for which neighbours are queried for.
         */
        CellListResult get_neighbours_for_index(const int i);
        /**
         * Used to calculate the full displacement tensor in an efficient
         * manner.
         */
        void get_displacement_tensor(
            py::array_t<double> displacements,
            py::array_t<double> distances,
            py::array_t<double> factors,
            py::array_t<int> original_indices,
            int n_atoms
        );
        /**
         * Used to calculate a sparse minimum-image neighbour list within the
         * cutoff. Only the pairs (i, j) with j < i are appended into the flat
         * arrays, ordered by i and then by j. The displacement is pos_i -
         * pos_j and the factor is the periodic copy of j. The diagonal is
         * omitted.
         */
        void get_displacement_list(
            py::array_t<int> original_indices,
            int n_atoms,
            vector<int>& row,
            vector<int>& col,
            vector<double>& distance,
            vector<double>& displacement,
            vector<int>& factor
        );
        py::array_t<int> indices_py;

    private:
        /**
         * Used to initialize the cell list. Querying for distances is only
         * possible after this initialization.
         */
        void init();

        // const py::detail::unchecked_reference<double, 2> positions;
        // const py::detail::unchecked_reference<int, 1> indices;
        // const py::detail::unchecked_reference<double, 2> factors;
        vector<vector<double>> positions;
        vector<int> indices;
        vector<vector<double>> factors;
        const double cutoff;
        const double cutoffSquared;
        double xmin;
        double xmax;
        double ymin;
        double ymax;
        double zmin;
        double zmax;
        double dx;
        double dy;
        double dz;
        int nx;
        int ny;
        int nz;
        vector<vector<vector<vector<int>>>> bins;
};

#endif
