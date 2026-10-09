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
#include "celllist.h"
#include <algorithm>
#include <utility>
#include <tuple>
#include <map>
#include <unordered_map>
#include <utility>
#include <math.h>

using namespace std;

/**
 * @brief Construct a new CellList
 *
 * @param positions Atomic positions as a [n_atoms, 3] array.
 * @param cutoff The radius for considering atoms to be connected. Set to 0 to
 * build full connectivity.
 */
CellList::CellList(py::array_t<double> positions, py::array_t<int> indices, py::array_t<double> factors, double cutoff)
    : indices_py(indices)
    , cutoff(cutoff)
    , cutoffSquared(cutoff*cutoff)
{
    if (cutoff <= 0) {
        throw invalid_argument("Cell list cutoff must be positive.");
    }

    // Copy the numpy data, as the numpy memory may get cleaned!
    auto positions_u = positions.unchecked<2>();
    for (int i = 0; i < positions_u.shape(0); ++i) {
        vector<double> row(positions_u.shape(1));
        for (int j = 0; j < positions_u.shape(1); ++j) {
            row[j] = positions_u(i, j);
        }
        this->positions.push_back(row);
    }
    auto factors_u = factors.unchecked<2>();
    for (int i = 0; i < factors_u.shape(0); ++i) {
        vector<double> row(factors_u.shape(1));
        for (int j = 0; j < factors_u.shape(1); ++j) {
            row[j] = factors_u(i, j);
        }
        this->factors.push_back(row);
    }
    auto indices_u = indices.unchecked<1>();
    vector<int> row(indices_u.shape(0));
    for (int i = 0; i < indices_u.shape(0); ++i) {
        row[i] = indices_u(i);
    }
    this->indices = row;

    this->init();
}

void CellList::init() {
    // Find cell limits
    this->xmin = this->xmax = this->positions[0][0];
    this->ymin = this->ymax = this->positions[0][1];
    this->zmin = this->zmax = this->positions[0][2];
    for (int i = 0; i < this->positions.size(); i++) {
        double x = this->positions[i][0];
        double y = this->positions[i][1];
        double z = this->positions[i][2];
        if (x < this->xmin) {
            this->xmin = x;
        };
        if (x > this->xmax) {
            this->xmax = x;
        };
        if (y < this->ymin) {
            this->ymin = y;
        };
        if (y > this->ymax) {
            this->ymax = y;
        };
        if (z < this->zmin) {
            this->zmin = z;
        };
        if (z > this->zmax) {
            this->zmax = z;
        };
    };

    // Add small padding to avoid floating point precision problems at the
    // boundary
    double padding = 0.0001;
    this->xmin -= padding;
    this->xmax += padding;
    this->ymin -= padding;
    this->ymax += padding;
    this->zmin -= padding;
    this->zmax += padding;

    // Determine amount and size of bins. The bins are made to be always of equal size.
    this->nx = max(1, int((this->xmax - this->xmin)/this->cutoff));
    this->ny = max(1, int((this->ymax - this->ymin)/this->cutoff));
    this->nz = max(1, int((this->zmax - this->zmin)/this->cutoff));
    this->dx = max(this->cutoff, (this->xmax - this->xmin)/this->nx);
    this->dy = max(this->cutoff, (this->ymax - this->ymin)/this->ny);
    this->dz = max(this->cutoff, (this->zmax - this->zmin)/this->nz);

    // Initialize the bin data structure. It is a 4D vector.
    this->bins = vector<vector<vector<vector<int>>>>(this->nx, vector<vector<vector<int>>>(this->ny, vector<vector<int>>(this->nz, vector<int>())));

    // Fill the bins with atom indices
    for (int idx = 0; idx < this->positions.size(); idx++) {
        double x = this->positions[idx][0];
        double y = this->positions[idx][1];
        double z = this->positions[idx][2];

        // Get bin index
        int i = (x - this->xmin)/this->dx;
        int j = (y - this->ymin)/this->dy;
        int k = (z - this->zmin)/this->dz;

        // Add atom index to the bin
        this->bins[i][j][k].push_back(idx);
    };
}

CellListResult CellList::get_neighbours_for_position(
    const double x,
    const double y,
    const double z
) {
    // The indices of the neighbouring atoms
    vector<int> neighbours;
    vector<double> distances;
    vector<double> distances_squared;
    vector<vector<double>> displacements;
    vector<vector<double>> factors;
    vector<int> indices_original;

    // Find bin for the given position
    int i0 = (x - this->xmin)/this->dx;
    int j0 = (y - this->ymin)/this->dy;
    int k0 = (z - this->zmin)/this->dz;

    // Get the bin ranges to check for each dimension.
    int istart = max(i0-1, 0);
    int iend = min(i0+1, this->nx-1);
    int jstart = max(j0-1, 0);
    int jend = min(j0+1, this->ny-1);
    int kstart = max(k0-1, 0);
    int kend = min(k0+1, this->nz-1);

    // Loop over neighbouring bins
    for (int i = istart; i <= iend; i++) {
        for (int j = jstart; j <= jend; j++) {
            for (int k = kstart; k <= kend; k++) {

                // For each atom in the current bin, calculate the actual distance
                vector<int> binIndices = this->bins[i][j][k];
                for (auto &idx : binIndices) {
                    double ix = this->positions[idx][0];
                    double iy = this->positions[idx][1];
                    double iz = this->positions[idx][2];
                    double deltax = x - ix;
                    double deltay = y - iy;
                    double deltaz = z - iz;
                    double distance_squared = deltax*deltax + deltay*deltay + deltaz*deltaz;
                    if (distance_squared <= this->cutoffSquared) {
                        neighbours.push_back(idx);
                        indices_original.push_back(this->indices[idx]);
                        distances.push_back(sqrt(distance_squared));
                        distances_squared.push_back(distance_squared);
                        displacements.push_back(vector<double>{deltax, deltay, deltaz});
                        factors.push_back(this->factors[idx]);
                    }
                }
            }
        }
    }
    return CellListResult{neighbours, distances, distances_squared, displacements, indices_original, factors};
}

CellListResult CellList::get_neighbours_for_index(const int idx)
{
    double x = this->positions[idx][0];
    double y = this->positions[idx][1];
    double z = this->positions[idx][2];
    CellListResult result = this->get_neighbours_for_position(x, y, z);

    return result;
}

void CellList::get_displacement_tensor(
    py::array_t<double> displacements,
    py::array_t<double> distances,
    py::array_t<double> factors,
    py::array_t<int> original_indices,
    int n_atoms
)
{
    auto original_indices_u = original_indices.unchecked<1>();
    auto distances_mu = distances.mutable_unchecked<2>();
    auto displacements_mu = displacements.mutable_unchecked<3>();
    auto factors_mu = factors.mutable_unchecked<3>();

    for (int i = 0; i < n_atoms; ++i) {

        // Report self-distance as zero
        distances_mu(i, i) = 0;
        for (int k=0; k < 3; ++k) {
            displacements_mu(i, i, k) = 0;
            factors_mu(i, i, k) = 0;
        }

        // Find bin for the given position
        double x = this->positions[i][0];
        double y = this->positions[i][1];
        double z = this->positions[i][2];
        int i0 = (x - this->xmin)/this->dx;
        int j0 = (y - this->ymin)/this->dy;
        int k0 = (z - this->zmin)/this->dz;

        // Get the bin ranges to check for each dimension.
        int istart = max(i0-1, 0);
        int iend = min(i0+1, this->nx-1);
        int jstart = max(j0-1, 0);
        int jend = min(j0+1, this->ny-1);
        int kstart = max(k0-1, 0);
        int kend = min(k0+1, this->nz-1);

        // Loop over neighbouring bins
        unordered_map<int, tuple<double, vector<double>, vector<double>>> min_map;
        for (int i_bin = istart; i_bin <= iend; i_bin++) {
            for (int j_bin = jstart; j_bin <= jend; j_bin++) {
                for (int k_bin = kstart; k_bin <= kend; k_bin++) {
                    vector<int> binIndices = this->bins[i_bin][j_bin][k_bin];
                    for (auto &idx : binIndices) {
                        int j = original_indices_u(idx);

                        // Unnecessary computation is skipped
                        if (j >= i) {
                            continue;
                        }
                        double ix = this->positions[idx][0];
                        double iy = this->positions[idx][1];
                        double iz = this->positions[idx][2];
                        double deltax = x - ix;
                        double deltay = y - iy;
                        double deltaz = z - iz;
                        double distance_squared = deltax*deltax + deltay*deltay + deltaz*deltaz;

                        // If distance is smaller than cutoff, and the found
                        // distance is smallest for this index, it is saved
                        if (distance_squared <= this->cutoffSquared) {
                            double distance = sqrt(distance_squared);
                            if (min_map.find(j) == min_map.end() || distance < get<0>(min_map[j])) {
                                min_map[j] = tuple<double, vector<double>, vector<double>>{distance, vector<double>{deltax, deltay, deltaz}, this->factors[idx]};
                            }
                        }
                    }
                }
            }
        }

        for (auto& it: min_map) {
            double distance = get<0>(it.second);
            vector<double> displacement = get<1>(it.second);
            vector<double> factor = get<2>(it.second);
            distances_mu(i, it.first) = distance;
            distances_mu(it.first, i) = distance;
            for (int k=0; k < 3; ++k) {
                displacements_mu(i, it.first, k) = displacement[k];
                displacements_mu(it.first, i, k) = -displacement[k];
                factors_mu(i, it.first, k) = factor[k];
                factors_mu(it.first, i, k) = -factor[k];
            }
        }
    }
}

void CellList::get_displacement_list(
    py::array_t<int> original_indices,
    int n_atoms,
    vector<int>& row,
    vector<int>& col,
    vector<double>& distance,
    vector<double>& displacement,
    vector<int>& factor
)
{
    auto original_indices_u = original_indices.unchecked<1>();

    // The minimum image for each neighbour is tracked in a flat buffer that is
    // reused for every atom. min_map maps the original index of a neighbour to
    // its position in the buffer.
    struct Neighbour {
        int j;
        double dist;
        double disp[3];
        int idx;
    };
    vector<Neighbour> found;
    unordered_map<int, int> min_map;

    // This mirrors get_displacement_tensor exactly, but instead of filling a
    // dense [n_atoms, n_atoms] buffer it appends the found minimum-image
    // neighbours into flat arrays.
    for (int i = 0; i < n_atoms; ++i) {
        found.clear();
        min_map.clear();

        // Find bin for the given position
        double x = this->positions[i][0];
        double y = this->positions[i][1];
        double z = this->positions[i][2];
        int i0 = (x - this->xmin)/this->dx;
        int j0 = (y - this->ymin)/this->dy;
        int k0 = (z - this->zmin)/this->dz;

        // Get the bin ranges to check for each dimension.
        int istart = max(i0-1, 0);
        int iend = min(i0+1, this->nx-1);
        int jstart = max(j0-1, 0);
        int jend = min(j0+1, this->ny-1);
        int kstart = max(k0-1, 0);
        int kend = min(k0+1, this->nz-1);

        // Loop over neighbouring bins
        for (int i_bin = istart; i_bin <= iend; i_bin++) {
            for (int j_bin = jstart; j_bin <= jend; j_bin++) {
                for (int k_bin = kstart; k_bin <= kend; k_bin++) {
                    const vector<int>& binIndices = this->bins[i_bin][j_bin][k_bin];
                    for (auto &idx : binIndices) {
                        int j = original_indices_u(idx);

                        // Unnecessary computation is skipped
                        if (j >= i) {
                            continue;
                        }
                        double ix = this->positions[idx][0];
                        double iy = this->positions[idx][1];
                        double iz = this->positions[idx][2];
                        double deltax = x - ix;
                        double deltay = y - iy;
                        double deltaz = z - iz;
                        double distance_squared = deltax*deltax + deltay*deltay + deltaz*deltaz;

                        // If distance is smaller than cutoff, and the found
                        // distance is smallest for this index, it is saved
                        if (distance_squared <= this->cutoffSquared) {
                            double dist = sqrt(distance_squared);
                            Neighbour neighbour = {j, dist, {deltax, deltay, deltaz}, idx};
                            auto it = min_map.find(j);
                            if (it == min_map.end()) {
                                min_map.emplace(j, (int)found.size());
                                found.push_back(neighbour);
                            } else if (dist < found[it->second].dist) {
                                found[it->second] = neighbour;
                            }
                        }
                    }
                }
            }
        }

        // Emit the neighbours ordered by their index
        sort(found.begin(), found.end(), [](const Neighbour& a, const Neighbour& b) {
            return a.j < b.j;
        });
        for (auto& neighbour: found) {
            row.push_back(i);
            col.push_back(neighbour.j);
            distance.push_back(neighbour.dist);
            const vector<double>& fac = this->factors[neighbour.idx];
            for (int k=0; k < 3; ++k) {
                displacement.push_back(neighbour.disp[k]);
                factor.push_back((int)lround(fac[k]));
            }
        }
    }
}
