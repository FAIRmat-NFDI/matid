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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>  // Enables easy access to numpy arrays
#include <pybind11/stl.h>    // Enables automatic type conversion from C++ containers to python
#include "celllist.h"
#include "geometry.h"

namespace py = pybind11;
using namespace std;

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

// Notice that the name of the first argument to the module macro needs to
// correspond to the file name!
PYBIND11_MODULE(ext, m) {
    // Geometry
    m.def("extend_system", &extend_system, "Create a periodically extended system.");
    py::class_<ExtendedSystem>(m, "ExtendedSystem", py::module_local())
        .def(py::init<>())
        .def_readonly("positions", &ExtendedSystem::positions)
        .def_readonly("atomic_numbers", &ExtendedSystem::atomic_numbers)
        .def_readonly("indices", &ExtendedSystem::indices)
        .def_readonly("factors", &ExtendedSystem::factors);
    m.def("get_cell_list", &get_cell_list, "Get cell list for system.");
    m.def("get_displacement_tensor", &get_displacement_tensor, "Get displacement vectors respecting minimum image convention.");
    m.def("get_displacement_list", &get_displacement_list, "Get a sparse minimum-image neighbour list within a finite cutoff.");
    py::class_<SparseDistances>(m, "SparseDistances", py::module_local())
        .def(py::init<>())
        .def_readonly("row_ptr", &SparseDistances::row_ptr)
        .def_readonly("col", &SparseDistances::col)
        .def_readonly("distance", &SparseDistances::distance)
        .def_readonly("displacement", &SparseDistances::displacement)
        .def_readonly("factor", &SparseDistances::factor);

    // CellList
    py::class_<CellList>(m, "CellList", py::module_local())
        .def(py::init<py::array_t<double>, py::array_t<int>, py::array_t<double>, double>())
        .def("get_neighbours_for_index", &CellList::get_neighbours_for_index)
        .def("get_neighbours_for_position", &CellList::get_neighbours_for_position);
    py::class_<CellListResult>(m, "CellListResult", py::module_local())
        .def(py::init<>())
        .def_readonly("indices", &CellListResult::indices)
        .def_readonly("indices_original", &CellListResult::indices_original)
        .def_readonly("distances", &CellListResult::distances)
        .def_readonly("distances_squared", &CellListResult::distances_squared)
        .def_readonly("displacements", &CellListResult::displacements)
        .def_readonly("factors", &CellListResult::factors);
}
