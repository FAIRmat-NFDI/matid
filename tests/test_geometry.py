import time
from pathlib import Path

import numpy as np
import pytest
import ase.lattice.hexagonal
import ase.lattice.cubic
import ase.io
from ase.build import molecule, nanotube, bcc100, bulk
from ase import Atoms
from ase.visualize import view

import matid.geometry

# 0D system
sys0d = molecule("H2O")
sys0d.set_cell([3, 3, 3])
sys0d.center()

# 1D system
sys1d = nanotube(3, 3, length=6, bond=1.4, symbol="Si")
sys1d.set_cell((10, 10, 15))
sys1d.center()

# 2D system
sys2d = Atoms(
    symbols=[6, 6],
    cell=np.array(
        (
            [2.4595121467478055, 0.0, 0.0],
            [-1.2297560733739028, 2.13, 0.0],
            [0.0, 0.0, 10.0],
        )
    ),
    scaled_positions=np.array(
        (
            [0.3333333333333333, 0.6666666666666666, 0.5],
            [0.6666666666666667, 0.33333333333333337, 0.5],
        )
    ),
)
sys2d = sys2d.repeat((3, 3, 1))

# 3D system
sys3d = ase.lattice.cubic.Diamond(
    size=(1, 1, 1), symbol="Si", pbc=True, latticeconstant=5.430710
)

# Surface that has been split by the cell boundary
split = bcc100("Fe", size=(5, 1, 3), vacuum=8)
split.translate([0, 0, 9])
split.set_pbc(True)
split.wrap(pbc=True)


# Surface with a high amplitude wave
wavy = bcc100("Fe", size=(15, 3, 3), vacuum=8)
pos = wavy.get_positions()
x_len = np.linalg.norm(wavy.get_cell()[0, :])
x = pos[:, 0]
z = pos[:, 2]
z_new = z + 3 * np.sin(4 * (x / x_len) * np.pi)
pos_new = np.array(pos)
pos_new[:, 2] = z_new
wavy.set_positions(pos_new)
wavy.set_pbc(True)

# Graphite
graphite = ase.lattice.hexagonal.Graphite(
    size=(1, 1, 1), symbol="C", pbc=True, latticeconstant=(2.461, 6.708)
)

# Molecule with quite little vacuum between cells
small_vacuum = Atoms(
    symbols=["C", "O"],
    positions=[
        [1.13250529, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ],
    cell=[
        [5.29177211, 0.0, 0.0],
        [0.0, 5.29177211, 0.0],
        [0.0, 0.0, 5.29177211],
    ],
    pbc=True,
)
mx2 = ase.build.mx2(
    formula="MoS2", kind="2H", a=3.18, thickness=3.19, size=(5, 5, 1), vacuum=8
)
mx2 = mx2[[0, 12]]
tolerance = 0.2

com1 = bcc100("Fe", size=(3, 3, 4), vacuum=8)
adsorbate = ase.Atom(position=[4, 4, 4], symbol="H")
com1 += adsorbate
com1.set_pbc(True)
com1.translate([0, 0, 10])
com1.wrap()

com2 = Atoms(
    symbols=[
        "N",
        "N",
        "N",
        "N",
        "C",
        "C",
        "N",
        "C",
        "H",
        "C",
        "C",
        "N",
        "C",
        "H",
    ],
    positions=[
        [5.99520154, 4.36260352, 9.34466641],
        [3.93237808, 3.20844954, 9.59369566],
        [-0.44330457, 4.00755999, 9.58355994],
        [1.61974674, 2.84737344, 9.36529391],
        [2.67745295, 3.5184896, 9.09252221],
        [4.71467838, 4.12389331, 9.05494284],
        [2.65906057, 4.65710632, 8.18163931],
        [3.88950688, 5.03564733, 8.17378237],
        [4.26274113, 5.88360129, 7.59608619],
        [0.33924086, 3.07840333, 9.06923781],
        [-0.48556248, 2.14395639, 8.21180484],
        [7.03506045, 2.52272554, 8.20901382],
        [7.05303326, 3.68462936, 9.08992327],
        [-0.11207002, 1.28102047, 7.65690464],
    ],
    cell=[
        [8.751006887253668, 0.0, 0.0],
        [-4.372606804779377, 11.962358903815558, 0.0],
        [0.00037467328230266297, -0.000720566930414705, 16.891557236170406],
    ],
    pbc=True,
)

min1 = molecule("H2O")
min1.set_cell([3, 3, 3])


@pytest.mark.parametrize(
    "positions, cell, pbc, cutoff, expected_disp, expected_dist, expected_factors",
    [
        pytest.param(
            np.array(
                [
                    [0, 0, 0],
                    [0, 4, 0],
                ]
            ),
            np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]]),
            False,
            10,
            np.array([[[0, 0, 0], [0, -4, 0]], [[0, 4, 0], [0, 0, 0]]]),
            np.array([[0, 4], [4, 0]]),
            np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]),
            id="finite",
        ),
        pytest.param(
            np.array(
                [
                    [0, 0, 0],
                    [0, 4, 0],
                ]
            ),
            np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]]),
            [True, False, False],
            10,
            np.array([[[0, 0, 0], [0, -4, 0]], [[0, 4, 0], [0, 0, 0]]]),
            np.array([[0, 4], [4, 0]]),
            np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]),
            id="periodic-a",
        ),
        pytest.param(
            np.array(
                [
                    [0, 0, 0],
                    [0, 4, 0],
                ]
            ),
            np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]]),
            [False, True, False],
            10,
            np.array([[[0, 0, 0], [0, 1, 0]], [[0, -1, 0], [0, 0, 0]]]),
            np.array([[0, 1], [1, 0]]),
            np.array([[[0, 0, 0], [0, -1, 0]], [[0, 1, 0], [0, 0, 0]]]),
            id="periodic-b",
        ),
        pytest.param(
            np.array(
                [
                    [0, 0, 0],
                    [0, 4, 0],
                ]
            ),
            np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]]),
            [False, False, True],
            10,
            np.array([[[0, 0, 0], [0, -4, 0]], [[0, 4, 0], [0, 0, 0]]]),
            np.array([[0, 4], [4, 0]]),
            np.array([[[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]),
            id="periodic-c",
        ),
        pytest.param(
            np.array(
                [
                    [0, 0, 0],
                    [0, 4, 0],
                ]
            ),
            np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]]),
            True,
            10,
            np.array([[[0, 0, 0], [0, 1, 0]], [[0, -1, 0], [0, 0, 0]]]),
            np.array([[0, 1], [1, 0]]),
            np.array([[[0, 0, 0], [0, -1, 0]], [[0, 1, 0], [0, 0, 0]]]),
            id="periodic-abc",
        ),
        # pytest.param(
        #     np.array([
        #         [0, 0, 0],
        #         [0, 4, 0],
        #     ]),
        #     np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]]),
        #     True,
        #     0.1,
        #     np.array([
        #         [[0, 0, 0], [np.inf, np.inf, np.inf]],
        #         [[np.inf, np.inf, np.inf], [0, 0, 0]]
        #     ]),
        #     np.array([
        #         [0, np.inf],
        #         [np.inf, 0]
        #     ]),
        #     np.array([[[np.inf, np.inf, np.inf], [0, 0, 0]], [[np.inf, np.inf, np.inf], [0, 0, 0]]]),
        #     id='small-cutoff'
        # ),
        pytest.param(
            np.array(
                [
                    [-1, 0, 0],
                    [6, 0, 0],
                ]
            ),
            np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]]),
            True,
            10,
            np.array([[[0, 0, 0], [-2, 0, 0]], [[2, 0, 0], [0, 0, 0]]]),
            np.array([[0, 2], [2, 0]]),
            np.array([[[0, 0, 0], [-1, 0, 0]], [[1, 0, 0], [0, 0, 0]]]),
            id="outside-cell",
        ),
        pytest.param(
            np.array(
                [
                    [-1, 0, 0],
                    [6, 0, 0],
                ]
            ),
            np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]]),
            True,
            None,
            np.array([[[0, 0, 0], [-2, 0, 0]], [[2, 0, 0], [0, 0, 0]]]),
            np.array([[0, 2], [2, 0]]),
            np.array([[[0, 0, 0], [-1, 0, 0]], [[1, 0, 0], [0, 0, 0]]]),
            id="no-cutoff",
        ),
    ],
)
def test_displacement_tensor(
    positions, cell, pbc, cutoff, expected_disp, expected_dist, expected_factors
):
    (
        disp_tensor_ext,
        factors_ext,
        dist_mat_ext,
    ) = matid.geometry.get_displacement_tensor(
        positions,
        cell,
        pbc,
        cutoff=cutoff,
        return_distances=True,
        return_factors=True,
    )
    assert np.allclose(disp_tensor_ext, expected_disp)
    assert np.allclose(dist_mat_ext, expected_dist)
    assert np.allclose(factors_ext, expected_factors)


@pytest.mark.parametrize(
    "system, cell, pbc, expected_dimensionality",
    [
        pytest.param(sys0d, None, False, 0, id="0D-finite"),
        pytest.param(sys0d, [10, 10, 10], True, 0, id="0D-periodic-abc"),
        pytest.param(sys1d, None, True, 1, id="1D-periodic-abc"),
        pytest.param(sys1d, None, [False, True, True], 1, id="1D-periodic-bc"),
        pytest.param(sys1d, None, [False, False, True], 1, id="1D-periodic-c"),
        pytest.param(sys2d, None, True, 2, id="2D-periodic-abc"),
        pytest.param(sys2d, None, [True, True, False], 2, id="2D-periodic-ab"),
        pytest.param(sys3d, None, True, 3, id="3D-periodic-abc"),
        pytest.param(
            ase.io.read(
                Path(__file__).parent
                / "data/ROJiORHNwL4q0WTvNUy0mW5s2Buuq+PSX9X4dQR2r1cjQ9kBtuC-wI6MO8B.xyz"
            ),
            None,
            None,
            3,
            id="non-orthogonal-cell",
        ),
        pytest.param(split, None, None, 2, id="split-by-cell-boundary"),
        pytest.param(wavy, None, None, 2, id="wavy-surface"),
        pytest.param(wavy, None, None, 2, id="wavy-surface"),
        pytest.param(graphite, None, None, 3, id="graphite"),
        pytest.param(small_vacuum, None, None, 0, id="small-vacuum"),
    ],
)
def test_dimensionality(system, cell, pbc, expected_dimensionality):
    # Test with two radii setting alternatives
    for radii, cluster_threshold in [
        ("covalent", 2.7),
        ("vdw_covalent", 0.1),
    ]:
        if pbc is not None:
            system.set_pbc(pbc)
        if cell is not None:
            system.set_cell(cell)
        dimensionality = matid.geometry.get_dimensionality(
            system,
            radii=radii,
            cluster_threshold=cluster_threshold,
        )
        assert dimensionality == expected_dimensionality


@pytest.mark.parametrize(
    "cell, pbc, n_atoms",
    [
        pytest.param([0, 1, 1], [False, True, True], 9, id="a empty"),
        pytest.param([1, 0, 1], [True, False, True], 9, id="b empty"),
        pytest.param([1, 1, 0], [True, True, False], 9, id="c empty"),
        pytest.param([1, 0, 0], [True, False, False], 3, id="b, c empty"),
        pytest.param([0, 1, 0], [False, True, False], 3, id="a, c empty"),
        pytest.param([0, 0, 1], [False, False, True], 3, id="a, b empty"),
        pytest.param([0, 0, 0], [False, False, False], 1, id="a, b, c empty"),
    ],
)
def test_cell_completion(cell, pbc, n_atoms):
    """Tests that cell completion during the calculation of an extended system
    is performed correctly."""
    system = Atoms(
        positions=[[0.5, 0.5, 0.5]],
        symbols=["H"],
        cell=cell,
        pbc=pbc,
    )
    ext_system = matid.geometry.get_extended_system(system, cutoff=0.1)
    assert n_atoms == len(ext_system.indices)


@pytest.mark.parametrize(
    "system, pbc, cutoff, expected_indices, expected_factors",
    [
        pytest.param(
            mx2, False, 0, [0, 1], [[0, 0, 0], [0, 0, 0]], id="finite, zero cutoff"
        ),
        pytest.param(
            mx2, False, 1, [0, 1], [[0, 0, 0], [0, 0, 0]], id="finite, nonzero cutoff"
        ),
        pytest.param(
            mx2, True, 0, [0, 1], [[0, 0, 0], [0, 0, 0]], id="periodic, zero cutoff"
        ),
        pytest.param(
            mx2,
            True,
            1,
            np.tile([0, 1], 27),
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, -1.0],
                [0.0, 1.0, -1.0],
                [0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, -1.0, 1.0],
                [0.0, -1.0, 1.0],
                [0.0, -1.0, -1.0],
                [0.0, -1.0, -1.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, -1.0],
                [1.0, 0.0, -1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0],
                [1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [-1.0, 0.0, 1.0],
                [-1.0, 0.0, 1.0],
                [-1.0, 0.0, -1.0],
                [-1.0, 0.0, -1.0],
                [-1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [-1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 0.0],
                [-1.0, -1.0, 0.0],
                [-1.0, -1.0, 1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0],
            ],
            id="periodic, nonzero cutoff",
        ),
    ],
)
def test_extend_system(system, pbc, cutoff, expected_indices, expected_factors):
    """Test that the correct factor is returned when finding matches that
    are in the neighbouring cells.
    """
    system.set_pbc(pbc)
    extended_system = matid.geometry.get_extended_system(system, cutoff)
    assert np.array_equal(extended_system.indices, expected_indices)
    assert np.array_equal(extended_system.factors, expected_factors)


@pytest.mark.parametrize(
    "system, cutoff, position, expected_indices, expected_distances",
    [
        pytest.param(
            mx2, 0.1, [0, 0, 9.595], [0], [0], id="inside cutoff, zero distance"
        ),
        pytest.param(
            mx2, 0.1, [0, 0, 9.590], [0], [0.005], id="inside cutoff, nonzero distance"
        ),
        pytest.param(mx2, 0.1, [0, 0, 9.595 + 0.11], [], [], id="outside cutoff"),
    ],
)
def test_cell_list_position(
    system, cutoff, position, expected_indices, expected_distances
):
    """Test that the correct factor is returned when finding matches that
    are in the neighbouring cells.
    """
    cell_list = matid.ext.CellList(
        system.get_positions(),
        range(len(system)),
        np.tile([0, 0, 0], (len(system), 1)),
        cutoff,
    )
    result = cell_list.get_neighbours_for_position(
        position[0], position[1], position[2]
    )
    assert np.array_equal(result.indices, expected_indices)
    assert np.allclose(result.distances, expected_distances, rtol=0, atol=1e-8)


@pytest.mark.parametrize(
    "system, pbc, position, expected_matches, expected_factors",
    [
        pytest.param(
            mx2, False, [0, 0, 9.595], [0], (0, 0, 0), id="finite, within tolerance"
        ),
        pytest.param(
            mx2,
            False,
            [0, 0, 9.595 + 1.1 * tolerance],
            [None],
            (0, 0, 0),
            id="finite, out of tolerance",
        ),
        pytest.param(
            mx2,
            True,
            [0, 0, 9.595],
            [0],
            (0, 0, 0),
            id="periodic, within tolerance, same cell",
        ),
        pytest.param(
            mx2,
            True,
            np.array([0, 0, 9.595]) + mx2.get_cell().sum(axis=0) * 2,
            [None],  # TODO: The new implementation should probably return a match here?
            (2, 2, 2),
            id="periodic, position way outside original cell",
        ),
        pytest.param(
            mx2,
            True,
            (mx2.get_positions()[0] + mx2.get_cell()[0]),
            [0],
            (1, 0, 0),
            id="match in +a",
        ),
        pytest.param(
            mx2,
            True,
            (mx2.get_positions()[0] - mx2.get_cell()[0]),
            [0],
            (-1, 0, 0),
            id="match in -a",
        ),
        pytest.param(
            mx2,
            True,
            (mx2.get_positions()[0] + mx2.get_cell()[1]),
            [0],
            (0, 1, 0),
            id="match in +b",
        ),
        pytest.param(
            mx2,
            True,
            (mx2.get_positions()[0] - mx2.get_cell()[1]),
            [0],
            (0, -1, 0),
            id="match in -b",
        ),
        pytest.param(
            mx2,
            True,
            (mx2.get_positions()[0] + mx2.get_cell()[2]),
            [0],
            (0, 0, 1),
            id="match in +c",
        ),
        pytest.param(
            mx2,
            True,
            (mx2.get_positions()[0] - mx2.get_cell()[2]),
            [0],
            (0, 0, -1),
            id="match in -c",
        ),
    ],
)
def test_matches(system, pbc, position, expected_matches, expected_factors):
    """Test that the correct factor is returned when finding matches that
    are in the neighbouring cells.
    """
    system.set_pbc(pbc)

    # New CPP implementation
    cell_list = matid.geometry.get_cell_list(
        system.get_positions(),
        system.get_cell(),
        system.get_pbc(),
        tolerance,
        tolerance,
    )
    matches_ext, _, _, factors_ext = matid.geometry.get_matches(
        system,
        cell_list,
        np.array(position)[None, :],
        numbers=[system.get_atomic_numbers()[0]],
        tolerance=tolerance,
    )

    # Make sure that the atom is found in the correct copy
    assert tuple(factors_ext[0]) == expected_factors

    # Make sure that the correct atom is found
    assert np.array_equal(matches_ext, expected_matches)


@pytest.mark.parametrize(
    "cell, rel_pos, expected_pos, wrap, pbc",
    [
        pytest.param(
            np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]]),
            np.array(
                [
                    [0, 0, 0],
                    [1, 1, 1],
                    [0.5, 0.5, 0.5],
                ]
            ),
            np.array(
                [
                    [0, 0, 0],
                    [2, 3, 1],
                    [1, 1.5, 0.5],
                ]
            ),
            False,
            False,
            id="inside, unwrapped",
        ),
        pytest.param(
            np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]]),
            np.array(
                [
                    [0, 0, 0],
                    [2, 2, 2],
                    [0.5, 1.5, 0.5],
                ]
            ),
            np.array(
                [
                    [0, 0, 0],
                    [4, 6, 2],
                    [1, 3.5, 0.5],
                ]
            ),
            False,
            False,
            id="outside, unwrapped",
        ),
        pytest.param(
            np.array([[1, 1, 0], [0, 2, 0], [1, 0, 1]]),
            np.array(
                [
                    [0, 0, 0],
                    [2, 2, 2],
                    [0.5, 1.5, 0.5],
                ]
            ),
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 1.5, 0.5],
                ]
            ),
            True,
            True,
            id="outside, wrapped",
        ),
    ],
)
def test_to_cartesian(cell, rel_pos, expected_pos, wrap, pbc):
    cart_pos = matid.geometry.to_cartesian(cell, rel_pos, wrap, pbc)
    assert np.allclose(cart_pos, expected_pos)


@pytest.mark.parametrize(
    "system, pbc, expected_com",
    [
        pytest.param(com1, True, [4.0, 4.0, 20.15], id="periodic"),
        pytest.param(com1, False, [3.58770672, 3.58770672, 10.00200455], id="finite"),
        pytest.param(
            com2,
            True,
            [6.2609094, 3.59987973, 8.90948045],
            id="non-orthorhombic cell, negative lattive vector components",
        ),
    ],
)
def test_center_of_mass(system, pbc, expected_com):
    """Tests that the center of mass correctly takes periodicity into account."""
    system.set_pbc(pbc)
    com = matid.geometry.get_center_of_mass(system)
    assert np.allclose(com, expected_com, atol=0.1)


@pytest.mark.parametrize(
    "system, axis, minimum_size, expected_cell, expected_pos",
    [
        pytest.param(
            min1,
            0,
            0.1,
            np.array([[0.1, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]),
            np.array(
                [
                    [0.5, 0.0, 0.039754],
                    [0.5, 0.254413, -0.15901567],
                    [0.5, -0.254413, -0.15901567],
                ]
            ),
            id="a basis",
        ),
        pytest.param(
            min1,
            1,
            0.1,
            np.array([[3.0, 0.0, 0.0], [0.0, 1.526478, 0.0], [0.0, 0.0, 3.0]]),
            np.array(
                [
                    [0.0, 0.5, 0.039754],
                    [0.0, 1.0, -0.15901567],
                    [0.0, 0.0, -0.15901567],
                ]
            ),
            id="b basis",
        ),
        pytest.param(
            min1,
            2,
            0.1,
            np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.596309]]),
            np.array([[0.0, 0.0, 1.0], [0.0, 0.254413, 0.0], [0.0, -0.254413, 0.0]]),
            id="c basis",
        ),
        pytest.param(
            min1,
            2,
            2,
            np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2]]),
            np.array(
                [
                    [0.0, 0.0, 0.64907725],
                    [0.0, 0.254413, 0.35092275],
                    [0.0, -0.254413, 0.35092275],
                ]
            ),
            id="smaller than minimum size",
        ),
    ],
)
def test_minimize_cell(system, axis, minimum_size, expected_cell, expected_pos):
    min_system = matid.geometry.get_minimized_cell(system, axis, minimum_size)
    min_cell = min_system.get_cell()
    min_pos = min_system.get_scaled_positions()
    assert np.allclose(min_cell, expected_cell, atol=0.001, rtol=0)
    assert np.allclose(min_pos, expected_pos, atol=0.001, rtol=0)


@pytest.mark.parametrize(
    "system, axis, expected_thickness",
    [
        pytest.param(min1, 0, 0, id="a basis"),
        pytest.param(min1, 1, 1.526478, id="b basis"),
        pytest.param(min1, 2, 0.596309, id="c basis"),
    ],
)
def test_thickness(system, axis, expected_thickness):
    thickness = matid.geometry.get_thickness(system, axis)
    assert thickness == expected_thickness


@pytest.mark.parametrize("threshold", [0.3, 0.65, 1.5])
@pytest.mark.parametrize("pbc", [True, False])
def test_radii_distance_edges(threshold, pbc):
    """The sparse connectivity used for cleaning clusters should match the
    dense distance submatrix route exactly, including the cluster ordering."""
    rng = np.random.default_rng(7)
    n_atoms = 300
    a = 15.0
    system = Atoms(
        symbols=["Cu"] * n_atoms,
        scaled_positions=rng.random((n_atoms, 3)),
        cell=[a, a, a],
        pbc=pbc,
    )
    radii = matid.geometry.get_radii("covalent", system.get_atomic_numbers())
    distances = matid.geometry.get_distances(system, radii, cutoff=6.0)
    for k in [1, 50, n_atoms]:
        indices = rng.permutation(n_atoms)[:k]
        submatrix = distances.get_radii_distance_submatrix(indices)
        rows, cols = distances.get_radii_distance_edges(indices, threshold)
        expected = np.nonzero(np.triu(submatrix <= threshold))
        assert np.array_equal(rows, expected[0])
        assert np.array_equal(cols, expected[1])
        assert matid.geometry.get_clusters_from_edges(
            k, rows, cols
        ) == matid.geometry.get_clusters(submatrix, threshold)
