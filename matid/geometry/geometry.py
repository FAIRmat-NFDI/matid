"""This module defines functions for deriving geometry related quantities from
a atomic system.
"""

import math

import numpy as np
from numpy.random import RandomState

from ase.data import covalent_radii
from ase.data.vdw_alvarez import vdw_radii
from ase import Atom, Atoms
import ase.geometry

import spglib
import networkx as nx

from matid.data.element_data import get_covalent_radii
from matid.core.linkedunits import Substitution
from matid.core.distances import Distances
from matid.data.constants import CLUSTER_THRESHOLD
import matid.geometry
import matid.ext


def get_dimensionality(
    system,
    cluster_threshold=CLUSTER_THRESHOLD,
    dist_matrix_radii_mic_1x=None,
    return_clusters=False,
    radii="covalent",
):
    """Used to calculate the dimensionality of a system with a modified
    Topology Scaling Algorithm (TSA) (Michael Ashton, Joshua Paul, Susan B.
    Sinnott, and Richard G. Hennig Phys. Rev. Lett. 118, 106101).

    Args:
        system (ASE.Atoms): The system for which the dimensionality is
            evaluated.
        cluster_threshold(float): The epsilon value for the DBSCAN algorithm
            that is used to identify clusters within the unit cell.
        dist_matrix_radii_pbc (np.ndarray): A precalculated distance matrix
            that takes in to account the periodicity and has the covalent radii of
            the atoms already subtracted.
        return_clusters(boolean): Whether the clusters are returned
        radii(str|np.ndarray): The radii to use for atoms. Use either a preset
            or a custom list or atomic radii where the atomic number is used as an
            index. The available presets are:

                - covalent: Covalent radii from DOI:10.1039/B801115J
                - vdw: van Der Waals radii from DOI:10.1039/C3DT50599E
                - vdw_covalent: preferably van Der Waals radii, covalent if vdw
                  not defined.

    Returns:
        int|none: The dimensionality of the system. If the dimensionality can't be
            evaluated because the system has multiple disconnected components in
            the original cell, None is returned.
        list: A list of clusters. Each entry in the list contains the indices
            of atoms in a cluster.
    """
    system_1x = system
    pbc = system_1x.get_pbc()
    num_1x = system_1x.get_atomic_numbers()
    cell_1x = system_1x.get_cell()

    # When calculating the displacement tensor we can ignore neighbouring
    # copies that are farther away than the clustering cutoff. The maximum mic
    # distance to allow is cluster_threshold + 2*max_radii
    radii_1x = get_radii(radii, num_1x)
    max_radii = radii_1x.max()
    cutoff = cluster_threshold + 2 * max_radii

    # 1x1x1 system
    if dist_matrix_radii_mic_1x is None:
        pos_1x = system.get_positions()
        _, dist_matrix_mic_1x = get_displacement_tensor(
            pos_1x,
            cell_1x,
            pbc,
            cutoff=cutoff,
            return_distances=True,
        )
        radii_matrix_1x = radii_1x[:, None] + radii_1x[None, :]
        dist_matrix_radii_mic_1x = dist_matrix_mic_1x - radii_matrix_1x

    # Check the number of clusters.
    clusters_1x = get_clusters(
        dist_matrix_radii_mic_1x, cluster_threshold, min_samples=1
    )
    n_clusters_1x = len(clusters_1x)

    # If the system consists of multiple components that are not connected
    # according to the clustering done here, then we cannot assess the
    # dimensionality.
    if n_clusters_1x > 1:
        dim = None
    else:
        # 2x2x2 system
        n_pbc = np.sum(pbc)
        if n_pbc > 0:
            repeats = np.array([1, 1, 1])
            repeats[pbc] = 2
            system_2x = system.repeat(repeats)
            pos_2x = system_2x.get_positions()
            cell_2x = system_2x.get_cell()
            _, dist_matrix_mic_2x = get_displacement_tensor(
                pos_2x,
                cell_2x,
                pbc,
                cutoff=cutoff,
                return_distances=True,
            )
            radii_2x = np.tile(radii_1x, repeats.prod())
            radii_matrix_2x = radii_2x[:, None] + radii_2x[None, :]
            dist_matrix_radii_mic_2x = dist_matrix_mic_2x - radii_matrix_2x

            # Check the number of clusters.
            clusters_2x = get_clusters(
                dist_matrix_radii_mic_2x, cluster_threshold, min_samples=1
            )
            n_clusters_2x = len(clusters_2x)

            # This is the analytic formula for the dimensionality based on cluster
            # size on a 2x supersystem.
            dim = int(n_pbc - math.log(n_clusters_2x, 2))
        else:
            dim = 0

    if return_clusters:
        return dim, clusters_1x
    else:
        return dim


def get_moments_of_inertia(system, weight=True):
    """Calculates geometric inertia tensor, i.e., inertia tensor but with
    all masses are set to 1.

    I_ij = sum_k m_k (delta_ij * r_k^2 - x_ki * x_kj)
    with r_k^2 = x_k1^2 + x_k2^2 x_k3^2

    Args:
        system(ASE Atoms): Atomic system.

    Returns:
        (np.ndarray, np.ndarray): The eigenvalues and eigenvectors of the
        geometric inertia tensor.
    """
    # Move the origin to the geometric center
    positions = system.get_positions()
    centroid = get_center_of_mass(system, weight)
    pos_shifted = positions - centroid

    # Calculate the geometric inertia tensor
    if weight:
        weights = system.get_masses()
    else:
        weights = np.ones((len(system)))
    x = pos_shifted[:, 0]
    y = pos_shifted[:, 1]
    z = pos_shifted[:, 2]
    I11 = np.sum(weights * (y**2 + z**2))
    I22 = np.sum(weights * (x**2 + z**2))
    I33 = np.sum(weights * (x**2 + y**2))
    I12 = np.sum(-weights * x * y)
    I13 = np.sum(-weights * x * z)
    I23 = np.sum(-weights * y * z)

    inertia_tensor = np.array([[I11, I12, I13], [I12, I22, I23], [I13, I23, I33]])

    evals, evecs = np.linalg.eigh(inertia_tensor)

    return evals, evecs


def get_center_of_mass(system):
    """Calculates the center of mass and also takes the periodicity of the
    system into account.

    The algorithm is replicated from:
    https://en.wikipedia.org/wiki/Center_of_mass#Systems_with_periodic_boundary_conditions

    Args:
        system(ase.Atoms): The system for which the center of mass is
            calculated.

    Returns:
        np.ndarray: The cartesian positions of the center of mass in the given
        system.
    """
    pbc = system.get_pbc()
    relative_positions = system.get_scaled_positions()
    masses = system.get_masses()
    total_mass = np.sum(masses)
    cell = system.get_cell()

    rel_com = np.zeros((3))
    for i_comp in range(3):
        i_pbc = pbc[i_comp]
        i_pos = relative_positions[:, i_comp]
        if i_pbc:
            theta = i_pos * 2 * np.pi
            xi = np.cos(theta) * masses
            zeta = np.sin(theta) * masses

            xi_mean = np.mean(xi)
            zeta_mean = np.mean(zeta)

            mean_theta = np.arctan2(-zeta_mean, -xi_mean) + np.pi
            com_rel = mean_theta / (2 * np.pi)
            rel_com[i_comp] = com_rel
        else:
            rel_com[i_comp] = np.sum(i_pos * masses) / total_mass

    com_cart = to_cartesian(cell, rel_com)[0, :]

    return com_cart


def get_space_filling(system):
    """Calculates the ratio of vacuum to filled space by assuming covalent
    radii for the atoms.

    Args:
        system(ASE.Atoms): Atomic system.

    Returns:
        float: The ratio of occupied volume to the cell volume.
    """
    cell_volume = system.get_volume()
    atomic_numbers = system.get_atomic_numbers()
    occupied_volume = 0
    radii = get_covalent_radii(atomic_numbers)
    volumes = 4.0 / 3.0 * np.pi * radii**3
    occupied_volume = np.sum(volumes)
    ratio = occupied_volume / cell_volume

    return ratio


def make_random_displacement(system, delta, rng=None):
    """Dislocate every atom in the given system in a random direction but by
    the same amount. Uses an internal random number generator to avoid touching
    the global numpy.random.seed()-function.

    Args:
        system(ASE.Atoms): The system for which the displacement are performed.
        delta(float): The magnitude of the displacements.
        rng(np.random.RandomState): Random number generator.
    """
    if rng is None:
        rng = RandomState()
    pos = system.get_positions()
    n_atoms = len(system)
    disloc = rng.rand(n_atoms, 3)
    disloc /= np.linalg.norm(disloc, axis=1)[:, None]
    disloc *= delta
    new_pos = pos + disloc
    system.set_positions(new_pos)


def get_extended_system(system, cutoff=0):
    """Given a system and a cutoff value, returns a new system which has been
    extended so that for each atom the neighbourhood within the cutoff radius is
    present, taking periodic boundary conditions into account.

    Args:
        system(ASE.Atoms): System to extend
        cutoff(float): Radial cutoff

    Returns:
        ExtendedSystem object.
    """
    extended_system = matid.ext.extend_system(
        system.get_positions(),
        system.get_atomic_numbers(),
        system.get_cell(),
        system.get_pbc(),
        cutoff,
    )

    return extended_system


def get_clusters(dist_matrix, threshold, min_samples=1):
    """Used to detect clusters with the DBSCAN algorithm.

    Args:
        dist_matrix(np.ndarray): The 2D distance matrix from which the clusters
            are calculated.
        threshold(float): The epsilon threshold value for the clustering
        min_samples(int): The minimum allowed cluster size.

    Returns:
        list: A list of clusters, where each cluster is a list of indices for
        the elements belonging to the cluster.
    """
    # As the distances have been normalized with respect to the covalent radiis,
    # the distance matrix may in some cases have negative values. This simply
    # means that the distance between two atoms is smaller than the sum of
    # their covalent radiis. The distance values are here clipped to
    # zero to avoid problems in the cluster detection. Another option would be
    # to normalize the distances so that unity would correspond to the sum of
    # the two radii. In addition, any values larger than the threshold are
    # capped: this is required in order to handle infinite values which come
    # from distances larger than a set radial cutoff.
    np.clip(dist_matrix, a_min=0, a_max=1.1 * threshold, out=dist_matrix)

    # The clusters are the connected components of the graph that links any two
    # atoms whose (radii-subtracted) distance is within the threshold. This is
    # identical to DBSCAN with min_samples=1 and a precomputed metric, but avoids
    # the heavy sklearn import (which otherwise dominates the runtime on small
    # systems) and is faster per call. networkx is already a dependency used
    # elsewhere in matid.
    adjacency = dist_matrix <= threshold
    graph = nx.from_numpy_array(adjacency)
    return [list(component) for component in nx.connected_components(graph)]


def get_clusters_from_edges(n, rows, cols):
    """Used to detect clusters, i.e. connected components, from a sparse list
    of connections.

    Gives identical results to :func:`get_clusters` when given the pairs
    ``(i, j)``, ``i <= j``, for which ``dist_matrix[i, j] <= threshold`` in
    row-major order, but without requiring the dense ``n x n`` matrix.

    Args:
        n(int): Number of nodes.
        rows(np.ndarray): First node index of each connection.
        cols(np.ndarray): Second node index of each connection.

    Returns:
        list: A list of clusters, where each cluster is a list of indices for
        the elements belonging to the cluster.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from(zip(np.asarray(rows).tolist(), np.asarray(cols).tolist()))
    return [list(component) for component in nx.connected_components(graph)]


def get_covalent_distances(system, mic=True):
    """Returns a distance matrix where the covalent radii have been taken into
    account. Clips negative values to be zero.
    """
    if system is not None:
        if len(system) == 1:
            return np.array([[0]])

        # Calculate distance matrix with radii taken into account
        distance_matrix = system.get_all_distances(mic=mic)

    # Remove the radii from distances and clip out negative values
    numbers = system.get_atomic_numbers()
    radii = covalent_radii[numbers]
    radii_matrix = radii[None, :] + radii[:, None]
    distance_matrix -= radii_matrix
    np.clip(distance_matrix, 0, None, out=distance_matrix)

    return distance_matrix


def get_biggest_gap_indices(coordinates):
    """Given the list of coordinates for one axis, this function will find the
    maximum gap between them and return the index of the bottom and top
    coordinates. The bottom and top are defined as:

    ===       ===============    --->
        ^top    ^bot               ^axis direction
    """
    # Find the maximum vacuum gap for all basis vectors
    sorted_indices = np.argsort(coordinates)
    sorted_comp = coordinates[sorted_indices]
    rolled_comp = np.roll(sorted_comp, 1, axis=0)
    distances = sorted_comp - rolled_comp

    # The first distance is from first to last, so it needs to be
    # wrapped around
    distances[0] += 1

    # Find maximum gap
    bottom_index = sorted_indices[np.argmax(distances)]
    top_index = sorted_indices[np.argmax(distances) - 1]

    return bottom_index, top_index


def get_dimensions(system, vacuum_gaps):
    """Given a system with vacuum gaps, calculate its dimensions in the
    directions with vacuum gaps by also taking into account the atomic radii.
    """
    orig_cell_lengths = np.linalg.norm(system.get_cell(), axis=1)

    # Create a repeated copy of the system. The repetition is needed in order
    # to get gaps to neighbouring cell copies in the periodic dimensions with a
    # vacuum gap
    sys = system.copy()
    sys = sys.repeat([2, 2, 2])

    dimensions = [None, None, None]
    numbers = sys.get_atomic_numbers()
    positions = sys.get_scaled_positions()
    radii = covalent_radii[numbers]
    cell_lengths = np.linalg.norm(sys.get_cell(), axis=1)
    radii_in_cell_basis = radii[:, None] / cell_lengths[None, :]

    for i_dim, vacuum_gap in enumerate(vacuum_gaps):
        if vacuum_gap:
            # Make a data structure containing the atom location information as
            # intervals from one side of the atom to the other in each
            # dimension.
            intervals = Intervals()
            for i_pos, pos in enumerate(positions[:, i_dim]):
                i_radii = radii_in_cell_basis[i_pos, i_dim]
                i_axis_start = pos - i_radii
                i_axis_end = pos + i_radii
                intervals.add_interval(i_axis_start, i_axis_end)

            # Calculate the maximum distance between atoms, when taking radius
            # into account
            gap = intervals.get_max_distance_between_intervals()
            gap = gap * cell_lengths[i_dim]
            dimensions[i_dim] = orig_cell_lengths[i_dim] - gap

    return dimensions


def get_wrapped_positions(scaled_pos, precision=1e-5):
    """Wrap the given relative positions so that each element in the array
    is within the half-closed interval [0, 1)

    By wrapping values near 1 to 0 we will have a consistent way of
    presenting systems.
    """
    scaled_pos %= 1

    abs_zero = np.absolute(scaled_pos)
    abs_unity = np.absolute(abs_zero - 1)

    near_zero = np.where(abs_zero < precision)
    near_unity = np.where(abs_unity < precision)

    scaled_pos[near_unity] = 0
    scaled_pos[near_zero] = 0

    return scaled_pos


def get_displacement_tensor(
    positions,
    cell=None,
    pbc=False,
    cutoff=float("inf"),
    return_factors=False,
    return_distances=False,
):
    if cutoff is None:
        cutoff = float("inf")
    if cell is None:
        cell = np.eye(3)
    n_atoms = positions.shape[0]
    disp_tensor = np.full((n_atoms, n_atoms, 3), float("inf"))
    dist_mat = np.full((n_atoms, n_atoms), float("inf"))
    factors = np.full((n_atoms, n_atoms, 3), float("inf"))
    matid.ext.get_displacement_tensor(
        disp_tensor,
        dist_mat,
        factors,
        positions,
        cell,
        expand_pbc(pbc),
        cutoff,
        return_factors,
        return_distances,
    )

    result = [disp_tensor]
    if return_factors:
        result.append(factors)
    if return_distances:
        result.append(dist_mat)

    if len(result) == 1:
        return result[0]
    return tuple(result)


def expand_pbc(pbc):
    """Used to expand a pbc definition into an array of three booleans.

    Args:
        pbc(boolean or a list of booleans): The periodicity of the cell. This
            can be any of the values that is also supprted by ASE, namely: a
            boolean or a list of three booleans.

    Returns:
        np.ndarray of booleans: The periodicity expanded as an explicit list of
        three boolean values.
    """
    if pbc is True:
        new_pbc = [True, True, True]
    elif pbc is False:
        new_pbc = [False, False, False]
    elif len(pbc) == 3:
        new_pbc = pbc
    else:
        raise ValueError(
            "Could not interpret the given periodic boundary conditions: '{}'".format(
                pbc
            )
        )

    return np.array(new_pbc)


def change_basis(positions, basis, offset=None):
    """Transform the given cartesian coordinates to a basis that is defined by
    the given basis and origin offset.

    Args:
        positions(np.ndarray): Positions in cartesian coordinates.
        basis(np.ndarray): Basis to which to transform.
        offset(np.ndarray): Offset of the origins. A vector from the old basis
            origin to the new basis origin.
    Returns:
        np.ndarray: Relative positions in the new basis
    """

    if offset is not None:
        positions -= offset
    new_basis_inverse = np.linalg.inv(basis.T)
    pos_prime = np.dot(positions, new_basis_inverse.T)

    return pos_prime


def get_positions_within_basis(
    system, basis, origin, tolerance, mask=[True, True, True], pbc=True
):
    """Used to return the indices of positions that are inside a certain basis.
    Also takes periodic boundaries into account.

    Args:
        system(ASE.Atoms): System from which the positions are searched.
        basis(np.ndarray): New basis vectors.
        origin(np.ndarray): New origin of the basis in cartesian coordinates.
        tolerance(float): The matching tolerance in angstrom.
        mask(sequence of bool): Mask for selecting the basis's to consider.
        pbc(sequence of bool): The periodicity of the system.

    Returns:
        sequence of int: Indices of the atoms within this cell in the given
            system.
        np.ndarray: Relative positions of the found atoms.
        np.ndarray: The index of the periodic copy in which the position was
            found.
    """
    # If the search extend beyound the cell boundary and periodic boundaries
    # allow, we must divide the search area into multiple regions

    # Transform positions into the new basis
    cart_pos = system.get_positions()
    orig_cell = system.get_cell()
    pbc = expand_pbc(pbc)

    # We need to expand the system in different directions to get the

    # See if the new positions extend beyound the boundaries. The original
    # simulation cell is always convex, so we can just check the corners of
    # unit cell defined by the basis
    max_a = origin + basis[0, :]
    max_b = origin + basis[1, :]
    max_c = origin + basis[1, :]
    max_ab = origin + basis[0, :] + basis[1, :]
    max_ac = origin + basis[0, :] + basis[2, :]
    max_bc = origin + basis[1, :] + basis[2, :]
    max_abc = origin + basis[0, :] + basis[1, :] + basis[2, :]
    vectors = np.array((max_a, max_b, max_c, max_ab, max_ac, max_bc, max_abc))
    rel_vectors = to_scaled(orig_cell, vectors, wrap=False, pbc=system.get_pbc())
    factors = np.floor(rel_vectors).astype(int)
    min_factors = np.min(factors, axis=0)
    max_factors = np.max(factors, axis=0)
    a_range = range(min_factors[0], max_factors[0] + 1)
    b_range = range(min_factors[1], max_factors[1] + 1)
    c_range = range(min_factors[2], max_factors[2] + 1)
    factors = matid.geometry.cartesian((a_range, b_range, c_range))

    directions = []
    for factor in factors:
        a_per = factor[0]
        b_per = factor[1]
        c_per = factor[2]
        allow = True
        if a_per != 0 and not pbc[0]:
            allow = False
        if b_per != 0 and not pbc[1]:
            allow = False
        if c_per != 0 and not pbc[2]:
            allow = False
        if allow:
            directions.append(factor)

    # If the new cell is overflowing beyound the boundaries of the original
    # system, we have to also check the periodic copies.
    indices = []
    precs = tolerance / np.linalg.norm(basis, axis=1)
    orig_basis = system.get_cell()
    cell_pos = []
    factors = []
    for i_dir in directions:
        vec_new_cart = cart_pos + np.dot(i_dir, orig_basis)
        vec_new_rel = change_basis(vec_new_cart - origin, basis)

        # Find the atoms within the cell with a vectorized boundary check over
        # all positions at once, restricted to the masked axes.
        keep = np.ones(len(vec_new_rel), dtype=bool)
        for axis in range(3):
            if mask[axis]:
                col = vec_new_rel[:, axis]
                keep &= (col >= -precs[axis]) & (col <= 1 + precs[axis])

        # np.nonzero keeps ascending index order, matching the original
        # per-atom loop.
        found = np.nonzero(keep)[0]
        if found.size:
            indices.append(found)
            cell_pos.append(vec_new_rel[found])
            factors.append(np.tile(i_dir, (found.size, 1)))

    if indices:
        indices = np.concatenate(indices)
        cell_pos = np.concatenate(cell_pos)
        factors = np.concatenate(factors)
    else:
        indices = np.array([])
        cell_pos = np.array([])
        factors = np.array([])

    return indices, cell_pos, factors


def get_matches(
    system, cell_list, positions, numbers, tolerance, return_vacancies=True
):
    """Given a system and a list of cartesian positions and atomic numbers,
    returns a list of indices for the atoms corresponding to the given
    positions with some tolerance.

    Args:
        system(ASE.Atoms): System where to search the positions
        cell_list(CellList): The cell list for an appropriately extended version
            of the system.
        positions(np.ndarray): Positions to match in the system.
        tolerance(float): Maximum allowed distance for matching.
        return_vacancies(bool): Whether to build and return the list of vacancy
            ase.Atom objects. Constructing these is comparatively expensive, so
            callers that do not consume the vacancies can set this to False to
            skip the work.

    Returns:
        np.ndarray: indices of matched atoms
        list: list of substitutions
        list: list of vacancies
        np.ndarray: for each searched position, an integer array representing
            the number of the periodic copy where the match was found.
    """
    # The atomic numbers and positions are only read here, so the arrays are
    # accessed directly instead of through the copying getters: this function
    # is called once per searched unit cell and copying the full arrays every
    # time makes the region search scale quadratically with system size.
    atomic_numbers = system.numbers
    system_positions = system.positions
    matches = []
    substitutions = []
    copy_indices = np.zeros((len(positions), 3))
    vacancies = []
    cell = system.get_cell()

    # Scaled positions (floored) are only needed for the copy index of the
    # returned vacancies. Computing them once for all positions avoids a
    # per-vacancy linear solve, which dominates runtime for disordered systems.
    # When the caller does not consume the vacancies, this is skipped entirely.
    if return_vacancies:
        floored_factors = np.floor(to_scaled(cell, positions, wrap=False))

    # The already pre-computed cell-list is used in finding neighbours.
    for i, (position, atomic_number) in enumerate(zip(positions, numbers)):
        match = None
        substitution = None
        copy_index = None
        displacement = None
        cell_list_result = cell_list.get_neighbours_for_position(
            position[0], position[1], position[2]
        )
        indices = cell_list_result.indices_original
        if len(indices) > 0:
            distances = cell_list_result.distances
            factors = cell_list_result.factors
            min_distance_index = np.argmin(distances)
            closest_distance = distances[min_distance_index]
            closest_factor = factors[min_distance_index]
            closest_index = indices[min_distance_index]
            if closest_distance <= tolerance:
                closest_atomic_number = atomic_numbers[closest_index]
                copy_index = closest_factor
                if closest_atomic_number == atomic_number:
                    match = closest_index
                    substitution = None
                else:
                    substitution = Substitution(
                        closest_index,
                        system_positions[closest_index].copy(),
                        atomic_number,
                        closest_atomic_number,
                    )
        matches.append(match)
        substitutions.append(substitution)
        if match is None and substitution is None:
            if return_vacancies:
                vacancies.append(Atom(atomic_number, position=position))
                copy_index = floored_factors[i]
        # When vacancies are not requested, the vacancy copy index is left at
        # its initialized zero value (it is never consumed by such callers).
        if copy_index is not None:
            copy_indices[i] = copy_index

    return matches, substitutions, vacancies, copy_indices


def get_matches_simple(system, cell_list, positions, numbers, tolerance):
    """Given a system and a list of cartesian positions and atomic numbers,
    returns a list of indices for the atoms corresponding to the given
    positions with some tolerance.
    Args:
        system(ASE.Atoms): System where to search the positions
        cell_list(CellList): The cell list for an appropriately extended version
            of the system.
        positions(np.ndarray): Positions to match in the system.
        tolerance(float): Maximum allowed distance for matching.
    Returns:
        list: list of matched atoms or None is nothing was matched.
    """
    atomic_numbers = system.numbers
    cell = system.get_cell()
    pbc = system.get_pbc()
    matches = []
    displacements = []
    wrapped_positions = ase.geometry.wrap_positions(positions, cell, pbc)

    for wrapped_position, atomic_number in zip(wrapped_positions, numbers):
        match = None
        displacement = None
        cell_list_result = cell_list.get_neighbours_for_position(
            wrapped_position[0], wrapped_position[1], wrapped_position[2]
        )
        indices = cell_list_result.indices_original
        if len(indices) > 0:
            distances = cell_list_result.distances
            min_distance_index = np.argmin(distances)
            closest_distance = distances[min_distance_index]
            closest_index = indices[min_distance_index]
            if closest_distance <= tolerance:
                closest_atomic_number = atomic_numbers[closest_index]
                if closest_atomic_number == atomic_number:
                    match = closest_index
                    displacement = cell_list_result.displacements[min_distance_index]
        matches.append(match)
        displacements.append(displacement)

    return matches, displacements


def get_cell_list(positions, cell, pbc, extension, cutoff):
    """Given a system and a cutoff value, returns a cell list object.

    Args:
        positions(np.ndarray): Cartesian positions
        cell(np.ndarray): Cell as 3x3 array
        pbc(np.ndarray): Periodic boundary conditions as array of three booleans
        extension(float): How much the system should be extended for the search.
        cutoff(float): Radial cutoff

    Returns:
        CellList object.
    """
    return matid.ext.get_cell_list(positions, cell, pbc, extension, cutoff)


def to_scaled(cell, positions, wrap=False, pbc=False):
    """Used to transform a set of positions to the basis defined by the
    cell of this system.

    Args:
        system(ASE.Atoms): Reference system.
        positions (numpy.ndarray): The positions to scale
        wrap (numpy.ndarray): Whether the positions should be wrapped
            inside the cell.

    Returns:
        numpy.ndarray: The scaled positions
    """
    # Force 1D to 2D and check shape
    shape = positions.shape
    if len(shape) == 1:
        positions = positions[None, :]
        shape = positions.shape
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError(
            "The given positions are not compatible. Please provide positions "
            "as rows of a two-dimensional array."
        )
    pbc = expand_pbc(pbc)
    fractional = np.linalg.solve(cell.T, positions.T).T

    if wrap:
        for i, periodic in enumerate(pbc):
            if periodic:
                fractional[:, i] %= 1.0

    return fractional


def to_cartesian(cell, scaled_positions, wrap=False, pbc=False):
    """Used to transform a set of relative positions to the cartesian basis
    defined by the cell of this system.

    Args:
        cell (numpy.ndarray): 3x3 array with lattice vectors as rows.
        positions (numpy.ndarray): The positions to scale. These postiions
            should have a shape of [n, 3], where n is the number of positions.
        wrap (numpy.ndarray): Whether the positions should be wrapped
            inside the cell.

    Returns:
        numpy.ndarray: The cartesian positions
    """
    # Force 1D to 2D and check shape
    shape = scaled_positions.shape
    if len(shape) == 1:
        scaled_positions = scaled_positions[None, :]
        shape = scaled_positions.shape
    if len(shape) != 2 or shape[1] != 3:
        raise ValueError(
            "The given positions are not compatible. Please provide positions "
            "as rows of a two-dimensional array."
        )

    pbc = expand_pbc(pbc)
    if wrap:
        for i, periodic in enumerate(pbc):
            if periodic:
                scaled_positions[:, i] %= 1.0

    cartesian_positions = np.dot(scaled_positions, cell)
    return cartesian_positions


def translate(system, translation, relative=False):
    """Translates the positions by the given translation.

    Args:
        translation (1x3 numpy.array): The translation to apply.
        relative (bool): True if given translation is relative to cell
            vectors.
    """
    if relative:
        rel_pos = system.get_scaled_positions()
        rel_pos += translation
        system.set_scaled_positions(rel_pos)
    else:
        cart_pos = system.get_positions()
        cart_pos += translation
        system.set_positions(cart_pos)


def get_closest_direction(vec, directions, normalized=False):
    """Used to return the direction that is most parallel to a given one.

    Args:

    Returns:
    """
    if not normalized:
        directions = directions / np.linalg.norm(directions, axis=1)
    dots = np.abs(np.dot(vec, directions.T))
    index = np.argmax(dots)

    return index


class Intervals(object):
    """Handles list of intervals.

    This class allows sorting and adding up of intervals and taking into
    account if they overlap.
    """

    def __init__(self, intervals=None):
        """Args:
        intervals: List of intervals that are added.
        """
        self._intervals = []
        self._merged_intervals = []
        self._merged_intervals_need_update = True
        if intervals is not None:
            self.add_intervals(intervals)

    def _add_up(self, intervals):
        """Add up the length of intervals.

        Argument:
            intervals: List of intervals that are added up.

        Returns:
            Result of addition.
        """
        if len(intervals) < 1:
            return None
        result = 0.0
        for interval in intervals:
            result += abs(interval[1] - interval[0])
        return result

    def add_interval(self, a, b):
        """Add one interval.

        Args:
            a, b: Start and end of interval. The order does not matter.
        """
        self._intervals.append((min(a, b), max(a, b)))
        self._merged_intervals_need_update = True

    def add_intervals(self, intervals):
        """Add list of intervals.

        Args:
            intervals: List of intervals that are added.
        """
        for interval in intervals:
            if len(interval) == 2:
                self.add_interval(interval[0], interval[1])
            else:
                raise ValueError("Intervals must be tuples of length 2!")

    def set_intervals(self, intervals):
        """Set list of intervals.

        Args:
            intervals: List of intervals that are set.
        """
        self._intervals = []
        self.add_intervals(intervals)

    def remove_interval(self, i):
        """Remove one interval.

        Args:
            i: Index of interval that is removed.
        """
        try:
            del self._intervals[i]
            self._merged_intervals_need_update = True
        except IndexError:
            pass

    def get_intervals(self):
        """Returns the intervals."""
        return self._intervals

    def get_intervals_sorted_by_start(self):
        """Returns list with intervals ordered by their start."""
        return sorted(self._intervals, key=lambda x: x[0])

    def get_intervals_sorted_by_end(self):
        """Returns list with intervals ordered by their end."""
        return sorted(self._intervals, key=lambda x: x[1])

    def get_merged_intervals(self):
        """Returns list of merged intervals so that they do not overlap anymore."""
        if self._merged_intervals_need_update:
            if len(self._intervals) < 1:
                return self._intervals
            # sort intervals in list by their start
            sorted_by_start = self.get_intervals_sorted_by_start()
            # add first interval
            merged = [sorted_by_start[0]]
            # start from second interval
            for current in sorted_by_start[1:]:
                previous = merged[-1]
                # new interval if not current and previous are not overlapping
                if previous[1] < current[0]:
                    merged.append(current)
                # merge if current and previous are overlapping and if end of previous is expanded by end of current
                elif previous[1] < current[1]:
                    merged[-1] = (previous[0], current[1])
            self._merged_intervals = merged
            self._merged_intervals_need_update = False
        return self._merged_intervals

    def get_max_distance_between_intervals(self):
        """Returns the maximum distance between the intervals while accounting for overlap."""
        if len(self._intervals) < 2:
            return None
        merged_intervals = self.get_merged_intervals()
        distances = []
        if len(merged_intervals) == 1:
            return 0.0
        for i in range(len(merged_intervals) - 1):
            distances.append(abs(merged_intervals[i + 1][0] - merged_intervals[i][1]))
        return max(distances)

    def add_up_intervals(self):
        """Returns the added up lengths of intervals without accounting for overlap."""
        return self._add_up(self._intervals)

    def add_up_merged_intervals(self):
        """Returns the added up lengths of merged intervals in order to account for overlap."""
        return self._add_up(self.get_merged_intervals())


def get_thickness(system, axis):
    """Get the thickness of an atomic system along a basis vector direction.

    Args:
        system(ase.Atoms): The system from which the thickness is evaluated.
        axis(int): The index of the unit cell basis in which direction the
            thickness is evaluated.

    Returns:
        float: The thickness of the system as measured from the center of the
        topmost atom to the center of lowermost atom.
    """
    pos = system.get_scaled_positions()[:, axis]
    pos_min = pos.min()
    pos_max = pos.max()
    basis = system.get_cell()[axis, :]
    thickness = (pos_max - pos_min) * np.linalg.norm(basis)

    return thickness


def get_minimized_cell(system, axis, min_size):
    """Used to resize the cell in the given cell direction so that all atoms
    are within the cell.

    Args:
        system(ase.Atoms): The system to minimize. Must have a cell with
            non-zero volume.
        axis(int): Index of the axis to minimize.
        axis(min_size): The minimum size for the cell in the given direction.

    Returns:
        ase.Atoms: The new minimized system.
    """
    # Grow the cell to fit all atoms
    rel_pos = system.get_scaled_positions(wrap=False)
    num = system.get_atomic_numbers()
    pbc = system.get_pbc()
    basis = system.get_cell()
    c = basis[axis, :]
    c_length = np.linalg.norm(c)
    c_norm = c / c_length
    c_comp = rel_pos[:, axis]

    min_index = np.argmin(c_comp, axis=0)
    max_index = np.argmax(c_comp, axis=0)

    pos_min_rel = np.zeros((3))
    pos_min_rel[axis] = c_comp[min_index]
    pos_max_rel = np.zeros((3))
    pos_max_rel[axis] = c_comp[max_index]

    pos_min_cart = matid.geometry.to_cartesian(basis, pos_min_rel)
    pos_max_cart = matid.geometry.to_cartesian(basis, pos_max_rel)
    c_real_cart = pos_max_cart - pos_min_cart
    c_size = np.linalg.norm(c_real_cart)

    # We demand a minimum size for the c-vector even if the system seems to
    # be purely 2-dimensional. This is done because the 3D-space cannot be
    # searched properly if one dimension is flat.
    if c_size < min_size:
        c_inflated_cart = min_size * c_norm
        c_new_cart = c_inflated_cart
    else:
        c_new_cart = c_real_cart
    new_basis = np.array(basis)
    new_basis[axis, :] = c_new_cart

    new_scaled_pos = rel_pos - pos_min_rel
    new_cart_test = matid.geometry.to_cartesian(basis, new_scaled_pos)
    new_scaled_pos = matid.geometry.to_scaled(new_basis, new_cart_test)

    # Translate the system to be in the middle if size is smaller than the
    # given minimum size
    if c_size < min_size:
        offset_cart = (c_real_cart - c_inflated_cart) / 2
        offset_rel = matid.geometry.to_scaled(new_basis, offset_cart)
        new_scaled_pos -= offset_rel

    # Create translated system
    minimized_system = Atoms(
        cell=new_basis, scaled_positions=new_scaled_pos, symbols=num, pbc=pbc
    )

    return minimized_system


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Args:
        arrays(sequence of arrays): The arrays from which the product is
            created.
        out(ndarray): Array to place the cartesian product in.

    Returns:
        ndarray: 2-D array of shape (M, len(arrays)) containing cartesian
        products formed of input arrays.

    Example:
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m : (j + 1) * m, 1:] = out[0:m, 1:]
    return out


def get_crystallinity(symmetry_analyser):
    """Quantifies the crystallinity of the structure as a ratio of symmetries
    per number of unique atoms in primitive cell. This metric can be used to
    distinguish between amorphous and 'regular' crystals.

    The number of symmetry operations corresponds to the symmetry operations
    corresponding to the hall number of the structure. The symmetry operations
    as given by spglib.get_symmetry() are specific to the original structure,
    and they have not been reduced to the symmetries of the space group.

    Args:
        symmetry_analyser (): A SymmetryAnalyzer object that has been
            initialized with an atomic structure.

    Returns:
        (float) A ratio of symmetries per unique atoms in the primitive cell.
    """
    # Get the number of equivalent atoms in the primitive cell.
    n_unique_atoms_prim = len(symmetry_analyser.get_equivalent_atoms_primitive())

    hall_number = symmetry_analyser.get_hall_number()
    sym_ops = spglib.get_symmetry_from_database(hall_number)
    n_symmetries = len(sym_ops["rotations"])

    ratio = n_symmetries / float(n_unique_atoms_prim)

    return ratio


def get_distances(system: Atoms, radii="covalent", cutoff=float("inf")) -> Distances:
    """Returns complete distance information.

    Args:
        system: The system from which distances are calculated from.
        radii: The atomic radii to use, see get_radii.
        cutoff: Radial cutoff for the distance calculation. Pairs that are
            farther apart than this (using the minimum image convention) are
            reported as infinite distance. Defaults to infinity, which produces
            the full dense distance matrix. A finite cutoff can dramatically
            speed up the calculation when only local distances are needed.
    Returns:
        A Distances instance.
    """
    pos = system.get_positions()
    cell = system.get_cell()
    pbc = system.get_pbc()
    atomic_numbers = system.get_atomic_numbers()
    radii = get_radii(radii, atomic_numbers)
    n_atoms = len(atomic_numbers)

    # With a finite cutoff we only compute the local neighbours and store them
    # sparsely, avoiding the O(n^2) dense matrices entirely.
    if np.isfinite(cutoff):
        sparse = matid.ext.get_displacement_list(
            pos, np.asarray(cell), expand_pbc(pbc), float(cutoff)
        )
        return Distances.from_sparse(
            n_atoms,
            np.asarray(sparse.row),
            np.asarray(sparse.col),
            np.asarray(sparse.distance),
            np.asarray(sparse.displacement),
            np.asarray(sparse.factor),
            radii,
        )

    # Infinite cutoff: build the full dense distance matrices.
    if pbc.any():
        disp_tensor_mic, disp_factors, dist_matrix_mic = get_displacement_tensor(
            pos, cell, pbc, cutoff=cutoff, return_factors=True, return_distances=True
        )
    else:
        disp_tensor_mic, dist_matrix_mic = get_displacement_tensor(
            pos, cutoff=cutoff, return_distances=True
        )
        disp_factors = np.zeros(disp_tensor_mic.shape)

    # Calculate the distance matrix where the periodicity and the covalent
    # radii have been taken into account
    dist_matrix_radii_mic = np.array(dist_matrix_mic)
    radii_matrix = radii[:, None] + radii[None, :]
    dist_matrix_radii_mic -= radii_matrix

    return Distances(
        disp_tensor_mic,
        disp_factors,
        dist_matrix_mic,
        dist_matrix_radii_mic,
    )


def get_radii(radii, atomic_numbers=None) -> np.ndarray:
    """Returns an array of atomic radii for each atom."""
    if isinstance(radii, str):
        if radii == "covalent":
            radii = covalent_radii
        elif radii == "vdw":
            radii = vdw_radii
        elif radii == "vdw_covalent":
            radii = np.array(
                [
                    vdw_radii[i] if not np.isnan(vdw_radii[i]) else covalent_radii[i]
                    for i in range(len(vdw_radii))
                ]
            )
        radii = radii[atomic_numbers]
    return radii


def swap_basis(atoms: Atoms, a: int, b: int):
    """Used to swap two bases in a system. The geometry remains identical.

    Args:
        a: First index to swap
        b: Second index to swap
    """
    cell_old = atoms.get_cell()
    pbc_old = atoms.get_pbc()
    cell_new = np.array(cell_old)
    cell_new[a] = cell_old[b]
    cell_new[b] = cell_old[a]
    pbc_new = np.array(pbc_old)
    pbc_new[a] = pbc_old[b]
    pbc_new[b] = pbc_old[a]
    atoms.set_cell(cell_new)
    atoms.set_pbc(pbc_new)


def complete_cell(a, b, length):
    """Given two basis vectors a and b, creates a third one that is
    orthogonal and has the length of 2 * maximum 2D cell height.

    Args:
        a(np.ndarray): First basis vector
        b(np.ndarray): Second basis vector
        length(float): Length of the basis

    Returns:
        np.ndarray: The third basis vector
    """
    c = np.cross(a, b)
    c_norm = c / np.linalg.norm(c)
    c_norm = c_norm[None, :]
    c = c_norm * length

    return c
