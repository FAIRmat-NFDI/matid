import numpy as np

from ase import Atoms

from matid.data import constants

import networkx as nx


class LinkedUnitCollection(dict):
    """Represents a collection of similar cells that are connected in 3D space
    to form a structure, e.g. a surface.

    Essentially this is a special flavor of a regular dictionary: the keys can
    only be a sequence of three integers, and the values should be LinkedUnits.
    """

    def __init__(
        self,
        system,
        cell,
        is_2d,
    ):
        """
        Args:
            system(ase.Atoms): A reference to the system from which this
                LinkedUniCollection is gathered.
            cell(ase.Atoms): The prototype cell that is used in finding this
                region.
            is_2d(boolean): Whether this system represents a 2D-material or not.
        """
        self.system = system
        self.cell = cell
        self.is_2d = is_2d
        self._search_graph_cache = nx.MultiDiGraph()
        self._search_graph_log = []
        self._search_graph_n_replayed = 0
        self._index_cell_map = {}
        self._used_points = set()
        self._basis_indices = None
        self._pos_tol = None
        dict.__init__(self)

    def add_search_edge(self, cell_index, seed_index, target_cell, match, multiplier):
        """Records a link between two cells that was found during the search.

        Creating the networkx graph is comparatively expensive and it is not
        needed by all users. The links are therefore only recorded here and
        the graph is created on demand when _search_graph is accessed.
        """
        self._search_graph_log.append(
            (cell_index, seed_index, target_cell, match, multiplier)
        )

    @property
    def _search_graph(self):
        """A networkx.MultiDiGraph of the links between the cells that were
        found during the search."""
        G = self._search_graph_cache
        log = self._search_graph_log
        for i in range(self._search_graph_n_replayed, len(log)):
            cell_index, seed_index, target_cell, match, multiplier = log[i]
            G.add_node(cell_index, index=seed_index)
            G.add_node(target_cell, index=match)
            G.add_edge(cell_index, target_cell, multiplier=multiplier)
        self._search_graph_n_replayed = len(log)
        return G

    def __setitem__(self, key, value):
        # Transform key to tuple, check length
        try:
            key = tuple(key)
        except Exception:
            raise TypeError(
                "Could not transform the given key '{}' into tuple.".format(key)
            )
        if len(key) != 3:
            raise ValueError(
                "The given coordinate '{}' does not have three components.".format(key)
            )

        # Check that old unit is not overwritten
        if key in dict.keys(self):
            raise ValueError("Overriding existing units is not supported.")

        dict.__setitem__(self, key, value)

    def recreate_valid(self):
        """Used to recreate a new Atoms object, where each atom is created from
        a single unit cell. Atoms that were found not to belong to the periodic
        unit cell are not included.
        """
        recreated_system = Atoms(
            cell=self.system.get_cell(),
            pbc=self.system.get_pbc(),
        )
        for unit in self.values():
            i_valid_indices = np.array([x for x in unit.basis_indices if x is not None])
            if len(i_valid_indices) != 0:
                i_atoms = self.system[i_valid_indices]
                recreated_system += i_atoms

        return recreated_system

    def get_basis_indices(self):
        """Returns the indices of the atoms that were found to belong to a unit
        cell basis in the LinkedUnits in this collection as a single list.

        Returns:
            np.ndarray: Indices of the atoms in the original system that belong
            to this collection of LinkedUnits.
        """
        if self._basis_indices is None:
            indices = set()
            for unit in self.values():
                for index in unit.basis_indices:
                    if index is not None:
                        indices.add(index)
            self._basis_indices = indices

        return self._basis_indices

    def get_connected_directions(self):
        """During the tracking of the region the information about searches
        that matched an atom twice but with a negated multiplier are stored.
        """
        G = self._search_graph
        dir_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        directions = set([0, 1, 2])
        for node in G.nodes():
            node_edges = G.in_edges(node, data=True)
            dir_to_remove = set()
            for direction in directions:
                dir_vector = dir_vectors[direction]
                positive = False
                negative = False
                for edge in node_edges:
                    multiplier = edge[2]["multiplier"]
                    if np.array_equal(multiplier, dir_vector):
                        positive = True
                    if np.array_equal(multiplier, -dir_vector):
                        negative = True
                    if positive and negative:
                        break
                if positive and negative:
                    dir_to_remove.add(direction)
            directions -= dir_to_remove

        connected_directions = np.array([True, True, True])
        connected_directions[list(directions)] = False
        return connected_directions


class LinkedUnit:
    """Represents a cell that is connected to others in 3D space to form a
    structure, e.g. a surface.
    """

    def __init__(
        self,
        index,
        seed_index,
        seed_coordinate,
        cell,
        basis_indices,
        substitutions,
        vacancies,
    ):
        """
        Args:
            index(tuple of three ints):
            seed_index(int):
            seed_coordinate():
            cell(np.ndarray): Cell for this unit. Can change from unit to unit.
            basis_indices(sequence of ints and Nones): A sequence where there
                is an index or None for each atom that is supposed to be in the
                basis
            substitute_indices(sequence of ints and Nones): If basis atom is
                replaced by a foreign atom, the index of the substitutional atom is
                here.
        """
        self.index = index
        self.seed_index = seed_index
        self.seed_coordinate = seed_coordinate
        self.cell = cell
        self.basis_indices = basis_indices
        self.substitutions = substitutions
        self.vacancies = vacancies


class Substitution:
    """Represents a substitutional point defect."""

    def __init__(self, index, position, original_element, substitutional_element):
        """
        Args:
            index (int): Index of the subtitutional defect in the original
                system.
            original (ase.Atom): The original atom that was substituted
            substitution (ase.Atom): The substitutional atom

        """
        self.index = index
        self.original_element = original_element
        self.substitutional_element = substitutional_element
