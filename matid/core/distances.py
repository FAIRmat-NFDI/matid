import numpy as np


class Distances:
    """Container for all distance information that has been extracted from a
    system.

    Supports two backing representations:

    - Dense: the full ``[n, n]`` / ``[n, n, 3]`` matrices. Produced when
      :func:`matid.geometry.get_distances` is called with an infinite cutoff.
    - Sparse: a minimum-image neighbour list (COO/CSR) holding only pairs within
      a finite cutoff. Produced when a finite cutoff is given. This avoids
      allocating the dense ``O(n^2)`` matrices.

    Consumers that only need local distances should use the ``get_*`` accessor
    methods, which work transparently for both representations. The dense matrix
    properties (``disp_tensor_mic`` etc.) are only available in dense mode; in
    sparse mode they raise, since materializing them would defeat the purpose.
    """

    def __init__(
        self,
        disp_tensor_mic,
        disp_factors,
        dist_matrix_mic,
        dist_matrix_radii_mic,
        cell_list=None,
    ):
        # Dense mode (backwards compatible positional constructor).
        self._dense = True
        self._disp_tensor_mic = disp_tensor_mic
        self._disp_factors = disp_factors
        self._dist_matrix_mic = dist_matrix_mic
        self._dist_matrix_radii_mic = dist_matrix_radii_mic
        self.cell_list = cell_list

    @classmethod
    def from_sparse(cls, n, row, col, distance, displacement, factor, radii):
        """Construct a sparse-backed Distances from a COO neighbour list.

        The minimum-image displacement of an entry ``(row, col)`` is
        ``pos_row - pos_col``, matching the dense ``disp_tensor_mic`` convention.
        Both pair directions are expected to be present in the input.

        Args:
            n: Number of atoms.
            row, col: Integer COO indices, shape (nnz,).
            distance: Pairwise distances, shape (nnz,).
            displacement: Displacement vectors, shape (nnz, 3).
            factor: Periodic image factors, shape (nnz, 3).
            radii: Per-atom radii, shape (n,).
        """
        self = cls.__new__(cls)
        self._dense = False
        self._n = n
        self._radii = np.asarray(radii)
        self.cell_list = None

        # Group the COO entries by row to form a CSR-style structure, so that
        # all neighbours of an atom are a contiguous slice.
        order = np.lexsort((col, row))
        self._col = np.asarray(col)[order]
        self._dist = np.asarray(distance)[order]
        self._disp = np.asarray(displacement)[order]
        self._fac = np.asarray(factor)[order]
        sorted_row = np.asarray(row)[order]
        self._row_ptr = np.searchsorted(sorted_row, np.arange(n + 1))
        return self

    @classmethod
    def from_csr(cls, n, row_ptr, col, distance, displacement, factor, radii):
        """Construct a sparse-backed Distances from a CSR neighbour list, where
        the neighbours of atom ``i`` are stored in the slice
        ``row_ptr[i]:row_ptr[i + 1]``. The arrays are used as is, without
        copying.

        Args:
            n: Number of atoms.
            row_ptr: Row offsets, shape (n + 1,).
            col, distance, displacement, factor: Neighbour data in the same
                format as for :meth:`from_sparse`, grouped by row.
            radii: Per-atom radii, shape (n,).
        """
        self = cls.__new__(cls)
        self._dense = False
        self._n = n
        self._radii = np.asarray(radii)
        self.cell_list = None
        self._row_ptr = np.asarray(row_ptr)
        self._col = np.asarray(col)
        self._dist = np.asarray(distance)
        self._disp = np.asarray(displacement)
        self._fac = np.asarray(factor)
        return self

    # ------------------------------------------------------------------
    # Dense matrix properties (only valid in dense / infinite-cutoff mode).
    # ------------------------------------------------------------------
    def _require_dense(self, name):
        if not self._dense:
            raise AttributeError(
                "'{}' is a dense O(n^2) matrix and is not available for a "
                "sparse (finite-cutoff) Distances. Use the get_* accessor "
                "methods instead.".format(name)
            )

    @property
    def disp_tensor_mic(self):
        self._require_dense("disp_tensor_mic")
        return self._disp_tensor_mic

    @property
    def disp_factors(self):
        self._require_dense("disp_factors")
        return self._disp_factors

    @property
    def dist_matrix_mic(self):
        self._require_dense("dist_matrix_mic")
        return self._dist_matrix_mic

    @property
    def dist_matrix_radii_mic(self):
        self._require_dense("dist_matrix_radii_mic")
        return self._dist_matrix_radii_mic

    # ------------------------------------------------------------------
    # Accessors that work in both dense and sparse mode.
    # ------------------------------------------------------------------
    def _row_slice(self, i):
        return slice(self._row_ptr[i], self._row_ptr[i + 1])

    def get_displacement_column(self, i):
        """Returns ``disp_tensor_mic[:, i]`` of shape (n, 3), i.e. the vectors
        ``pos_a - pos_i`` for every atom ``a``. Entries beyond the cutoff are
        infinite; the ``i``-th entry is zero."""
        if self._dense:
            return self._disp_tensor_mic[:, i]
        out = np.full((self._n, 3), np.inf)
        s = self._row_slice(i)
        # Stored row entries are pos_i - pos_a; the column needs pos_a - pos_i.
        out[self._col[s]] = -self._disp[s]
        out[i] = 0.0
        return out

    def get_factor_column(self, i):
        """Returns ``disp_factors[i, :]`` of shape (n, 3). Entries beyond the
        cutoff are infinite; the ``i``-th entry is zero."""
        if self._dense:
            return self._disp_factors[i]
        out = np.full((self._n, 3), np.inf)
        s = self._row_slice(i)
        out[self._col[s]] = self._fac[s]
        out[i] = 0.0
        return out

    def get_radii_distance_row(self, i):
        """Returns ``dist_matrix_radii_mic[i, :]`` of shape (n,). Entries beyond
        the cutoff are infinite; the ``i``-th entry is ``-2 * radii[i]``."""
        if self._dense:
            return self._dist_matrix_radii_mic[i]
        out = np.full(self._n, np.inf)
        s = self._row_slice(i)
        cols = self._col[s]
        out[cols] = self._dist[s] - (self._radii[i] + self._radii[cols])
        out[i] = -2.0 * self._radii[i]
        return out

    def get_radii_distance_submatrix(self, indices):
        """Returns ``dist_matrix_radii_mic[np.ix_(indices, indices)]``. Entries
        for pairs beyond the cutoff are infinite."""
        indices = np.asarray(indices)
        if self._dense:
            return self._dist_matrix_radii_mic[np.ix_(indices, indices)]
        k = len(indices)
        out = np.full((k, k), np.inf)
        # Map global atom index -> local position within ``indices`` (-1 when the
        # atom is not part of this submatrix). This lets each row filter its
        # neighbours with vectorized numpy indexing instead of a per-entry dict
        # lookup.
        g2l = np.full(self._n, -1, dtype=np.intp)
        g2l[indices] = np.arange(k)
        for li, gi in enumerate(indices):
            s = self._row_slice(gi)
            cols = self._col[s]
            lj = g2l[cols]
            sel = lj >= 0
            out[li, lj[sel]] = self._dist[s][sel] - (
                self._radii[gi] + self._radii[cols[sel]]
            )
        di = np.arange(k)
        out[di, di] = -2.0 * self._radii[indices]
        return out

    def get_radii_distance_edges(self, indices, threshold):
        """Returns the local index pairs ``(li, lj)`` with ``li <= lj`` for which
        ``dist_matrix_radii_mic[np.ix_(indices, indices)] <= threshold``,
        including the diagonal. The pairs are returned in row-major order.

        This is a sparse alternative to :meth:`get_radii_distance_submatrix`
        for finding connectivity within a large group of atoms: the memory
        footprint scales with the number of connected pairs instead of the
        square of the number of atoms.

        Returns:
            Two integer arrays ``(li, lj)``.
        """
        indices = np.asarray(indices, dtype=np.intp)
        k = len(indices)
        if self._dense:
            submatrix = self._dist_matrix_radii_mic[np.ix_(indices, indices)]
            li, lj = np.nonzero(np.triu(submatrix <= threshold))
            return li, lj

        # Gather the neighbour list entries of all requested rows at once.
        g2l = np.full(self._n, -1, dtype=np.intp)
        g2l[indices] = np.arange(k)
        starts = self._row_ptr[indices]
        lengths = self._row_ptr[indices + 1] - starts
        li = np.repeat(np.arange(k), lengths)
        offsets = np.cumsum(lengths) - lengths
        entries = np.arange(lengths.sum()) + np.repeat(starts - offsets, lengths)
        cols = self._col[entries]
        lj = g2l[cols]
        sel = lj > li
        li, lj, gi, cols = li[sel], lj[sel], indices[li[sel]], cols[sel]
        dist = self._dist[entries[sel]] - (self._radii[gi] + self._radii[cols])
        sel = dist <= threshold
        li, lj = li[sel], lj[sel]

        # Add the diagonal and order the pairs row-major.
        diag = np.arange(k)
        diag = diag[-2.0 * self._radii[indices] <= threshold]
        li = np.concatenate((li, diag))
        lj = np.concatenate((lj, diag))
        order = np.lexsort((lj, li))
        return li[order], lj[order]
