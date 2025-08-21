"""
A collection of classes for mapping indices to variable names and vice versa.
"""

import sys
import numpy as np

from openmdao.core.constants import INT_DTYPE


# default size of array for which we use a FlatTwoWayRangeMapper instead of a TwoWayRangeTree
MAX_FLAT_RANGE_SIZE = 1000


def key_size2range_iter(key_size_iter):
    """
    Convert an iterable of (key, size) tuples to an iterable of (key, start, stop) tuples.

    Parameters
    ----------
    key_size_iter : iterable of (key, size) tuples
        Iterable of (key, size) tuples.  key must be hashable.

    Yields
    ------
    key, (start, stop)
        key, (start, stop) tuples, where start and stop define the range of indices for the key.
    """
    start = end = 0
    for key, size in key_size_iter:
        end += size
        yield key, (start, end)
        start = end


class RangeMapper(dict):
    """
    A mapper of variable names to ranges.

    Parameters
    ----------
    key_size_iter : iterator over (key, size) tuples
        Iterator over (key, size) tuples.
    key_ind_iter : iterator over (key, ind) tuples, optional
        Iterator over (key, ind) tuples.

    Attributes
    ----------
    total_size : int
        The total size of the mapper.
    inds : dict
        A dictionary of indices for each key that has indices.
    """

    def __init__(self, key_size_iter, key_ind_iter=None):
        """
        Initialize a RangeMapper.
        """
        super().__init__()
        for key, tup in key_size2range_iter(key_size_iter):
            self[key] = tup
        try:
            self.total_size = tup[1]
        except Exception:
            self.total_size = 0

        # these inds, if they exist, are used when mapping matching indices from one mapper to
        # another, for example, when mapping from a solution vector into a row or column of a
        # jacobian.
        self.inds = {}
        if key_ind_iter is not None:
            for key, ind in key_ind_iter:
                self.inds[key] = ind

    def local_index(self, key, full_index):
        """
        Get the local index into the key's range.

        Parameters
        ----------
        key : object
            Key corresponding to an index range.
        full_index : int
            The index into the full array.

        Returns
        -------
        int or None
            The local index into the key's range.  None if the full index is not in the key's range.
        """
        start, stop = self[key]
        if full_index < start or full_index >= stop:
            return None
        return full_index - start

    def global_index(self, key, local_index):
        """
        Get the index into the full array corresponding to the given local index.

        Parameters
        ----------
        key : object
            Key corresponding to an index range.
        local_index : int
            The local index into the key's range.

        Returns
        -------
        int
            The index into the full array corresponding to the given local index.
        """
        return self[key][0] + local_index

    def offset(self, key):
        """
        Get the offset corresponding to the given key.

        Parameters
        ----------
        key : object (must be hashable)
            Data corresponding to an index range.

        Returns
        -------
        int
            The offset corresponding to the given key.
        """
        return self[key][0]

    def sizeof(self, key):
        """
        Get the size corresponding to the given key.

        Parameters
        ----------
        key : object (must be hashable)
            Key corresponding to an index range.

        Returns
        -------
        int
            The size corresponding to the given key.
        """
        start, stop = self[key]
        return stop - start


class TwoWayRangeMapper(RangeMapper):
    """
    A mapper of indices to variable names and vice versa.

    Parameters
    ----------
    key_size_iter : iterator over (key, size) tuples
        Iterator over (key, size) tuples.
    key_ind_iter : iterator over (key, ind) tuples, optional
        Iterator over (key, ind) tuples.
    """

    @staticmethod
    def create(key_size_iter, max_flat_range_size=MAX_FLAT_RANGE_SIZE):
        """
        Return a TwoWayRangeMapper that maps indices to variable names and vice versa.

        Parameters
        ----------
        key_size_iter : iterator over (key, size) tuples
            Iterator over (key, size) tuples.
        max_flat_range_size : int
            If the total array size is less than this, a FlatTwoWayRangeMapper will be returned
            instead of a TwoWayRangeTree.  Default is 1000.

        Returns
        -------
        FlatTwoWayRangeMapper or TwoWayRangeTree
            A TwoWayRangeMapper that maps indices to variable key and relative indices.
        """
        ranges = list(key_size_iter)
        total_size = sum(size for _, size in ranges)
        if total_size <= max_flat_range_size:
            return FlatTwoWayRangeMapper(ranges)
        else:
            return TwoWayRangeTree(ranges)

    def get_key(self, idx):
        """
        Find the key corresponding to the given index.

        Parameters
        ----------
        idx : int
            The index into the full array.
        """
        raise NotImplementedError("get_key method must be implemented by subclass.")

    def get_key_rel(self, idx):
        """
        Find the key and relative index corresponding to the matched range.

        Parameters
        ----------
        idx : int
            The index into the full array.

        Returns
        -------
        object or None
            The key corresponding to the matched range, or None if not found.
        int or None
            The relative index into the matched range, or None if not found.
        """
        key = self.get_key(idx)
        if key is None:
            return (None, None)

        return (key, idx - self.offset(key))

    def inds2keys(self, inds):
        """
        Find the set of keys corresponding to the given indices.

        Parameters
        ----------
        inds : iter of int
            The array indices.

        Returns
        -------
        set of object
            The set of keys corresponding to the given indices.
        """
        return {self.get_key(idx) for idx in inds}

    def between_iter(self, start_key, stop_key):
        """
        Iterate over (key, start, stop) tuples between the given start and stop keys.

        Parameters
        ----------
        start_key : object
            Key corresponding to an index range.
        stop_key : object
            Key corresponding to an index range.

        Yields
        ------
        (obj, int, int)
            (key, relative start index, relative stop index), where key is a hashable object.
        """
        started = False
        for key, (start, stop) in self.items():
            if key == start_key:
                yield (key, 0, stop - start)
                if start_key == stop_key:
                    break
                started = True
            elif started:
                if key == stop_key:
                    yield (key, 0, stop - start)
                    break
                else:
                    yield (key, 0, stop - start)

    def overlap_iter(self, key, other):
        """
        Find the set of keys that overlap between this mapper and another.

        Parameters
        ----------
        key : object
            Key corresponding to an index range.
        other : TwoWayRangeMapper
            Another mapper.

        Yields
        ------
        (obj, int, int, obj, int, int)
            (key, start, stop, otherkey, otherstart, otherstop).
        """
        start, stop = self[key]

        start_key, start_rel = other.get_key_rel(start)
        if start_key is None:
            return

        stop_key, stop_rel = other.get_key_rel(stop - 1)

        overlaps = [list(tup) for tup in other.between_iter(start_key, stop_key)]
        overlaps[0][1] = start_rel
        overlaps[-1][2] = stop_rel + 1

        start = stop = 0
        for k, kstart, kstop in overlaps:
            stop += kstop - kstart
            yield (key, start, stop, k, kstart, kstop)
            start = stop

    def dump(self, stream=sys.stdout):
        """
        Dump the contents of the mapper to stdout.

        Parameters
        ----------
        stream : file-like object, optional
            The stream to dump the contents to. Default is sys.stdout.
        """
        for key, (start, stop) in self.items():
            print(f'{key}: {start} - {stop}', file=stream)


class RangeTreeNode(object):
    """
    A node in a binary search tree of sizes, mapping key to an index range.

    Parameters
    ----------
    key : object
        Data corresponding to an index range.
    start : int
        Starting index of the variable.
    stop : int
        Ending index of the variable.

    Attributes
    ----------
    key : object
        Data corresponding to an index range.
    start : int
        Starting index of the variable.
    stop : int
        Ending index of the variable.
    left : RangeTreeNode or None
        Left child node.
    right : RangeTreeNode or None
        Right child node.
    """

    __slots__ = ['key', 'start', 'stop', 'left', 'right']

    def __init__(self, key, start, stop):
        """
        Initialize a RangeTreeNode.
        """
        self.key = key
        self.start = start
        self.stop = stop
        self.left = None
        self.right = None

    def __repr__(self):
        """
        Return a string representation of the RangeTreeNode.
        """
        return f"RangeTreeNode({self.key}, ({self.start}:{self.stop}))"


class TwoWayRangeTree(TwoWayRangeMapper):
    """
    A binary search tree of sizes, mapping key to an index range.

    Allows for fast lookup of the key corresponding to a given index. The sizes must be
    contiguous, but they can be of different sizes.

    Search complexity is O(log2 n). Uses less memory than FlatTwoWayRangeMapper when total array
    size is large.

    Parameters
    ----------
    key_size_iter : iterator over (key, size) tuples
        Iterator over (key, size) tuples.

    Attributes
    ----------
    root : RangeTreeNode
        Root node of the binary search tree.
    """

    def __init__(self, key_size_iter):
        """
        Initialize a TwoWayRangeTree.
        """
        super().__init__(key_size_iter)
        self.root = self.build([(key, start, stop) for key, (start, stop) in self.items()])

    def get_key(self, idx):
        """
        Find the key corresponding to the given index.

        Parameters
        ----------
        idx : int
            The index into the full array.

        Returns
        -------
        object or None
            The key corresponding to the given index, or None if not found.
        int or None
            The rank corresponding to the given index, or None if not found.
        """
        node = self.root
        while node is not None:
            if idx < node.start:
                node = node.left
            elif idx >= node.stop:
                node = node.right
            else:
                return node.key

    def build(self, ranges):
        """
        Build a binary search tree to map indices to variable key.

        Parameters
        ----------
        ranges : list of (key, start, stop)
            List of (key, start, stop) tuples, where start and stop
            define the range of indices for the key. Ranges must be ordered and contiguous.
            key must be hashable.

        Returns
        -------
        RangeTreeNode
            Root node of the binary search tree.
        """
        mid = len(ranges) // 2

        key, start, stop = ranges[mid]

        node = RangeTreeNode(key, start, stop)

        left_slices = ranges[:mid]
        right_slices = ranges[mid + 1:]

        if left_slices:
            node.left = self.build(left_slices)

        if right_slices:
            node.right = self.build(right_slices)

        return node


class FlatTwoWayRangeMapper(TwoWayRangeMapper):
    """
    A flat list mapping indices to variable key and relative indices.

    Parameters
    ----------
    key_size_iter : iterator over (key, size) tuples
        Iterator over (key, size) tuples.

    Attributes
    ----------
    ranges : list of (key, start, stop)
        List of (key, start, stop) tuples, where start and stop define the range of
        indices for that key. Ranges must be contiguous. key must be hashable.
    """

    def __init__(self, key_size_iter):
        """
        Initialize a FlatTwoWayRangeMapper.
        """
        super().__init__(key_size_iter)
        self.ranges = [None] * self.total_size
        for key, (start, stop) in self.items():
            self.ranges[start:stop] = [(key, start, stop)] * (stop - start)

    def get_key(self, idx):
        """
        Find the key corresponding to the given index.

        Parameters
        ----------
        idx : int
            The index into the full array.

        Returns
        -------
        object or None
            The key corresponding to the given index, or None if not found.
        """
        try:
            return self.ranges[idx][0]
        except IndexError:
            return None


def get_scatter_arrays(src_mapper, dest_mapper):
    """
    Get the scatter arrays to copy data from the src_mapper to the dest_mapper.

    Not all keys in the dest_mapper need to be in the src_mapper. In cases where a jacobian
    has both inputs and outputs as column variables, two sets of scatter arrays will need to
    be computed, one for the inputs and one for the outputs.

    Parameters
    ----------
    src_mapper : TwoWayRangeMapper
        The mapper to copy data from.
    dest_mapper : TwoWayRangeMapper
        The mapper to copy data to.

    Returns
    -------
    (ndarray, ndarray)
        The scatter arrays to copy data from the src_mapper to the dest_mapper.
    """
    src_inds = []
    dest_inds = []

    for key, (start, stop) in dest_mapper.items():
        if key in src_mapper:
            src_start, src_stop = src_mapper[key]
            if key in dest_mapper.inds:
                inds = dest_mapper.inds[key]
                # Handle the case where inds might be a list or array
                if isinstance(inds, (list, np.ndarray)):
                    src_inds.append(np.array(inds, dtype=INT_DTYPE) + src_start)
                else:
                    src_inds.append(np.array([inds], dtype=INT_DTYPE) + src_start)
                dest_inds.append(np.arange(start, stop, dtype=INT_DTYPE))
            else:
                src_inds.append(range(src_start, src_stop))
                dest_inds.append(range(start, stop))

    if src_inds:
        src_array = np.concatenate(src_inds)
    else:
        src_array = np.zeros(0, dtype=INT_DTYPE)

    if dest_inds:
        dest_array = np.concatenate(dest_inds)
    else:
        dest_array = np.zeros(0, dtype=INT_DTYPE)

    return src_array, dest_array
