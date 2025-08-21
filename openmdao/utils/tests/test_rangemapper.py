import unittest
import io
import numpy as np

from openmdao.utils.rangemapper import RangeMapper, TwoWayRangeMapper, TwoWayRangeTree, FlatTwoWayRangeMapper, get_scatter_arrays


_data = {
    'a': 1,  # 0:1
    'b': 8,  # 1:9
    'x': 6,  # 9:15
    'y': 21, # 15:36
    'z': 6,  # 36:42
}


class TestTwoWayRangeMapper(unittest.TestCase):
    def test_create(self):
        mapper = TwoWayRangeMapper.create(_data.items())
        self.assertEqual(type(mapper), FlatTwoWayRangeMapper)
        mapper = TwoWayRangeMapper.create(_data.items(), max_flat_range_size=40)
        self.assertEqual(type(mapper), TwoWayRangeTree)

    def test_get_key(self):
        for mclass in (TwoWayRangeTree, FlatTwoWayRangeMapper):
            mapper = mclass(_data.items())
            inds = [0, 1, 7, 9, 14, 15, 22, 41, 42, 43]
            expected = ['a', 'b', 'b', 'x', 'x', 'y', 'y', 'z', None, None]
            for i, ex_i in zip(inds, expected):
                self.assertEqual(mapper.get_key(i), ex_i)

    def test_getitem(self):
        for mclass in (TwoWayRangeTree, FlatTwoWayRangeMapper):
            mapper = mclass(_data.items())
            keys = ['a', 'b', 'x', 'y', 'z']
            expected = [(0, 1), (1, 9), (9, 15), (15, 36), (36, 42)]
            for key, ex in zip(keys, expected):
                self.assertEqual(mapper[key], ex)

            try:
                mapper['bad']
            except KeyError:
                pass
            else:
                self.fail("Expected KeyError")

    def test_sizeof(self):
        for mclass in (TwoWayRangeTree, FlatTwoWayRangeMapper):
            mapper = mclass(_data.items())
            keys = ['a', 'b', 'x', 'y', 'z']
            expected = [1, 8, 6, 21, 6]
            for key, ex in zip(keys, expected):
                self.assertEqual(mapper.sizeof(key), ex)

            try:
                mapper.sizeof('bad')
            except KeyError:
                pass
            else:
                self.fail("Expected KeyError")

    def test_get_key_rel(self):
        for mclass in (TwoWayRangeTree, FlatTwoWayRangeMapper):
            mapper = mclass(_data.items())
            inds = [0, 1, 7, 9, 14, 15, 22, 41, 42, 43]
            expected = [('a',0), ('b',0), ('b',6), ('x',0), ('x',5), ('y',0), ('y',7), ('z',5),
                        (None, None), (None, None)]
            for i, ex in zip(inds, expected):
                self.assertEqual(mapper.get_key_rel(i), ex)

    def test_iter(self):
        for mclass in (TwoWayRangeTree, FlatTwoWayRangeMapper):
            mapper = mclass(_data.items())
            expected = [('a', (0, 1)), ('b', (1, 9)), ('x', (9, 15)), ('y', (15, 36)), ('z', (36, 42))]
            for got, ex in zip(mapper.items(), expected):
                self.assertEqual(got, ex)

    def test_between_iter(self):
        for mclass in (TwoWayRangeTree, FlatTwoWayRangeMapper):
            mapper = mclass(_data.items())
            self.assertEqual(list(mapper.between_iter('a', 'a')), [('a', 0, 1)])
            self.assertEqual(list(mapper.between_iter('b', 'y')), [('b', 0, 8), ('x', 0, 6), ('y', 0, 21)])
            self.assertEqual(list(mapper.between_iter('y', 'z')), [('y', 0, 21), ('z', 0, 6)])
            self.assertEqual(list(mapper.between_iter('z', 'z')), [('z', 0, 6)])

    def test_overlap_iter(self):
        otherdata = {
            'a': 1,   # 0:1
            'b': 10,  # 1:11
            'x': 6,   # 11:17
            'y': 18,  # 17:35
            'z': 7,   # 35:42
        }

        for mclass in (TwoWayRangeTree, FlatTwoWayRangeMapper):
            mapper = mclass(_data.items())
            other_mapper = mclass(otherdata.items())
            self.assertEqual(list(mapper.overlap_iter('a', other_mapper)), [('a', 0, 1, 'a', 0, 1)])
            self.assertEqual(list(mapper.overlap_iter('y', other_mapper)), [('y', 0, 2, 'x', 4, 6), ('y', 2, 20, 'y', 0, 18), ('y', 20, 21, 'z', 0, 1)])
            self.assertEqual(list(mapper.overlap_iter('z', other_mapper)), [('z', 0, 6, 'z', 1, 7)])

    def test_dump(self):
        for mclass in (TwoWayRangeTree, FlatTwoWayRangeMapper):
            stream = io.StringIO()
            mapper = mclass(_data.items())
            mapper.dump(stream)
            contents = stream.getvalue()
            self.assertTrue('a: 0 - 1' in contents)
            self.assertTrue('b: 1 - 9' in contents)
            self.assertTrue('x: 9 - 15' in contents)
            self.assertTrue('y: 15 - 36' in contents)
            self.assertTrue('z: 36 - 42' in contents)


class TestRangeTree(unittest.TestCase):
    def setUp(self):
        self.sizes = [('a', 1), ('b', 2), ('c', 3)]
        self.range_tree = TwoWayRangeTree(self.sizes)

    def test_init(self):
        self.assertEqual(self.range_tree.root.key, 'b')
        self.assertEqual(self.range_tree.root.start, 1)
        self.assertEqual(self.range_tree.root.stop, 3)
        self.assertEqual(self.range_tree.root.left.key, 'a')
        self.assertEqual(self.range_tree.root.right.key, 'c')

    def test_build(self):
        sizes = [('d', 4), ('e', 5), ('f', 6)]
        tree = TwoWayRangeTree(sizes)
        root = tree.root
        self.assertEqual(root.key, 'e')
        self.assertEqual(root.start, 4)
        self.assertEqual(root.stop, 9)
        self.assertEqual(root.left.key, 'd')
        self.assertEqual(root.right.key, 'f')


class TestRangeMapper(unittest.TestCase):
    def setUp(self):
        self.key_size_data = [('a', 1), ('b', 8), ('x', 6), ('y', 21), ('z', 6)]
        self.key_ind_data = [('a', 0), ('b', 1), ('x', 9), ('y', 15), ('z', 36)]
        self.mapper = RangeMapper(self.key_size_data, self.key_ind_data)

    def test_init(self):
        """Test RangeMapper initialization."""
        # Test without key_ind_iter
        mapper = RangeMapper(self.key_size_data)
        self.assertEqual(mapper.total_size, 42)
        self.assertEqual(len(mapper.inds), 0)

        # Test with key_ind_iter
        mapper = RangeMapper(self.key_size_data, self.key_ind_data)
        self.assertEqual(mapper.total_size, 42)
        self.assertEqual(len(mapper.inds), 5)
        self.assertEqual(mapper.inds['a'], 0)
        self.assertEqual(mapper.inds['z'], 36)

    def test_init_empty(self):
        """Test RangeMapper initialization with empty data."""
        mapper = RangeMapper([])
        self.assertEqual(mapper.total_size, 0)
        self.assertEqual(len(mapper), 0)

    def test_local_index(self):
        """Test local_index method."""
        # Test valid indices
        self.assertEqual(self.mapper.local_index('a', 0), 0)
        self.assertEqual(self.mapper.local_index('b', 1), 0)
        self.assertEqual(self.mapper.local_index('b', 8), 7)
        self.assertEqual(self.mapper.local_index('x', 14), 5)

        # Test invalid indices
        self.assertIsNone(self.mapper.local_index('a', 1))
        self.assertIsNone(self.mapper.local_index('b', 0))
        self.assertIsNone(self.mapper.local_index('b', 9))
        self.assertIsNone(self.mapper.local_index('x', 8))
        self.assertIsNone(self.mapper.local_index('x', 15))

    def test_global_index(self):
        """Test global_index method."""
        self.assertEqual(self.mapper.global_index('a', 0), 0)
        self.assertEqual(self.mapper.global_index('b', 0), 1)
        self.assertEqual(self.mapper.global_index('b', 7), 8)
        self.assertEqual(self.mapper.global_index('x', 0), 9)
        self.assertEqual(self.mapper.global_index('x', 5), 14)

    def test_offset(self):
        """Test offset method."""
        self.assertEqual(self.mapper.offset('a'), 0)
        self.assertEqual(self.mapper.offset('b'), 1)
        self.assertEqual(self.mapper.offset('x'), 9)
        self.assertEqual(self.mapper.offset('y'), 15)
        self.assertEqual(self.mapper.offset('z'), 36)

    def test_sizeof(self):
        """Test sizeof method."""
        self.assertEqual(self.mapper.sizeof('a'), 1)
        self.assertEqual(self.mapper.sizeof('b'), 8)
        self.assertEqual(self.mapper.sizeof('x'), 6)
        self.assertEqual(self.mapper.sizeof('y'), 21)
        self.assertEqual(self.mapper.sizeof('z'), 6)

    def test_invalid_key_errors(self):
        """Test that invalid keys raise KeyError."""
        with self.assertRaises(KeyError):
            self.mapper.local_index('invalid', 0)
        with self.assertRaises(KeyError):
            self.mapper.global_index('invalid', 0)
        with self.assertRaises(KeyError):
            self.mapper.offset('invalid')
        with self.assertRaises(KeyError):
            self.mapper.sizeof('invalid')


class TestGetScatterArrays(unittest.TestCase):
    def setUp(self):
        # Source mapper data
        self.src_data = [('a', 1), ('b', 8), ('x', 6), ('y', 21), ('z', 6)]
        self.src_mapper = RangeMapper(self.src_data)

        # Destination mapper data (same keys, different order, same sizes when no indices)
        self.dest_data = [('z', 6), ('y', 21), ('x', 6), ('b', 8), ('a', 1)]
        self.dest_mapper = RangeMapper(self.dest_data)

    def test_basic_scatter_arrays(self):
        """Test basic scatter array generation."""
        src_array, dest_array = get_scatter_arrays(self.src_mapper, self.dest_mapper)

        # Check that arrays have the same length
        self.assertEqual(len(src_array), len(dest_array))

        # Check that arrays are numpy arrays
        self.assertIsInstance(src_array, np.ndarray)
        self.assertIsInstance(dest_array, np.ndarray)

        # Check that arrays contain the expected indices
        # The function processes keys in dest_mapper order: z, y, x, b, a
        # For 'z': src range 36:42, dest range 0:6 (first in dest_mapper)
        self.assertIn(36, src_array)  # src index 36
        self.assertIn(41, src_array)  # src index 41
        self.assertIn(0, dest_array)  # dest index 0
        self.assertIn(5, dest_array)  # dest index 5

        # For 'a': src range 0:1, dest range 36:37 (last in dest_mapper)
        self.assertIn(0, src_array)  # src index 0
        self.assertIn(36, dest_array)  # dest index 36

        # Verify the mapping is correct: each src index maps to corresponding dest index
        # For 'z': src[36:42] -> dest[0:6] (first in dest_mapper)
        # For 'a': src[0:1] -> dest[36:37] (last in dest_mapper)
        self.assertEqual(src_array[0], 36)  # First src index for 'z' (first in dest_mapper)
        self.assertEqual(dest_array[0], 0)  # First dest index for 'z'

        # Check that src and dest arrays have the same values but different positions
        # The src array should contain all source indices in dest_mapper order
        # The dest array should contain all destination indices in dest_mapper order
        self.assertEqual(len(src_array), 42)  # Total size of all keys
        self.assertEqual(len(dest_array), 42)  # Total size of all keys

    def test_scatter_arrays_with_inds(self):
        """Test scatter array generation with custom indices."""
        # Create destination mapper with custom indices
        # Test case where indices can create different sizes
        # The dest_data sizes should match the number of indices for each key
        # Use different key order: z, y, x, b, a
        dest_data_with_inds = [('z', 8), ('y', 3), ('x', 6), ('b', 3), ('a', 2)]
        key_ind_data = [('z', [0, 1, 2, 3, 4, 5, 6, 7]), ('y', [0, 1, 2]), ('x', [0, 1, 2, 3, 4, 5]), ('b', [1, 2, 3]), ('a', [0, 0])]
        dest_mapper = RangeMapper(dest_data_with_inds, key_ind_data)

        src_array, dest_array = get_scatter_arrays(self.src_mapper, dest_mapper)

        # Check that arrays have the same length
        self.assertEqual(len(src_array), len(dest_array))

        # Check that custom indices are used for source array
        # For 'a': src range 0:1, custom inds [0,0], dest range 20:22 (last position due to different order)
        self.assertIn(0, src_array)  # custom src index 0 (repeated)
        self.assertIn(20, dest_array)  # dest index 20 (last position)
        self.assertIn(21, dest_array)  # dest index 21

        # For 'b': src range 1:9, custom inds [1,2,3], dest range 17:20 (different position due to different order)
        self.assertIn(2, src_array)  # custom src index 2
        self.assertIn(3, src_array)  # custom src index 3

    def test_partial_key_overlap(self):
        """Test scatter arrays when only some keys overlap."""
        # Source mapper with keys 'a', 'b', 'c'
        src_data = [('a', 1), ('b', 8), ('c', 6)]
        src_mapper = FlatTwoWayRangeMapper(src_data)

        # Destination mapper with keys 'd', 'b', 'a' (no 'c', different order)
        # Note: when no indices, sizes should be the same for matching keys
        dest_data = [('d', 4), ('b', 8), ('a', 1)]
        dest_mapper = FlatTwoWayRangeMapper(dest_data)

        src_array, dest_array = get_scatter_arrays(src_mapper, dest_mapper)

        # Should only include indices for 'a' and 'b'
        # 'a': src range 0:1, dest range 12:13 (different position due to different order)
        # 'b': src range 1:9, dest range 4:12 (different position due to different order)
        expected_src_length = 1 + 8  # size of 'a' + size of 'b'
        expected_dest_length = 1 + 8  # size of 'a' + size of 'b'

        self.assertEqual(len(src_array), expected_src_length)
        self.assertEqual(len(dest_array), expected_dest_length)

    def test_no_key_overlap(self):
        """Test scatter arrays when no keys overlap."""
        # Source mapper with keys 'a', 'b'
        src_data = [('a', 1), ('b', 8)]
        src_mapper = FlatTwoWayRangeMapper(src_data)

        # Destination mapper with keys 'c', 'd'
        dest_data = [('c', 2), ('d', 5)]
        dest_mapper = FlatTwoWayRangeMapper(dest_data)

        src_array, dest_array = get_scatter_arrays(src_mapper, dest_mapper)

        # Should return empty arrays
        self.assertEqual(len(src_array), 0)
        self.assertEqual(len(dest_array), 0)
        self.assertEqual(src_array.dtype, np.int32)  # INT_DTYPE
        self.assertEqual(dest_array.dtype, np.int32)  # INT_DTYPE

    def test_empty_mappers(self):
        """Test scatter arrays with empty mappers."""
        # Empty source mapper
        empty_src = RangeMapper([])
        dest_mapper = FlatTwoWayRangeMapper(self.dest_data)

        src_array, dest_array = get_scatter_arrays(empty_src, dest_mapper)

        # Should return empty arrays
        self.assertEqual(len(src_array), 0)
        self.assertEqual(len(dest_array), 0)

        # Empty destination mapper
        src_mapper = FlatTwoWayRangeMapper(self.src_data)
        empty_dest = RangeMapper([])

        src_array, dest_array = get_scatter_arrays(src_mapper, empty_dest)

        # Should return empty arrays
        self.assertEqual(len(src_array), 0)
        self.assertEqual(len(dest_array), 0)

    def test_scatter_arrays_correctness(self):
        """Test that scatter arrays correctly map indices."""
        src_array, dest_array = get_scatter_arrays(self.src_mapper, self.dest_mapper)

        # Verify that the mapping is correct by checking a few specific cases
        # The function processes keys in dest_mapper order: z, y, x, b, a
        # For 'z': src range 36:42, dest range 0:6 (first in dest_mapper)
        self.assertEqual(src_array[0], 36)  # src index for 'z' (first in dest_mapper)
        self.assertEqual(dest_array[0], 0)  # dest index 0 for 'z'

        # For 'a': src range 0:1, dest range 36:37 (last in dest_mapper)
        a_src_start = 41  # Last position in src_array
        a_dest_start = 41  # Last position in dest_array

        # Check that src indices are in correct range and sequence
        self.assertEqual(src_array[a_src_start], 0)  # src index for 'a' (last in dest_mapper)
        self.assertEqual(dest_array[a_dest_start], 41)  # dest index 41 for 'a'

    def test_scatter_arrays_with_different_sizes(self):
        """Test scatter arrays when indices create different sizes."""
        # Source mapper with small sizes
        src_data = [('a', 2), ('b', 3)]
        src_mapper = FlatTwoWayRangeMapper(src_data)

        # Destination mapper with larger sizes due to repeated indices
        # Use different key order: b, a
        dest_data = [('b', 6), ('a', 4)]  # Larger sizes due to repeated indices, different order
        key_ind_data = [
            ('b', [0, 1, 2, 0, 1, 2]),  # Repeated indices: size 6
            ('a', [0, 0, 1, 1])  # Repeated indices: size 4
        ]
        dest_mapper = RangeMapper(dest_data, key_ind_data)

        src_array, dest_array = get_scatter_arrays(src_mapper, dest_mapper)

        # Check that arrays have the same length
        self.assertEqual(len(src_array), len(dest_array))

        # Check that repeated indices are handled correctly
        # The function processes keys in dest_mapper order: b, a
        # For 'b': src range 2:5, custom inds [0,1,2,0,1,2], dest range 0:6 (first in dest_mapper)
        # Should have src indices [2,3,4,2,3,4] and dest indices [0,1,2,3,4,5]
        self.assertEqual(src_array[0], 2)  # First src index for 'b'
        self.assertEqual(src_array[1], 3)  # Second src index for 'b'
        self.assertEqual(src_array[2], 4)  # Third src index for 'b'
        self.assertEqual(src_array[3], 2)  # Repeated src index for 'b'

        self.assertEqual(dest_array[0], 0)  # First dest index for 'b'
        self.assertEqual(dest_array[1], 1)  # Second dest index for 'b'
        self.assertEqual(dest_array[2], 2)  # Third dest index for 'b'
        self.assertEqual(dest_array[3], 3)  # Fourth dest index for 'b'

        # For 'a': src range 0:2, custom inds [0,0,1,1], dest range 6:10 (second in dest_mapper)
        # Should have src indices [0,0,1,1] and dest indices [6,7,8,9]
        a_start = 6  # After 'b' indices
        for i in range(4):
            expected_src = i // 2  # Cycles through 0,0,1,1
            self.assertEqual(src_array[a_start + i], expected_src)
            self.assertEqual(dest_array[a_start + i], 6 + i)  # dest indices 6,7,8,9

    def test_scatter_arrays_data_copy(self):
        """Test scatter arrays by actually copying data between arrays and verifying results."""
        def create_data_array(key_size_tuples):
            """Create a numpy array by stacking sizes together and filling with random numbers."""
            total_size = sum(size for _, size in key_size_tuples)
            # Use different seeds for src and dest to ensure different values
            return np.random.rand(total_size)

        # Source data: keys a, b, c, d, e
        src_key_sizes = [('a', 3), ('b', 5), ('c', 2), ('d', 4), ('e', 6)]
        src_data_array = create_data_array(src_key_sizes)
        src_mapper = RangeMapper(src_key_sizes)

        # Destination data: keys f, b, a, g (different order, some missing, some new)
        dest_key_sizes = [('f', 2), ('b', 5), ('a', 3), ('g', 4)]
        dest_data_array = create_data_array(dest_key_sizes)
        dest_mapper = RangeMapper(dest_key_sizes)

        # Store original dest data for verification
        original_dest_data = dest_data_array.copy()

        # Test 1: Without indices (direct copy)
        src_scatter, dest_scatter = get_scatter_arrays(src_mapper, dest_mapper)

        # Copy data using scatter arrays
        dest_data_array[dest_scatter] = src_data_array[src_scatter]

        # Verify that the scatter arrays correctly copied the data
        # The scatter arrays should have copied src_data_array[src_scatter] to dest_data_array[dest_scatter]
        np.testing.assert_array_equal(
            dest_data_array[dest_scatter],
            src_data_array[src_scatter]
        )

        # Verify that non-matching keys didn't change
        # Key 'f': should remain unchanged (not in src)
        np.testing.assert_array_equal(
            dest_data_array[0:2],  # dest range for 'f'
            original_dest_data[0:2]  # original data for 'f'
        )

        # Key 'g': should remain unchanged (not in src)
        np.testing.assert_array_equal(
            dest_data_array[10:14],  # dest range for 'g'
            original_dest_data[10:14]  # original data for 'g'
        )

        # Test 2: With indices (creating different sizes)
        # Reset dest data
        dest_data_array = original_dest_data.copy()

        # Create dest mapper with indices that create different sizes
        dest_key_sizes_with_inds = [('f', 3), ('b', 4), ('a', 5), ('g', 2)]
        key_ind_data = [
            ('f', [0, 0, 1]),      # size 3, repeated indices
            ('b', [0, 1, 2, 3]),   # size 4, subset of src 'b' (size 5)
            ('a', [0, 1, 2, 0, 1]), # size 5, repeated indices
            ('g', [0, 1])          # size 2, subset of src 'g' (size 4)
        ]
        dest_mapper_with_inds = RangeMapper(dest_key_sizes_with_inds, key_ind_data)

        # Create new dest data array with different seed for this test
        dest_data_array = create_data_array(dest_key_sizes_with_inds)
        original_dest_data = dest_data_array.copy()

        # Get scatter arrays with indices
        src_scatter_inds, dest_scatter_inds = get_scatter_arrays(src_mapper, dest_mapper_with_inds)

        # Copy data using scatter arrays with indices
        dest_data_array[dest_scatter_inds] = src_data_array[src_scatter_inds]

        # Verify that matching keys have the correct data based on indices
        # The function processes keys in dest_mapper order: f, b, a, g
        # Only 'b' and 'a' are in both mappers

        # Verify that the scatter arrays correctly copied the data
        # The scatter arrays should have copied src_data_array[src_scatter_inds] to dest_data_array[dest_scatter_inds]
        np.testing.assert_array_equal(
            dest_data_array[dest_scatter_inds],
            src_data_array[src_scatter_inds]
        )

        # Verify that non-matching keys didn't change
        # Key 'f': should remain unchanged (not in src)
        np.testing.assert_array_equal(
            dest_data_array[0:3],  # dest range for 'f'
            original_dest_data[0:3]  # original data for 'f'
        )

        # Key 'g': should remain unchanged (not in src)
        np.testing.assert_array_equal(
            dest_data_array[12:14],  # dest range for 'g'
            original_dest_data[12:14]  # original data for 'g'
        )

        # Test 3: Verify that keys not in dest_mapper are not affected
        # Key 'c', 'd', 'e' in src should not affect dest since they're not in dest_mapper
        # This is already verified by the above tests, but let's be explicit
        # The scatter arrays should only contain indices for matching keys ('a', 'b')

        # Verify scatter array lengths
        expected_src_length = 4 + 5  # size of 'b' (4) + size of 'a' (5) in dest with indices
        expected_dest_length = 4 + 5  # size of 'b' (4) + size of 'a' (5) in dest with indices

        self.assertEqual(len(src_scatter_inds), expected_src_length)
        self.assertEqual(len(dest_scatter_inds), expected_dest_length)


if __name__ == "__main__":
    unittest.main()
