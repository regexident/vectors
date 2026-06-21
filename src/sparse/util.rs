// MARK: Helper functions

#[cfg(feature = "alloc")]
use num_traits::Zero;

/// Sorts entries by index, deduplicates (keeping an arbitrary value for each index), drops zeros.
/// Returns the number of kept entries (callers should use `indices[..ret]`, `values[..ret]`).
///
/// ## Algorithm (combined heapsort + filter)
///
/// ```text
/// Phase 1 — Build max-heap (heap property: parent ≥ children by index).
///
/// Phase 2 — Extract in descending order, filtering on the fly.
///   Kept region grows from the right: [len-kept, len).
///
///   Each extraction yields the largest remaining index.
///   Since extraction is descending, the first occurrence of any index
///   value is the "last" in ascending order — equivalent to "keep last"
///   for an unstable sort.
///
///   For each extracted element at position `end`:
///     1. If zero (and drop_zeros): discard — leave at `end`.
///     2. If same index as `indices[len-kept]` (leftmost kept): duplicate, discard.
///     3. Otherwise: keep — swap into kept region at `len-kept-1`, increment `kept`.
///
/// Phase 3 — Compact: swap kept region [len-kept, len) to the front [0, kept).
/// ```
#[cfg(feature = "alloc")]
#[must_use]
pub fn canonicalize_entries<T, Idx>(
    indices: &mut [Idx],
    values: &mut [T],
    drop_zeros: bool,
) -> usize
where
    Idx: Ord + Copy,
    T: Zero + PartialEq + Clone,
{
    let len = indices.len();
    if len <= 1 {
        if len == 0 {
            return 0;
        }
        if drop_zeros && values[0].is_zero() {
            return 0;
        }
        return 1;
    }

    // Phase 1 — Build max-heap.
    for start in (0..len / 2).rev() {
        sift_down(indices, values, start, len);
    }

    // Phase 2 — Extract & filter.
    // Kept region: [len - kept, len).  Invariant: descending order by index.
    let mut kept = 0;

    for end in (1..len).rev() {
        // Extract the max (root) to position `end`.
        // The heap shrinks to [0, end), and `end` becomes the sorted suffix.
        indices.swap(0, end);
        values.swap(0, end);
        sift_down(indices, values, 0, end);

        if drop_zeros && values[end].is_zero() {
            continue;
        }

        // `indices[len - kept]` is the leftmost kept element (smallest kept index).
        // Since extraction is descending, a matching index means duplicate.
        let keep_pos = len - kept;
        if kept == 0 || indices[end] != indices[keep_pos] {
            let dest = keep_pos - 1;
            if dest != end {
                indices.swap(dest, end);
                values.swap(dest, end);
            }
            kept += 1;
        }
    }

    // Position 0 holds the smallest element (heap of size 1, never extracted).
    if !(drop_zeros && values[0].is_zero()) {
        let keep_pos = len - kept;
        if kept == 0 || indices[0] != indices[keep_pos] {
            let dest = keep_pos - 1;
            if dest != 0 {
                indices.swap(dest, 0);
                values.swap(dest, 0);
            }
            kept += 1;
        }
    }

    // Phase 3 — Compact: move kept elements from right side to front.
    let kept_start = len - kept;
    for i in 0..kept {
        indices.swap(i, kept_start + i);
        values.swap(i, kept_start + i);
    }

    kept
}

#[cfg(feature = "alloc")]
fn sift_down<T, Idx: Ord>(indices: &mut [Idx], values: &mut [T], mut root: usize, end: usize) {
    loop {
        let left = 2 * root + 1;
        let right = 2 * root + 2;
        let mut largest = root;

        if left < end && indices[left] > indices[largest] {
            largest = left;
        }
        if right < end && indices[right] > indices[largest] {
            largest = right;
        }

        if largest == root {
            break;
        }

        indices.swap(root, largest);
        values.swap(root, largest);
        root = largest;
    }
}

#[cfg(all(test, feature = "alloc"))]
mod tests {
    use super::*;

    #[test]
    fn canonicalize_entries_single() {
        let mut indices = [3usize];
        let mut values = [42i32];
        let len = canonicalize_entries(&mut indices, &mut values, false);
        assert_eq!(len, 1);
        assert_eq!(indices[..len], [3]);
        assert_eq!(values[..len], [42]);
    }

    #[test]
    fn canonicalize_entries_single_zero() {
        let mut indices = [3usize];
        let mut values = [0i32];
        let len = canonicalize_entries(&mut indices, &mut values, true);
        assert_eq!(len, 0);
    }

    #[test]
    fn canonicalize_entries_two_elements() {
        let mut indices = [2, 1];
        let mut values = [20, 10];
        let len = canonicalize_entries(&mut indices, &mut values, false);
        assert_eq!(len, 2);
        assert_eq!(&indices[..len], [1, 2]);
        assert_eq!(&values[..len], [10, 20]);
    }

    #[test]
    fn canonicalize_entries_reverse_order() {
        let mut indices = [8, 5, 3, 2, 1];
        let mut values = [10, 20, 30, 40, 50];
        let len = canonicalize_entries(&mut indices, &mut values, false);
        assert_eq!(len, 5);
        assert_eq!(&indices[..len], [1, 2, 3, 5, 8]);
        assert_eq!(&values[..len], [50, 40, 30, 20, 10]);
    }

    #[test]
    fn canonicalize_entries_values_stay_in_sync() {
        let mut indices = [5, 3, 1, 4, 2];
        let mut values = [50, 30, 10, 40, 20];
        let len = canonicalize_entries(&mut indices, &mut values, false);
        for i in 0..len {
            assert_eq!(values[i], indices[i] * 10);
        }
    }

    #[test]
    fn canonicalize_entries_no_dedup_no_zero() {
        let mut indices = [3, 1, 2];
        let mut values = [30, 10, 20];
        let len = canonicalize_entries(&mut indices, &mut values, false);
        assert_eq!(len, 3);
        assert_eq!(&indices[..len], [1, 2, 3]);
        assert_eq!(&values[..len], [10, 20, 30]);
    }

    #[test]
    fn canonicalize_entries_dedup_keeps_last() {
        let mut indices = [2, 1, 2, 1];
        let mut values = [20, 10, 21, 11];
        let len = canonicalize_entries(&mut indices, &mut values, false);
        assert_eq!(len, 2);
        assert_eq!(&indices[..len], [1, 2]);
        // Each surviving value must correspond to its index.
        assert_eq!(values[0] % 10, 1);
        assert_eq!(values[1] % 10, 0);
    }

    #[test]
    fn canonicalize_entries_drops_zeros() {
        let mut indices = [2, 0, 1];
        let mut values = [0, 5, 3];
        let len = canonicalize_entries(&mut indices, &mut values, true);
        assert_eq!(len, 2);
        assert_eq!(&indices[..len], [0, 1]);
        assert_eq!(&values[..len], [5, 3]);
    }

    #[test]
    fn canonicalize_entries_empty() {
        let mut indices: [usize; 0] = [];
        let mut values: [i32; 0] = [];
        let len = canonicalize_entries(&mut indices, &mut values, true);
        assert_eq!(len, 0);
    }

    #[test]
    fn canonicalize_entries_all_zeros() {
        let mut indices = [1, 2, 3];
        let mut values = [0, 0, 0];
        let len = canonicalize_entries(&mut indices, &mut values, true);
        assert_eq!(len, 0);
    }

    #[test]
    fn canonicalize_entries_all_duplicates() {
        let mut indices = [5, 5, 5];
        let mut values = [10, 20, 30];
        let len = canonicalize_entries(&mut indices, &mut values, false);
        assert_eq!(len, 1);
        assert_eq!(indices[0], 5);
        assert!(values[0] == 10 || values[0] == 20 || values[0] == 30);
    }
}
