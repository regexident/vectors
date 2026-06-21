use core::cmp::Ordering;

/// The result of a join between entries at matching positions.
#[derive(Clone, PartialEq, Debug)]
pub enum Join<'a, T> {
    /// An entry that is only present in the left vector.
    Left(&'a T),
    /// An entry only present in the right vector.
    Right(&'a T),
    /// An entry present in both vectors.
    Both(&'a T, &'a T),
}

/// Iterator returned by [`outer_join`].
pub struct OuterJoin<'a, T, Idx, F> {
    left_i: &'a [Idx],
    left_v: &'a [T],
    right_i: &'a [Idx],
    right_v: &'a [T],
    left_pos: usize,
    right_pos: usize,
    merge: F,
}

impl<'a, T, Idx, F> Iterator for OuterJoin<'a, T, Idx, F>
where
    Idx: Ord + Copy + 'a,
    F: Fn(Idx, Join<'a, T>) -> T,
{
    type Item = (Idx, T);

    fn next(&mut self) -> Option<Self::Item> {
        let left_done = self.left_pos >= self.left_i.len();
        let right_done = self.right_pos >= self.right_i.len();

        if left_done && right_done {
            return None;
        }

        if right_done || (!left_done && self.left_i[self.left_pos] < self.right_i[self.right_pos]) {
            let key = self.left_i[self.left_pos];
            let item = (
                key,
                (self.merge)(key, Join::Left(&self.left_v[self.left_pos])),
            );
            self.left_pos += 1;

            Some(item)
        } else if left_done || self.right_i[self.right_pos] < self.left_i[self.left_pos] {
            let key = self.right_i[self.right_pos];
            let item = (
                key,
                (self.merge)(key, Join::Right(&self.right_v[self.right_pos])),
            );
            self.right_pos += 1;

            Some(item)
        } else {
            let key = self.left_i[self.left_pos];
            let item = (
                key,
                (self.merge)(
                    key,
                    Join::Both(&self.left_v[self.left_pos], &self.right_v[self.right_pos]),
                ),
            );
            self.left_pos += 1;
            self.right_pos += 1;

            Some(item)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left_remaining = self.left_i.len() - self.left_pos;
        let right_remaining = self.right_i.len() - self.right_pos;
        let max = left_remaining + right_remaining;
        let min = left_remaining.max(right_remaining);

        (min, Some(max))
    }
}

/// Apply a merge function to an outer join of two sorted sparse-vector entries.
///
/// Returns an iterator yielding `(index, merged-value)` pairs in ascending
/// index order. Indices present in only one side are merged via [`Join::Left`]
/// or [`Join::Right`]; indices present in both yield [`Join::Both`].
#[must_use]
pub fn outer_join<'a, T, Idx, F>(
    left_i: &'a [Idx],
    left_v: &'a [T],
    right_i: &'a [Idx],
    right_v: &'a [T],
    merge: F,
) -> OuterJoin<'a, T, Idx, F>
where
    Idx: Ord + Copy + 'a,
    F: Fn(Idx, Join<'a, T>) -> T,
{
    OuterJoin {
        left_i,
        left_v,
        right_i,
        right_v,
        left_pos: 0,
        right_pos: 0,
        merge,
    }
}

/// Iterator returned by [`inner_join`].
pub struct InnerJoin<'a, T, Idx, F> {
    left_i: &'a [Idx],
    left_v: &'a [T],
    right_i: &'a [Idx],
    right_v: &'a [T],
    left_pos: usize,
    right_pos: usize,
    merge: F,
}

impl<'a, T, Idx, F> Iterator for InnerJoin<'a, T, Idx, F>
where
    Idx: Ord + Copy + 'a,
    F: Fn(Idx, &'a T, &'a T) -> T,
{
    type Item = (Idx, T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.left_pos < self.left_i.len() && self.right_pos < self.right_i.len() {
            match self.left_i[self.left_pos].cmp(&self.right_i[self.right_pos]) {
                Ordering::Less => {
                    self.left_pos += 1;
                }
                Ordering::Greater => {
                    self.right_pos += 1;
                }
                Ordering::Equal => {
                    let key = self.left_i[self.left_pos];
                    let item = (
                        key,
                        (self.merge)(
                            key,
                            &self.left_v[self.left_pos],
                            &self.right_v[self.right_pos],
                        ),
                    );
                    self.left_pos += 1;
                    self.right_pos += 1;

                    return Some(item);
                }
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left_remaining = self.left_i.len() - self.left_pos;
        let right_remaining = self.right_i.len() - self.right_pos;

        (0, Some(left_remaining.min(right_remaining)))
    }
}

/// Inner join: yields only entries where both sides share an index.
///
/// Returns an iterator yielding `(index, merged-value)` pairs for indices
/// present in both inputs, in ascending index order.
#[must_use]
pub fn inner_join<'a, T, Idx, F>(
    left_i: &'a [Idx],
    left_v: &'a [T],
    right_i: &'a [Idx],
    right_v: &'a [T],
    merge: F,
) -> InnerJoin<'a, T, Idx, F>
where
    Idx: Ord + Copy + 'a,
    F: Fn(Idx, &'a T, &'a T) -> T,
{
    InnerJoin {
        left_i,
        left_v,
        right_i,
        right_v,
        left_pos: 0,
        right_pos: 0,
        merge,
    }
}

/// Threshold ratio (large/small) above which galloping is used instead of merge.
///
/// Chosen by sweeping `{4, 8, 16, 32, 64}` on skewed scenarios and picking
/// the value where galloping reliably beats merge without regressing balanced inputs.
pub const ADAPTIVE_THRESHOLD: usize = 16;

/// A galloping inner-join iterator over two sorted index/value slice pairs.
///
/// Uses exponential (galloping) search to skip ahead in the larger array,
/// yielding O(small * log large) comparisons in the worst case instead of
/// O(small + large). Most effective when the larger side is at least
/// [`ADAPTIVE_THRESHOLD`] times the size of the smaller side.
pub struct InnerJoinGalloping<'a, T, Idx, F> {
    small_i: &'a [Idx],
    small_v: &'a [T],
    large_i: &'a [Idx],
    large_v: &'a [T],
    small_pos: usize,
    cursor: usize,
    merge: F,
}

impl<'a, T, Idx, F> Iterator for InnerJoinGalloping<'a, T, Idx, F>
where
    Idx: Ord + Copy + 'a,
    F: Fn(Idx, &'a T, &'a T) -> T,
{
    type Item = (Idx, T);

    fn next(&mut self) -> Option<Self::Item> {
        while self.small_pos < self.small_i.len() && self.cursor < self.large_i.len() {
            let target = unsafe { *self.small_i.get_unchecked(self.small_pos) };
            let current = unsafe { *self.large_i.get_unchecked(self.cursor) };

            match target.cmp(&current) {
                Ordering::Equal => {
                    let item = (
                        target,
                        (self.merge)(
                            target,
                            &self.small_v[self.small_pos],
                            &self.large_v[self.cursor],
                        ),
                    );
                    self.small_pos += 1;
                    self.cursor += 1;
                    return Some(item);
                }
                Ordering::Less => {
                    self.small_pos += 1;
                }
                Ordering::Greater => {
                    let n = self.large_i.len();
                    if self.cursor + 1 >= n {
                        self.cursor = n;
                        self.small_pos += 1;
                        continue;
                    }

                    // Phase 1: exponential search for first probe >= target
                    let (last_lt, final_step) = {
                        let mut step: usize = 1;
                        let mut last_lt = self.cursor;
                        loop {
                            let probe = self.cursor.saturating_add(step);
                            if probe >= n {
                                break (last_lt, 0);
                            }
                            if unsafe { *self.large_i.get_unchecked(probe) } >= target {
                                break (last_lt, step);
                            }
                            last_lt = probe;
                            step = match step.checked_shl(1) {
                                Some(s) => s,
                                None => break (last_lt, 0),
                            };
                        }
                    };

                    if final_step == 0 {
                        self.cursor = n;
                        self.small_pos += 1;
                        continue;
                    }

                    // Phase 2: one-sided galloping (halving) toward target
                    let mut step = final_step >> 1;
                    let mut pos = last_lt;
                    while step > 0 {
                        let probe = pos + step;
                        if probe < n && unsafe { *self.large_i.get_unchecked(probe) } < target {
                            pos = probe;
                        }
                        step >>= 1;
                    }

                    let insertion = pos + 1;
                    if insertion < n && unsafe { *self.large_i.get_unchecked(insertion) } == target
                    {
                        let item = (
                            target,
                            (self.merge)(
                                target,
                                &self.small_v[self.small_pos],
                                &self.large_v[insertion],
                            ),
                        );
                        self.cursor = insertion + 1;
                        self.small_pos += 1;
                        return Some(item);
                    } else {
                        self.cursor = insertion;
                        self.small_pos += 1;
                    }
                }
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let small_remaining = self.small_i.len() - self.small_pos;
        (0, Some(small_remaining))
    }
}

/// Creates a galloping inner-join iterator over two sorted index/value slice pairs.
///
/// The smaller of the two inputs is iterated element-by-element while the
/// larger side is searched with exponential (galloping) probes. For balanced
/// inputs prefer [`inner_join`]; for skewed inputs this can be significantly faster.
#[must_use]
pub fn inner_join_galloping<'a, T, Idx, F>(
    left_i: &'a [Idx],
    left_v: &'a [T],
    right_i: &'a [Idx],
    right_v: &'a [T],
    merge: F,
) -> InnerJoinGalloping<'a, T, Idx, F>
where
    T: 'a,
    Idx: 'a + Ord + Copy,
{
    if left_i.len() <= right_i.len() {
        InnerJoinGalloping {
            small_i: left_i,
            small_v: left_v,
            large_i: right_i,
            large_v: right_v,
            small_pos: 0,
            cursor: 0,
            merge,
        }
    } else {
        InnerJoinGalloping {
            small_i: right_i,
            small_v: right_v,
            large_i: left_i,
            large_v: left_v,
            small_pos: 0,
            cursor: 0,
            merge,
        }
    }
}

/// Inner join with adaptive dispatch between merge and galloping strategies.
///
/// Returns a `Vec` of `(index, merged-value)` pairs. Uses the merge-based
/// iterator for balanced inputs and the galloping iterator when one side
/// is at least [`ADAPTIVE_THRESHOLD`] times larger than the other.
#[must_use]
pub fn inner_join_adaptive<'a, T, Idx, F>(
    left_i: &'a [Idx],
    left_v: &'a [T],
    right_i: &'a [Idx],
    right_v: &'a [T],
    merge: F,
) -> Vec<(Idx, T)>
where
    T: 'a + Copy,
    Idx: 'a + Ord + Copy,
    F: Fn(Idx, &'a T, &'a T) -> T,
{
    let (n, m) = (left_i.len(), right_i.len());
    if n == 0 || m == 0 {
        return Vec::new();
    }
    if n.max(m) >= ADAPTIVE_THRESHOLD * n.min(m) {
        inner_join_galloping(left_i, left_v, right_i, right_v, merge).collect()
    } else {
        inner_join(left_i, left_v, right_i, right_v, merge).collect()
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn outer_merge(_idx: u32, join: Join<'_, i32>) -> i32 {
        match join {
            Join::Left(l) => *l,
            Join::Right(r) => *r,
            Join::Both(l, r) => *l + *r,
        }
    }

    fn inner_merge(_idx: u32, l: &i32, r: &i32) -> i32 {
        *l + *r
    }

    #[test]
    fn outer_join_disjoint() {
        let left_i = [0_u32, 2];
        let left_v = [10_i32, 20];
        let right_i = [1_u32, 3];
        let right_v = [100_i32, 200];

        let result: Vec<_> =
            outer_join(&left_i, &left_v, &right_i, &right_v, outer_merge).collect();

        assert_eq!(result, vec![(0, 10), (1, 100), (2, 20), (3, 200)]);
    }

    #[test]
    fn outer_join_overlapping() {
        let left_i = [0_u32, 1, 2];
        let left_v = [10_i32, 20, 30];
        let right_i = [1_u32, 2, 3];
        let right_v = [100_i32, 200, 300];

        let result: Vec<_> =
            outer_join(&left_i, &left_v, &right_i, &right_v, outer_merge).collect();

        assert_eq!(result, vec![(0, 10), (1, 120), (2, 230), (3, 300)]);
    }

    #[test]
    fn outer_join_empty_sides() {
        let empty_i: [u32; 0] = [];
        let empty_v: [i32; 0] = [];
        let right_i = [1_u32, 2];
        let right_v = [100_i32, 200];

        let result: Vec<_> =
            outer_join(&empty_i, &empty_v, &right_i, &right_v, outer_merge).collect();
        assert_eq!(result, vec![(1, 100), (2, 200)]);

        let result: Vec<_> =
            outer_join(&right_i, &right_v, &empty_i, &empty_v, outer_merge).collect();
        assert_eq!(result, vec![(1, 100), (2, 200)]);

        let result: Vec<_> =
            outer_join(&empty_i, &empty_v, &empty_i, &empty_v, outer_merge).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn inner_join_intersection() {
        let left_i = [0_u32, 1, 2, 4];
        let left_v = [10_i32, 20, 30, 40];
        let right_i = [1_u32, 2, 3, 4];
        let right_v = [100_i32, 200, 300, 400];

        let result: Vec<_> =
            inner_join(&left_i, &left_v, &right_i, &right_v, inner_merge).collect();

        assert_eq!(result, vec![(1, 120), (2, 230), (4, 440)]);
    }

    #[test]
    fn inner_join_no_overlap() {
        let left_i = [0_u32, 2, 4];
        let left_v = [10_i32, 20, 30];
        let right_i = [1_u32, 3, 5];
        let right_v = [100_i32, 200, 300];

        let result: Vec<_> =
            inner_join(&left_i, &left_v, &right_i, &right_v, inner_merge).collect();

        assert!(result.is_empty());
    }

    #[test]
    fn size_hints_are_bounded() {
        let left_i = [0_u32, 1, 2];
        let left_v = [10_i32, 20, 30];
        let right_i = [1_u32, 2, 3, 4];
        let right_v = [100_i32, 200, 300, 400];

        let outer = outer_join(&left_i, &left_v, &right_i, &right_v, outer_merge);
        assert_eq!(outer.size_hint(), (4, Some(7)));

        let inner = inner_join(&left_i, &left_v, &right_i, &right_v, inner_merge);
        assert_eq!(inner.size_hint(), (0, Some(3)));
    }

    #[test]
    fn galloping_intersection() {
        let left_i = [0_u32, 1, 2, 4];
        let left_v = [10_i32, 20, 30, 40];
        let right_i = [1_u32, 2, 3, 4];
        let right_v = [100_i32, 200, 300, 400];

        let result: Vec<_> =
            inner_join_galloping(&left_i, &left_v, &right_i, &right_v, inner_merge).collect();

        assert_eq!(result, vec![(1, 120), (2, 230), (4, 440)]);
    }

    #[test]
    fn galloping_small_left() {
        let left_i = [0_u32, 10, 100];
        let left_v = [10_i32, 20, 30];
        let right_i: Vec<u32> = (0..=200u32).step_by(2).collect();
        let right_v: Vec<i32> = (0..=200i32).step_by(2).collect();

        let expected: Vec<_> =
            inner_join(&left_i, &left_v, &right_i, &right_v, inner_merge).collect();
        let result: Vec<_> =
            inner_join_galloping(&left_i, &left_v, &right_i, &right_v, inner_merge).collect();

        assert_eq!(result, expected);
    }

    #[test]
    fn galloping_small_right() {
        let left_i: Vec<u32> = (0..=200u32).step_by(2).collect();
        let left_v: Vec<i32> = (0..=200i32).step_by(2).collect();
        let right_i = [0_u32, 10, 100];
        let right_v = [10_i32, 20, 30];

        let expected: Vec<_> =
            inner_join(&left_i, &left_v, &right_i, &right_v, inner_merge).collect();
        let result: Vec<_> =
            inner_join_galloping(&left_i, &left_v, &right_i, &right_v, inner_merge).collect();

        assert_eq!(result, expected);
    }

    #[test]
    fn galloping_no_overlap() {
        let left_i = [0_u32, 2, 4];
        let left_v = [10_i32, 20, 30];
        let right_i = [1_u32, 3, 5];
        let right_v = [100_i32, 200, 300];

        let result: Vec<_> =
            inner_join_galloping(&left_i, &left_v, &right_i, &right_v, inner_merge).collect();

        assert!(result.is_empty());
    }

    #[test]
    fn galloping_empty_sides() {
        let empty: [u32; 0] = [];
        let empty_v: [i32; 0] = [];
        let right_i = [1_u32, 2];
        let right_v = [100_i32, 200];

        let result: Vec<_> =
            inner_join_galloping(&empty, &empty_v, &right_i, &right_v, inner_merge).collect();
        assert!(result.is_empty());

        let result: Vec<_> =
            inner_join_galloping(&right_i, &right_v, &empty, &empty_v, inner_merge).collect();
        assert!(result.is_empty());

        let result: Vec<_> =
            inner_join_galloping(&empty, &empty_v, &empty, &empty_v, inner_merge).collect();
        assert!(result.is_empty());
    }

    #[test]
    fn galloping_all_common() {
        let left_i: Vec<u32> = (0..100).collect();
        let left_v: Vec<i32> = (0..100).collect();
        let right_i: Vec<u32> = (0..100).collect();
        let right_v: Vec<i32> = (0..100).collect();

        let expected: Vec<_> =
            inner_join(&left_i, &left_v, &right_i, &right_v, inner_merge).collect();
        let result: Vec<_> =
            inner_join_galloping(&left_i, &left_v, &right_i, &right_v, inner_merge).collect();

        assert_eq!(result, expected);
    }

    #[test]
    fn adaptive_inner_join_matches_merge() {
        let cases: [(Vec<u32>, Vec<u32>); 4] = [
            (vec![0, 1, 2, 4], vec![1, 2, 3, 4]),
            (vec![0, 10, 100], (0..=200u32).step_by(2).collect()),
            ((0..=200u32).step_by(2).collect(), vec![0, 10, 100]),
            (vec![0, 2, 4], vec![1, 3, 5]),
        ];
        let values = |i: &[u32]| -> Vec<i32> { i.iter().map(|&x| x as i32 * 10).collect() };

        for (l_i, r_i) in cases {
            let l_v = values(&l_i);
            let r_v = values(&r_i);

            let expected: Vec<_> = inner_join(&l_i, &l_v, &r_i, &r_v, inner_merge).collect();
            let result = inner_join_adaptive(&l_i, &l_v, &r_i, &r_v, inner_merge);

            assert_eq!(result, expected);
        }
    }
}
