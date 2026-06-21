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
}
