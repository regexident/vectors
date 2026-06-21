use std::cmp::Ordering;

/// The result of joining two sorted slices.
pub enum Join<L, R> {
    /// Both slices have a value at this index.
    Both(L, R),
    /// Only the left slice has a value at this index.
    Left(L),
    /// Only the right slice has a value at this index.
    Right(R),
}

// MARK: Outer Join

/// Creates an outer-join iterator over two sorted slices.
pub fn outer_join<'a, Idx, T>(
    left: &'a [(Idx, T)],
    right: &'a [(Idx, T)],
) -> OuterJoinIter<'a, Idx, T> {
    OuterJoinIter {
        left,
        right,
        left_pos: 0,
        right_pos: 0,
    }
}

/// An outer-join iterator over two sorted `&[(Idx, T)]` slices.
pub struct OuterJoinIter<'a, Idx, T> {
    left: &'a [(Idx, T)],
    right: &'a [(Idx, T)],
    left_pos: usize,
    right_pos: usize,
}

impl<Idx, T> Iterator for OuterJoinIter<'_, Idx, T>
where
    Idx: Ord + Copy,
    T: Copy,
{
    type Item = (Idx, Join<T, T>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.left_pos >= self.left.len() && self.right_pos >= self.right.len() {
            return None;
        }

        if self.left_pos >= self.left.len() {
            let (ri, rv) = self.right[self.right_pos];
            self.right_pos += 1;
            return Some((ri, Join::Right(rv)));
        }

        if self.right_pos >= self.right.len() {
            let (li, lv) = self.left[self.left_pos];
            self.left_pos += 1;
            return Some((li, Join::Left(lv)));
        }

        let (li, lv) = self.left[self.left_pos];
        let (ri, rv) = self.right[self.right_pos];

        match li.cmp(&ri) {
            Ordering::Equal => {
                self.left_pos += 1;
                self.right_pos += 1;
                Some((li, Join::Both(lv, rv)))
            }
            Ordering::Less => {
                self.left_pos += 1;
                Some((li, Join::Left(lv)))
            }
            Ordering::Greater => {
                self.right_pos += 1;
                Some((ri, Join::Right(rv)))
            }
        }
    }
}

// MARK: Inner Join

/// Creates an inner-join iterator over two sorted slices.
pub fn inner_join<'a, Idx, T>(
    left: &'a [(Idx, T)],
    right: &'a [(Idx, T)],
) -> InnerJoinIter<'a, Idx, T> {
    InnerJoinIter {
        left,
        right,
        left_pos: 0,
        right_pos: 0,
    }
}
/// An inner-join iterator over two sorted `&[(Idx, T)]` slices.
pub struct InnerJoinIter<'a, Idx, T> {
    left: &'a [(Idx, T)],
    right: &'a [(Idx, T)],
    left_pos: usize,
    right_pos: usize,
}

impl<Idx, T> Iterator for InnerJoinIter<'_, Idx, T>
where
    Idx: Ord + Copy,
    T: Copy,
{
    type Item = (Idx, (T, T));

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.left_pos >= self.left.len() || self.right_pos >= self.right.len() {
                return None;
            }

            let (li, lv) = self.left[self.left_pos];
            let (ri, rv) = self.right[self.right_pos];

            match li.cmp(&ri) {
                Ordering::Equal => {
                    self.left_pos += 1;
                    self.right_pos += 1;
                    return Some((li, (lv, rv)));
                }
                Ordering::Less => {
                    self.left_pos += 1;
                }
                Ordering::Greater => {
                    self.right_pos += 1;
                }
            }
        }
    }
}
