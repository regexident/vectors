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

// MARK: Slice-based join iterators

/// An outer-join iterator over two sorted `&[(usize, T)]` slices.
pub struct OuterJoinIter<'a, T> {
    left: &'a [(usize, T)],
    right: &'a [(usize, T)],
    left_pos: usize,
    right_pos: usize,
}

/// An inner-join iterator over two sorted `&[(usize, T)]` slices.
pub struct InnerJoinIter<'a, T> {
    left: &'a [(usize, T)],
    right: &'a [(usize, T)],
    left_pos: usize,
    right_pos: usize,
}

/// Creates an outer-join iterator over two sorted slices.
pub fn outer_join<'a, T>(left: &'a [(usize, T)], right: &'a [(usize, T)]) -> OuterJoinIter<'a, T> {
    OuterJoinIter {
        left,
        right,
        left_pos: 0,
        right_pos: 0,
    }
}

/// Creates an inner-join iterator over two sorted slices.
pub fn inner_join<'a, T>(left: &'a [(usize, T)], right: &'a [(usize, T)]) -> InnerJoinIter<'a, T> {
    InnerJoinIter {
        left,
        right,
        left_pos: 0,
        right_pos: 0,
    }
}

impl<T: Copy> Iterator for OuterJoinIter<'_, T> {
    type Item = (usize, Join<T, T>);

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

impl<T: Copy> Iterator for InnerJoinIter<'_, T> {
    type Item = (usize, (T, T));

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
