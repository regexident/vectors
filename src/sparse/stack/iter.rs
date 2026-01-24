// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

use std::iter::FromIterator;

use arrayvec::ArrayVec;

use super::SparseVector;

pub use sparse::iter::{IntoIter, Iter};

impl<T, const N: usize> FromIterator<(usize, T)> for SparseVector<T, N> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = (usize, T)>>(iter: I) -> Self {
        let mut components = ArrayVec::new();
        components.extend(iter);
        Self { components }
    }
}

impl<T, const N: usize> IntoIterator for SparseVector<T, N> {
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = IntoIter<ArrayVec<(usize, T), N>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self.components)
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a SparseVector<T, N>
where
    T: 'a + Copy,
{
    type Item = <Self::IntoIter as Iterator>::Item;
    type IntoIter = Iter<'a, T>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Iter::new(&self.components[..])
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use std::iter::{FromIterator, IntoIterator};

    use expectest::prelude::*;

    #[test]
    fn from_iter() {
        const VALUES: [(usize, f32); 5] = [(0, 0.0), (1, 1.0), (2, 0.5), (4, 0.25), (8, 0.125)];
        let subject = SparseVector::from_iter(VALUES.iter().cloned());
        let expected = ArrayVec::from(VALUES);
        expect!(subject.components).to(be_equal_to(expected));
    }

    #[test]
    fn into_iter() {
        let values = vec![(0, 0.1), (1, 0.2), (2, 0.3), (4, 0.4), (5, 0.5)];
        let subject = SparseVector::from([(0, 0.1), (1, 0.2), (2, 0.3), (4, 0.4), (5, 0.5)]);
        let output: Vec<_> = subject.into_iter().collect();
        expect!(output).to(be_equal_to(values));
    }
}
