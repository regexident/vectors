/// `SparseVector`'s `IntoIter`.
pub struct IntoIter<S>
where
    S: IntoIterator,
{
    inner: <S as IntoIterator>::IntoIter,
}

impl<S> IntoIter<S>
where
    S: IntoIterator,
{
    /// Creates an `IntoIter` from storage.
    pub fn new(storage: S) -> Self {
        IntoIter {
            inner: storage.into_iter(),
        }
    }
}

impl<T, S> Iterator for IntoIter<S>
where
    S: IntoIterator<Item = (usize, T)>,
{
    type Item = (usize, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<T, S> ExactSizeIterator for IntoIter<S>
where
    S: IntoIterator<Item = (usize, T)>,
    <S as IntoIterator>::IntoIter: ExactSizeIterator,
{
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// `&SparseVector`'s `Iter`.
pub struct Iter<'a, T> {
    inner: std::slice::Iter<'a, (usize, T)>,
}

impl<'a, T> Iter<'a, T> {
    /// Creates an `Iter` from a slice of sparse components.
    #[inline]
    pub fn new(slice: &'a [(usize, T)]) -> Self {
        Iter {
            inner: slice.iter(),
        }
    }
}

impl<'a, T> Iterator for Iter<'a, T>
where
    T: Copy,
{
    type Item = (usize, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().copied()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, T> ExactSizeIterator for Iter<'a, T>
where
    T: Copy,
{
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::sparse::SparseVector;

    #[test]
    fn into_iter() {
        let values = vec![(0, 0.1), (1, 0.2), (2, 0.3), (4, 0.4), (5, 0.5)];
        let sv = SparseVector::from(values.clone());
        let subject: Vec<_> = sv.into_iter().collect();
        assert_eq!(subject, values);
    }

    #[test]
    fn iter() {
        let values = vec![(0, 0.1), (1, 0.2), (2, 0.3), (4, 0.4), (5, 0.5)];
        let sv = SparseVector::from(values.clone());
        let subject: Vec<_> = sv.iter().collect();
        assert_eq!(subject, values);
    }
}
