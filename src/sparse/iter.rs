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

impl<Idx, T, S> Iterator for IntoIter<S>
where
    S: IntoIterator<Item = (Idx, T)>,
{
    type Item = (Idx, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<Idx, T, S> ExactSizeIterator for IntoIter<S>
where
    S: IntoIterator<Item = (Idx, T)>,
    <S as IntoIterator>::IntoIter: ExactSizeIterator,
{
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// `&SparseVector`'s `Iter`.
pub struct Iter<'a, Idx, T> {
    inner: std::slice::Iter<'a, (Idx, T)>,
}

impl<'a, Idx, T> Iter<'a, Idx, T> {
    /// Creates an `Iter` from a slice of sparse components.
    #[inline]
    pub fn new(slice: &'a [(Idx, T)]) -> Self {
        Iter {
            inner: slice.iter(),
        }
    }
}

impl<'a, Idx, T> Iterator for Iter<'a, Idx, T>
where
    Idx: Copy,
    T: Copy,
{
    type Item = (Idx, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().copied()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<'a, Idx, T> ExactSizeIterator for Iter<'a, Idx, T>
where
    Idx: Copy,
    T: Copy,
{
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

#[cfg(test)]
mod test {
    use crate::sparse::SparseVector;

    #[test]
    fn into_iter() {
        let values = vec![(0, 0.1), (1, 0.2), (2, 0.3), (4, 0.4), (5, 0.5)];
        let sv = SparseVector::try_from(values.clone()).unwrap();
        let subject: Vec<_> = sv.into_iter().collect();
        assert_eq!(subject, values);
    }

    #[test]
    fn iter() {
        let values = vec![(0, 0.1), (1, 0.2), (2, 0.3), (4, 0.4), (5, 0.5)];
        let sv = SparseVector::try_from(values.clone()).unwrap();
        let subject: Vec<_> = sv.iter().collect();
        assert_eq!(subject, values);
    }
}
