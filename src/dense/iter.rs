/// `DenseVector`'s `IntoIter`.
pub struct IntoIter<S>
where
    S: IntoIterator,
{
    index: usize,
    inner: <S as IntoIterator>::IntoIter,
}

impl<S> IntoIter<S>
where
    S: IntoIterator,
{
    /// Creates an `IntoIter` from storage.
    pub fn new(storage: S) -> Self {
        IntoIter {
            index: 0,
            inner: storage.into_iter(),
        }
    }
}

impl<T, S> Iterator for IntoIter<S>
where
    S: IntoIterator<Item = T>,
{
    type Item = (usize, T);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index;
        if let Some(value) = self.inner.next() {
            self.index += 1;
            Some((index, value))
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<S> ExactSizeIterator for IntoIter<S>
where
    S: IntoIterator,
    <S as IntoIterator>::IntoIter: ExactSizeIterator,
{
    #[inline]
    fn len(&self) -> usize {
        self.inner.len()
    }
}

/// `&DenseVector`'s `Iter`.
pub struct Iter<'a, T> {
    index: usize,
    inner: std::slice::Iter<'a, T>,
}

impl<'a, T> Iter<'a, T> {
    /// Creates an `Iter` from a slice of dense components.
    pub fn new(slice: &'a [T]) -> Self {
        Iter {
            index: 0,
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
        self.inner.next().map(|value| {
            let index = self.index;
            self.index += 1;
            (index, *value)
        })
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

    use crate::dense::DenseVector;

    #[test]
    fn into_iter() {
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let iter = IntoIter::new(values);
        let subject: Vec<_> = iter.collect();
        let expected = vec![(0, 0.1), (1, 0.2), (2, 0.3), (3, 0.4), (4, 0.5)];
        assert_eq!(subject, expected);
    }

    #[test]
    fn iter() {
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let dv = DenseVector::from(values);
        let subject: Vec<_> = dv.iter().collect();
        let expected = vec![(0, 0.1), (1, 0.2), (2, 0.3), (3, 0.4), (4, 0.5)];
        assert_eq!(subject, expected);
    }

    #[test]
    fn from_iter() {
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let subject: DenseVector<_, Vec<_>> = values.into_iter().collect();
        let expected: Vec<_> = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        assert_eq!(subject.as_slice(), expected.as_slice());
    }
}
