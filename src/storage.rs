use arrayvec::ArrayVec;

/// Backing storage for vector components.
///
/// Implemented by `Vec<T>` and `ArrayVec<T, N>`.
pub trait Storage<T>: IntoIterator<Item = T> + AsRef<[T]> + AsMut<[T]> + Extend<T> {
    /// Creates a new storage from an iterator.
    fn from_iter_in_place(iter: impl Iterator<Item = T>) -> Self;

    /// Truncates the storage to `len` elements.
    fn truncate(&mut self, len: usize);
}

impl<T> Storage<T> for Vec<T> {
    fn from_iter_in_place(iter: impl Iterator<Item = T>) -> Self {
        iter.collect()
    }

    fn truncate(&mut self, len: usize) {
        Vec::truncate(self, len);
    }
}

impl<T, const N: usize> Storage<T> for ArrayVec<T, N> {
    fn from_iter_in_place(iter: impl Iterator<Item = T>) -> Self {
        let mut av = ArrayVec::new();
        av.extend(iter);

        av
    }

    fn truncate(&mut self, len: usize) {
        ArrayVec::truncate(self, len);
    }
}
