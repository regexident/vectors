use arrayvec::ArrayVec;

/// Backing storage for vector components.
///
/// Implemented by `Vec<T>` and `ArrayVec<T, N>`.
pub trait Storage<T>:
    IntoIterator<Item = T> + AsRef<[T]> + AsMut<[T]> + FromIterator<T> + Extend<T>
{
    /// Returns the number of elements in the storage.
    fn len(&self) -> usize;

    /// Returns true if the storage has a length of 0.
    fn is_empty(&self) -> bool;

    /// Returns a slice containing the entire storage.
    fn as_slice(&self) -> &[T];

    /// Returns a mutable slice containing the entire storage.
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Truncates the storage to `len` elements.
    fn truncate(&mut self, len: usize);
}

impl<T> Storage<T> for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(&self)
    }

    fn is_empty(&self) -> bool {
        Vec::is_empty(&self)
    }

    fn as_slice(&self) -> &[T] {
        Vec::as_slice(self)
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        Vec::as_mut_slice(self)
    }

    fn truncate(&mut self, len: usize) {
        Vec::truncate(self, len);
    }
}

impl<T, const N: usize> Storage<T> for ArrayVec<T, N> {
    fn len(&self) -> usize {
        ArrayVec::len(&self)
    }

    fn is_empty(&self) -> bool {
        ArrayVec::is_empty(&self)
    }

    fn as_slice(&self) -> &[T] {
        ArrayVec::as_slice(self)
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        ArrayVec::as_mut_slice(self)
    }

    fn truncate(&mut self, len: usize) {
        ArrayVec::truncate(self, len);
    }
}
