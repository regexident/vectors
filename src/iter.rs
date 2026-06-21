/// A fallible equivalent of [`FromIterator`].
///
/// This trait is used when constructing a collection from an iterator
/// may fail (e.g., due to capacity limits or validation errors). The
/// associated [`Error`](TryFromIterator::Error) type captures the
/// failure mode.
pub trait TryFromIterator<T>: Sized {
    /// The type returned when construction fails.
    type Error;

    /// Attempts to construct `Self` from an iterator, returning an
    /// error if the input cannot be fully consumed.
    fn try_from_iter<I>(iter: I) -> Result<Self, Self::Error>
    where
        I: IntoIterator<Item = T>;
}

/// A variant of [`FromIterator`] that silently truncates when the
/// target collection has a fixed capacity.
///
/// Unlike [`FromIterator`], this trait does not grow the underlying
/// storage. Once capacity is exhausted, remaining items are discarded
/// without error.
pub trait FromIteratorLossy<T>: Sized {
    /// Constructs `Self` from an iterator, discarding items beyond
    /// the collection's fixed capacity.
    fn from_iter_lossy<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = T>;
}
