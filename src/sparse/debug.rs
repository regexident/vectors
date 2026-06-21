use std::fmt;

use super::SparseVector;
use crate::storage::Storage;

impl<T, Idx, S> fmt::Debug for SparseVector<Idx, T, S>
where
    Idx: fmt::Display,
    T: Copy + fmt::Debug,
    S: Storage<(Idx, T)>,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let _ = write!(f, "[");
        for (fmt_idx, (index, value)) in self.as_slice().iter().enumerate() {
            if fmt_idx > 0 {
                write!(f, ", ({}, {:?})", index, value)?;
            } else {
                write!(f, "({}, {:?})", index, value)?;
            }
        }
        let _ = write!(f, "]");
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn debug() {
        let vector = SparseVector::try_from(vec![(0, 0.2), (1, 0.5), (2, 1.0), (4, 2.0)]).unwrap();
        assert_eq!(
            format!("{:?}", vector),
            "[(0, 0.2), (1, 0.5), (2, 1.0), (4, 2.0)]"
        );
    }
}
