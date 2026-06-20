use std::fmt;

use super::DenseVector;
use crate::storage::Storage;

impl<T: fmt::Debug, S: Storage<T>> fmt::Debug for DenseVector<T, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let _ = write!(f, "[");
        for (index, item) in self.components.as_ref().iter().enumerate() {
            if index > 0 {
                write!(f, ", {:?}", item)?;
            } else {
                write!(f, "{:?}", item)?;
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
        let vector = DenseVector::from(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(format!("{:?}", vector), "[0.0, 0.25, 0.5, 0.75, 1.0]");
    }
}
