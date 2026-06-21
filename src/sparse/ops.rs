use std::ops::{Add, Mul, Sub};

use num_traits::{MulAdd, Zero};

use crate::{
    Index,
    sparse::{FromCanonicalPairs, Join, SparseVector, outer_join},
};

use super::{GenericSparseVec, SparseStorage};

impl<Idx, T, S> Add for GenericSparseVec<Idx, T, S>
where
    Idx: Index,
    T: Copy + Add<Output = T> + Zero + PartialEq,
    S: SparseStorage<Idx, T> + FromCanonicalPairs<Idx, T>,
    Self: SparseVector<Index = Idx, Value = T>,
{
    type Output = GenericSparseVec<Idx, T, S>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.is_empty() {
            return rhs;
        }

        if rhs.is_empty() {
            return self;
        }

        let items = outer_join(
            self.storage.indices(),
            self.storage.values(),
            rhs.storage.indices(),
            rhs.storage.values(),
            |_idx, join| match join {
                Join::Left(lhs) => *lhs,
                Join::Right(rhs) => *rhs,
                Join::Both(lhs, rhs) => *lhs + *rhs,
            },
        )
        .filter(|(_, v)| !v.is_zero());

        Self::Output::from(S::from_canonical_pairs(items))
    }
}

impl<Idx, T, S> Sub for GenericSparseVec<Idx, T, S>
where
    Idx: Index,
    T: Copy + Sub<Output = T> + Zero + PartialEq,
    S: SparseStorage<Idx, T> + FromCanonicalPairs<Idx, T>,
    Self: SparseVector<Index = Idx, Value = T>,
{
    type Output = GenericSparseVec<Idx, T, S>;

    fn sub(self, rhs: Self) -> Self::Output {
        if rhs.is_empty() {
            return self;
        }

        let items = outer_join(
            self.storage.indices(),
            self.storage.values(),
            rhs.storage.indices(),
            rhs.storage.values(),
            |_idx, join| match join {
                Join::Left(lhs) => *lhs,
                Join::Right(rhs) => T::zero() - *rhs,
                Join::Both(lhs, rhs) => *lhs - *rhs,
            },
        )
        .filter(|(_, v)| !v.is_zero());

        Self::Output::from(S::from_canonical_pairs(items))
    }
}

// MARK: Scalar multiplication

impl<Idx, T, S> Mul<T> for GenericSparseVec<Idx, T, S>
where
    Idx: Index,
    T: Copy + Mul<Output = T> + Zero + PartialEq,
    S: SparseStorage<Idx, T> + FromCanonicalPairs<Idx, T>,
    Self: SparseVector<Index = Idx, Value = T>,
{
    type Output = GenericSparseVec<Idx, T, S>;

    fn mul(self, rhs: T) -> Self::Output {
        if rhs == T::zero() {
            return Self::Output::from(S::from_canonical_pairs([]));
        }

        let items = self
            .storage
            .into_iter()
            .map(|(index, value)| (index, rhs * value))
            .filter(|(_, v)| !v.is_zero());

        Self::Output::from(S::from_canonical_pairs(items))
    }
}

// MARK: MulAdd

impl<Idx, T, S: Clone> MulAdd<T, &GenericSparseVec<Idx, T, S>> for GenericSparseVec<Idx, T, S>
where
    Idx: Index,
    T: Copy + Mul<Output = T> + Add<Output = T> + Zero + PartialEq,
    S: SparseStorage<Idx, T> + FromCanonicalPairs<Idx, T>,
    Self: SparseVector<Index = Idx, Value = T>,
{
    type Output = GenericSparseVec<Idx, T, S>;

    fn mul_add(self, a: T, b: &Self) -> Self::Output {
        if self.is_empty() || a == T::zero() {
            return b.clone();
        }

        if b.is_empty() {
            return self.mul(a);
        }

        let items = outer_join(
            self.storage.indices(),
            self.storage.values(),
            b.storage.indices(),
            b.storage.values(),
            |_idx, join| match join {
                Join::Left(lhs) => *lhs * a,
                Join::Right(rhs) => *rhs,
                Join::Both(lhs, rhs) => (*lhs * a) + *rhs,
            },
        );

        Self::Output::from(S::from_canonical_pairs(items.filter(|(_, v)| !v.is_zero())))
    }
}
