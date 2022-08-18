use std::marker::PhantomData;

mod sealed {
    pub trait FixedSizeSealed {}

    impl<T, const N: usize> FixedSizeSealed for super::FixedSizeEmbedding<T, N> {}
}

pub trait FixedSize {
    type T: sealed::FixedSizeSealed;
}

#[macro_export]
macro_rules! impl_fixed_type {
    ($t:ty, $n:expr) => {
        impl FixedSize for $t {
            type T = FixedSizeEmbedding<$t, $n>;
        }
    };
}

/// Note: Technically `N` depends on `T` but this is currently NOT supported by Rust, see
/// <https://github.com/rust-lang/rust/issues/60551>.
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct FixedSizeEmbedding<T, const N: usize> {
    data: [u8; N],
    _type: PhantomData<T>,
}

impl<T, const N: usize> FixedSizeEmbedding<T, N> {
    pub fn new(data: [u8; N]) -> Self {
        Self {
            data,
            _type: PhantomData::default(),
        }
    }

    pub fn data(&self) -> &[u8; N] {
        &self.data
    }

    pub fn into_data(&self) -> [u8; N] {
        self.data
    }
}
