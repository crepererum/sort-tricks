use crate::{
    embedding::{FixedSize, FixedSizeEmbedding},
    impl_fixed_type,
};

// ==================== bool ====================

impl From<bool> for FixedSizeEmbedding<bool, 1> {
    fn from(val: bool) -> Self {
        let data = if val { [1] } else { [0] };
        Self::new(data)
    }
}

impl From<FixedSizeEmbedding<bool, 1>> for bool {
    fn from(embedding: FixedSizeEmbedding<bool, 1>) -> Self {
        embedding.data()[0] != 0
    }
}

impl_fixed_type!(bool, 1);

// ==================== char ====================

impl From<char> for FixedSizeEmbedding<char, 4> {
    fn from(val: char) -> Self {
        let u = val as u32;
        let e: FixedSizeEmbedding<u32, 4> = u.into();
        Self::new(e.into_data())
    }
}

impl From<FixedSizeEmbedding<char, 4>> for char {
    fn from(embedding: FixedSizeEmbedding<char, 4>) -> Self {
        let u: u32 = FixedSizeEmbedding::<u32, 4>::new(embedding.into_data()).into();
        char::from_u32(u).unwrap()
    }
}

impl_fixed_type!(char, 4);

// ==================== signed integers ====================

macro_rules! impl_signed_integer {
    ($t:ty, $n:expr) => {
        impl From<$t> for FixedSizeEmbedding<$t, $n> {
            fn from(val: $t) -> Self {
                let mut data = val.to_be_bytes();
                // flip first bit, see <https://en.wikipedia.org/wiki/Two%27s_complement>.
                data[0] ^= 0x80;
                Self::new(data)
            }
        }

        impl From<FixedSizeEmbedding<$t, $n>> for $t {
            fn from(embedding: FixedSizeEmbedding<$t, $n>) -> Self {
                let mut data = embedding.into_data();
                // flip first bit, see <https://en.wikipedia.org/wiki/Two%27s_complement>.
                data[0] ^= 0x80;
                <$t>::from_be_bytes(data)
            }
        }

        impl_fixed_type!($t, $n);
    };
}

impl_signed_integer!(i8, 1);
impl_signed_integer!(i16, 2);
impl_signed_integer!(i32, 4);
impl_signed_integer!(i64, 8);
impl_signed_integer!(i128, 16);

// ==================== unsigned integers ====================

macro_rules! impl_unsigned_integer {
    ($t:ty, $n:expr) => {
        impl From<$t> for FixedSizeEmbedding<$t, $n> {
            fn from(val: $t) -> Self {
                Self::new(val.to_be_bytes())
            }
        }

        impl From<FixedSizeEmbedding<$t, $n>> for $t {
            fn from(embedding: FixedSizeEmbedding<$t, $n>) -> Self {
                <$t>::from_be_bytes(embedding.into_data())
            }
        }

        impl_fixed_type!($t, $n);
    };
}

impl_unsigned_integer!(u8, 1);
impl_unsigned_integer!(u16, 2);
impl_unsigned_integer!(u32, 4);
impl_unsigned_integer!(u64, 8);
impl_unsigned_integer!(u128, 16);

#[cfg(test)]
mod tests {
    use crate::{test_embedding_roundtrip, test_embedding_sorting};

    test_embedding_roundtrip!(bool, test_bool_roundtrip);
    test_embedding_sorting!(bool, test_bool_sorting);

    test_embedding_roundtrip!(char, test_char_roundtrip);
    test_embedding_sorting!(char, test_char_sorting);

    test_embedding_roundtrip!(i8, test_i8_roundtrip);
    test_embedding_sorting!(i8, test_i8_sorting);

    test_embedding_roundtrip!(i16, test_i16_roundtrip);
    test_embedding_sorting!(i16, test_i16_sorting);

    test_embedding_roundtrip!(i32, test_i32_roundtrip);
    test_embedding_sorting!(i32, test_i32_sorting);

    test_embedding_roundtrip!(i64, test_i64_roundtrip);
    test_embedding_sorting!(i64, test_i64_sorting);

    test_embedding_roundtrip!(i128, test_i128_roundtrip);
    test_embedding_sorting!(i128, test_i128_sorting);

    test_embedding_roundtrip!(u8, test_u8_roundtrip);
    test_embedding_sorting!(u8, test_u8_sorting);

    test_embedding_roundtrip!(u16, test_u16_roundtrip);
    test_embedding_sorting!(u16, test_u16_sorting);

    test_embedding_roundtrip!(u32, test_u32_roundtrip);
    test_embedding_sorting!(u32, test_u32_sorting);

    test_embedding_roundtrip!(u64, test_u64_roundtrip);
    test_embedding_sorting!(u64, test_u64_sorting);

    test_embedding_roundtrip!(u128, test_u128_roundtrip);
    test_embedding_sorting!(u128, test_u128_sorting);
}