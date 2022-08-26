use crate::{
    embedding::{
        helper::{VariableSizeEmbeddingDecoder, VariableSizeEmbeddingEncoder},
        FixedSize, FixedSizeEmbedding, VariableSize, VariableSizeEmbedding,
    },
    impl_fixed_type, impl_variable_type,
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

// ==================== floats ====================

#[cfg(any(test, feature = "ordered-float"))]
mod floats {
    use super::*;
    use ordered_float::OrderedFloat;

    macro_rules! impl_float {
        ($t:ty, $n:expr, $ti:ty, $tu:ty, $shift:expr) => {
            impl From<OrderedFloat<$t>> for FixedSizeEmbedding<OrderedFloat<$t>, $n> {
                fn from(val: OrderedFloat<$t>) -> Self {
                    // See https://github.com/rust-lang/rust/blob/9c20b2a8cc7588decb6de25ac6a7912dcef24d65/library/core/src/num/f32.rs#L1176-L1260
                    let mut i = val.to_bits() as $ti;
                    i ^= (((i >> $shift) as $tu) >> 1) as $ti;
                    let e: FixedSizeEmbedding<$ti, $n> = i.into();
                    Self::new(e.into_data())
                }
            }

            impl From<FixedSizeEmbedding<OrderedFloat<$t>, $n>> for OrderedFloat<$t> {
                fn from(embedding: FixedSizeEmbedding<OrderedFloat<$t>, $n>) -> Self {
                    // See https://github.com/rust-lang/rust/blob/9c20b2a8cc7588decb6de25ac6a7912dcef24d65/library/core/src/num/f32.rs#L1176-L1260
                    let mut i: $ti =
                        FixedSizeEmbedding::<$ti, $n>::new(embedding.into_data()).into();
                    i ^= (((i >> $shift) as $tu) >> 1) as $ti;
                    OrderedFloat(<$t>::from_bits(i as $tu))
                }
            }

            impl_fixed_type!(OrderedFloat<$t>, $n);
        };
    }

    impl_float!(f32, 4, i32, u32, 31);
    impl_float!(f64, 8, i64, u64, 63);
}

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

// ==================== String ====================

impl From<String> for VariableSizeEmbedding<String> {
    fn from(val: String) -> Self {
        VariableSizeEmbedding::new(val.into_bytes())
    }
}

impl From<VariableSizeEmbedding<String>> for String {
    fn from(embedding: VariableSizeEmbedding<String>) -> Self {
        Self::from_utf8(embedding.into_data()).unwrap()
    }
}

impl_variable_type!(String);

// ==================== Variable Tuples ====================

impl<T1, T2> From<(T1, T2)> for VariableSizeEmbedding<(T1, T2)>
where
    T1: VariableSize,
    T2: VariableSize,
{
    fn from(val: (T1, T2)) -> Self {
        let mut encoder = VariableSizeEmbeddingEncoder::default();
        encoder.push_variable(val.0.into());
        encoder.push_variable(val.1.into());
        encoder.finalize()
    }
}

impl<T1, T2> From<VariableSizeEmbedding<(T1, T2)>> for (T1, T2)
where
    T1: VariableSize,
    T2: VariableSize,
{
    fn from(embedding: VariableSizeEmbedding<(T1, T2)>) -> Self {
        let mut decoder = VariableSizeEmbeddingDecoder::new(embedding);
        let t1 = decoder.read_variable().into();
        let t2 = decoder.read_variable().into();
        decoder.finalize();
        (t1, t2)
    }
}

impl<T1, T2> VariableSize for (T1, T2)
where
    T1: VariableSize,
    T2: VariableSize,
{
}

#[cfg(test)]
mod tests {
    use crate::{
        test_fixed_embedding_roundtrip, test_fixed_embedding_sorting,
        test_variable_embedding_roundtrip, test_variable_embedding_sorting,
    };
    use ordered_float::OrderedFloat;

    test_fixed_embedding_roundtrip!(bool, test_bool_roundtrip);
    test_fixed_embedding_sorting!(bool, test_bool_sorting);

    test_fixed_embedding_roundtrip!(char, test_char_roundtrip);
    test_fixed_embedding_sorting!(char, test_char_sorting);

    test_fixed_embedding_roundtrip!(OrderedFloat<f32>, test_f32_roundtrip);
    test_fixed_embedding_sorting!(OrderedFloat<f32>, test_f32_sorting);

    test_fixed_embedding_roundtrip!(OrderedFloat<f64>, test_f64_roundtrip);
    test_fixed_embedding_sorting!(OrderedFloat<f64>, test_f64_sorting);

    test_fixed_embedding_roundtrip!(i8, test_i8_roundtrip);
    test_fixed_embedding_sorting!(i8, test_i8_sorting);

    test_fixed_embedding_roundtrip!(i16, test_i16_roundtrip);
    test_fixed_embedding_sorting!(i16, test_i16_sorting);

    test_fixed_embedding_roundtrip!(i32, test_i32_roundtrip);
    test_fixed_embedding_sorting!(i32, test_i32_sorting);

    test_fixed_embedding_roundtrip!(i64, test_i64_roundtrip);
    test_fixed_embedding_sorting!(i64, test_i64_sorting);

    test_fixed_embedding_roundtrip!(i128, test_i128_roundtrip);
    test_fixed_embedding_sorting!(i128, test_i128_sorting);

    test_fixed_embedding_roundtrip!(u8, test_u8_roundtrip);
    test_fixed_embedding_sorting!(u8, test_u8_sorting);

    test_fixed_embedding_roundtrip!(u16, test_u16_roundtrip);
    test_fixed_embedding_sorting!(u16, test_u16_sorting);

    test_fixed_embedding_roundtrip!(u32, test_u32_roundtrip);
    test_fixed_embedding_sorting!(u32, test_u32_sorting);

    test_fixed_embedding_roundtrip!(u64, test_u64_roundtrip);
    test_fixed_embedding_sorting!(u64, test_u64_sorting);

    test_fixed_embedding_roundtrip!(u128, test_u128_roundtrip);
    test_fixed_embedding_sorting!(u128, test_u128_sorting);

    test_variable_embedding_roundtrip!(String, test_string_roundtrip);
    test_variable_embedding_sorting!(String, test_string_sorting);

    test_variable_embedding_roundtrip!((String, String), test_tuple2_string_roundtrip);
    test_variable_embedding_sorting!((String, String), test_tuple2_string_sorting);
}
