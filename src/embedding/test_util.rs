#[macro_export]
macro_rules! test_embedding_roundtrip {
    ($t:ty, $name:ident) => {
        #[allow(unused_imports)]
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn $name(orig: $t) {
                #[allow(unused_imports)]
                use $crate::embedding::*;

                let embedding: <$t as FixedSize>::T = orig.clone().into();
                let recovered: $t = embedding.into();
                assert_eq!(orig, recovered);
            }
        }
    };
}

#[macro_export]
macro_rules! test_embedding_sorting {
    ($t:ty, $name:ident) => {
        #[allow(unused_imports)]
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn $name(input: Vec<$t>) {
                #[allow(unused_imports)]
                use $crate::embedding::*;

                let mut embedding: Vec<<$t as FixedSize>::T> = input.iter().cloned().map(|x| x.into()).collect();
                embedding.sort();
                let actual: Vec<$t> = embedding.into_iter().map(|x| x.into()).collect();

                let mut expected = input;
                expected.sort();

                assert_eq!(actual, expected);
            }
        }
    };
}
