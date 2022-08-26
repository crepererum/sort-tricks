use std::{
    io::{Cursor, Read},
    marker::PhantomData,
};

use super::VariableSizeEmbedding;

#[derive(Debug)]
pub struct VariableSizeEmbeddingEncoder<T> {
    data: Vec<u8>,
    _type: PhantomData<T>,
}

impl<T> Default for VariableSizeEmbeddingEncoder<T> {
    fn default() -> Self {
        Self {
            data: Default::default(),
            _type: Default::default(),
        }
    }
}

impl<T> VariableSizeEmbeddingEncoder<T> {
    pub fn push_variable<S>(&mut self, data: VariableSizeEmbedding<S>) {
        let n_nulls = data.data().iter().filter(|x| **x == 0).count();
        self.data.reserve(data.data().len() + n_nulls + 1);
        for x in data.data() {
            if *x == 0 {
                self.data.push(0);
                self.data.push(1);
            } else {
                self.data.push(*x);
            }
        }
        self.data.push(0);
        self.data.push(0);
    }

    pub fn finalize(self) -> VariableSizeEmbedding<T> {
        VariableSizeEmbedding::new(self.data)
    }
}

#[derive(Debug, Default)]
pub struct VariableSizeEmbeddingDecoder<T> {
    data: Cursor<Vec<u8>>,
    _type: PhantomData<T>,
}

impl<T> VariableSizeEmbeddingDecoder<T> {
    pub fn new(data: VariableSizeEmbedding<T>) -> Self {
        Self {
            data: Cursor::new(data.into_data()),
            _type: PhantomData::default(),
        }
    }

    fn read_byte(&mut self) -> u8 {
        let mut buf = [0];
        self.data.read_exact(&mut buf).expect("data left");
        buf[0]
    }

    pub fn read_variable<S>(&mut self) -> VariableSizeEmbedding<S> {
        let mut data = vec![];

        loop {
            let next = self.read_byte();
            let decoded = if next == 0 {
                match self.read_byte() {
                    0 => {
                        break;
                    }
                    1 => 0,
                    _ => panic!("invalid encoding"),
                }
            } else {
                next
            };
            data.push(decoded);
        }

        VariableSizeEmbedding::new(data)
    }

    pub fn finalize(self) {
        let pos = self.data.position() as usize;
        assert_eq!(pos, self.data.into_inner().len());
    }
}
