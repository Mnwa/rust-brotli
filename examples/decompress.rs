extern crate simd_brotli;
#[cfg(not(feature = "std"))]
fn main() {
    panic!("For no-stdlib examples please see the tests")
}
#[cfg(feature = "std")]
fn main() {
    use std::io;
    let stdin = &mut io::stdin();
    {
        use std::io::{Read, Write};
        let mut reader = simd_brotli::Decompressor::new(
            stdin, 4096, // buffer size
        );
        let mut buf = [0u8; 4096];
        loop {
            match reader.read(&mut buf[..]) {
                Err(e) => {
                    if let io::ErrorKind::Interrupted = e.kind() {
                        continue;
                    }
                    panic!("{}", e);
                }
                Ok(size) => {
                    if size == 0 {
                        break;
                    }
                    if let Err(e) = io::stdout().write_all(&buf[..size]) {
                        panic!("{}", e)
                    }
                }
            }
        }
    }
}
