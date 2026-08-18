use crate::alloc::{Allocator, SliceWrapperMut};
#[cfg(feature = "std")]
use std::io;
#[cfg(feature = "std")]
use std::io::{Error, ErrorKind, Write};

#[cfg(feature = "std")]
pub use alloc_stdlib::StandardAlloc;
use brotli_decompressor::CustomWrite;
#[cfg(feature = "std")]
pub use brotli_decompressor::{IntoIoWriter, IoWriterWrapper};

use super::backward_references::BrotliEncoderParams;
use super::combined_alloc::BrotliAlloc;
use super::encode::{
    BrotliEncoderDestroyInstance, BrotliEncoderOperation, BrotliEncoderParameter,
    BrotliEncoderStateStruct,
};
use super::interface;
use crate::enc::combined_alloc::allocate;

#[cfg(feature = "std")]
pub struct CompressorWriterCustomAlloc<
    W: Write,
    BufferType: SliceWrapperMut<u8>,
    Alloc: BrotliAlloc,
>(CompressorWriterCustomIo<io::Error, IntoIoWriter<W>, BufferType, Alloc>);

#[cfg(feature = "std")]
impl<W: Write, BufferType: SliceWrapperMut<u8>, Alloc: BrotliAlloc>
    CompressorWriterCustomAlloc<W, BufferType, Alloc>
{
    pub fn new(w: W, buffer: BufferType, alloc: Alloc, q: u32, lgwin: u32) -> Self {
        CompressorWriterCustomAlloc::<W, BufferType, Alloc>(CompressorWriterCustomIo::<
            Error,
            IntoIoWriter<W>,
            BufferType,
            Alloc,
        >::new(
            IntoIoWriter::<W>(w),
            buffer,
            alloc,
            Error::new(ErrorKind::InvalidData, "Invalid Data"),
            Error::new(ErrorKind::WriteZero, "No room in output."),
            q,
            lgwin,
        ))
    }

    pub fn get_ref(&self) -> &W {
        &self.0.get_ref().0
    }
    pub fn get_mut(&mut self) -> &mut W {
        &mut self.0.get_mut().0
    }
    pub fn into_inner(self) -> W {
        self.0.into_inner().0
    }
}

#[cfg(feature = "std")]
impl<W: Write, BufferType: SliceWrapperMut<u8>, Alloc: BrotliAlloc> Write
    for CompressorWriterCustomAlloc<W, BufferType, Alloc>
{
    fn write(&mut self, buf: &[u8]) -> Result<usize, Error> {
        self.0.write(buf)
    }
    fn flush(&mut self) -> Result<(), Error> {
        self.0.flush()
    }
}

#[cfg(feature = "std")]
/// A [`Write`] adapter that Brotli-compresses bytes into an inner writer.
///
/// Call [`CompressorWriter::into_inner`] to finish the Brotli stream and
/// recover the writer after the last input has been written.
pub struct CompressorWriter<W: Write>(
    CompressorWriterCustomAlloc<
        W,
        <StandardAlloc as Allocator<u8>>::AllocatedMemory,
        StandardAlloc,
    >,
);

#[cfg(feature = "std")]
impl<W: Write> CompressorWriter<W> {
    /// Creates a compressing writer.
    ///
    /// `q` is the compression quality from 0 through 11 and `lgwin` is the
    /// base-2 logarithm of the sliding window size. Passing zero for
    /// `buffer_size` selects the default size of 4096 bytes.
    ///
    /// # Example
    ///
    /// ```
    /// use simd_brotli::CompressorWriter;
    /// use std::io::Write;
    ///
    /// let mut writer = CompressorWriter::new(Vec::new(), 4096, 9, 22);
    /// writer.write_all(b"some input").unwrap();
    /// let compressed = writer.into_inner();
    /// assert!(!compressed.is_empty());
    /// ```
    pub fn new(w: W, buffer_size: usize, q: u32, lgwin: u32) -> Self {
        let mut alloc = StandardAlloc::default();
        let buffer = allocate::<u8, _>(
            &mut alloc,
            if buffer_size == 0 { 4096 } else { buffer_size },
        );
        CompressorWriter::<W>(CompressorWriterCustomAlloc::new(w, buffer, alloc, q, lgwin))
    }

    /// Creates a compressing writer with the full encoder parameter set.
    ///
    /// # Example
    ///
    /// ```
    /// use simd_brotli::CompressorWriter;
    /// use simd_brotli::enc::BrotliEncoderParams;
    /// use std::io::Write;
    ///
    /// let mut params = BrotliEncoderParams::default();
    /// params.quality = 5;
    /// params.lgwin = 20;
    /// let mut writer = CompressorWriter::with_params(Vec::new(), 4096, &params);
    /// writer.write_all(b"parameterized input").unwrap();
    /// let compressed = writer.into_inner();
    /// assert!(!compressed.is_empty());
    /// ```
    pub fn with_params(w: W, buffer_size: usize, params: &BrotliEncoderParams) -> Self {
        let mut writer = Self::new(w, buffer_size, params.quality as u32, params.lgwin as u32);
        (writer.0).0.state.params = params.clone();
        writer
    }

    /// Returns a shared reference to the inner writer.
    ///
    /// The inner writer may not yet contain the final bytes of the Brotli
    /// stream. Use [`CompressorWriter::into_inner`] to finish it.
    ///
    /// # Example
    ///
    /// ```
    /// use simd_brotli::CompressorWriter;
    ///
    /// let writer = CompressorWriter::new(Vec::new(), 4096, 9, 22);
    /// assert!(writer.get_ref().is_empty());
    /// ```
    pub fn get_ref(&self) -> &W {
        self.0.get_ref()
    }

    /// Returns a mutable reference to the inner writer.
    ///
    /// Writing unrelated bytes through this reference would corrupt the
    /// compressed stream. It can safely be used for operations that do not
    /// change the writer's logical contents, such as reserving capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use simd_brotli::CompressorWriter;
    ///
    /// let mut writer = CompressorWriter::new(Vec::new(), 4096, 9, 22);
    /// writer.get_mut().reserve(1024);
    /// assert!(writer.get_ref().capacity() >= 1024);
    /// ```
    pub fn get_mut(&mut self) -> &mut W {
        self.0.get_mut()
    }

    /// Finishes the Brotli stream and returns the inner writer.
    ///
    /// # Example
    ///
    /// ```
    /// use simd_brotli::{CompressorWriter, Decompressor};
    /// use std::io::{Read, Write};
    ///
    /// let mut writer = CompressorWriter::new(Vec::new(), 4096, 9, 22);
    /// writer.write_all(b"round trip").unwrap();
    /// let compressed = writer.into_inner();
    ///
    /// let mut reader = Decompressor::new(compressed.as_slice(), 4096);
    /// let mut decoded = Vec::new();
    /// reader.read_to_end(&mut decoded).unwrap();
    /// assert_eq!(decoded, b"round trip");
    /// ```
    pub fn into_inner(self) -> W {
        self.0.into_inner()
    }
}

#[cfg(feature = "std")]
impl<W: Write> Write for CompressorWriter<W> {
    fn write(&mut self, buf: &[u8]) -> Result<usize, Error> {
        self.0.write(buf)
    }
    fn flush(&mut self) -> Result<(), Error> {
        self.0.flush()
    }
}

pub struct CompressorWriterCustomIo<
    ErrType,
    W: CustomWrite<ErrType>,
    BufferType: SliceWrapperMut<u8>,
    Alloc: BrotliAlloc,
> {
    output_buffer: BufferType,
    total_out: Option<usize>,
    output: Option<W>,
    error_if_invalid_data: Option<ErrType>,
    state: BrotliEncoderStateStruct<Alloc>,
    error_if_zero_bytes_written: Option<ErrType>,
}
pub fn write_all<ErrType, W: CustomWrite<ErrType>, ErrMaker: FnMut() -> Option<ErrType>>(
    writer: &mut W,
    mut buf: &[u8],
    mut error_to_return_if_zero_bytes_written: ErrMaker,
) -> Result<(), ErrType> {
    while !buf.is_empty() {
        match writer.write(buf) {
            Ok(bytes_written) => {
                if bytes_written != 0 {
                    buf = &buf[bytes_written..]
                } else {
                    match error_to_return_if_zero_bytes_written() {
                        Some(err) => {
                            return Err(err);
                        }
                        _ => {
                            return Ok(());
                        }
                    }
                }
            }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}
impl<ErrType, W: CustomWrite<ErrType>, BufferType: SliceWrapperMut<u8>, Alloc: BrotliAlloc>
    CompressorWriterCustomIo<ErrType, W, BufferType, Alloc>
{
    pub fn new(
        w: W,
        buffer: BufferType,
        alloc: Alloc,
        invalid_data_error_type: ErrType,
        error_if_zero_bytes_written: ErrType,
        q: u32,
        lgwin: u32,
    ) -> Self {
        let mut ret = CompressorWriterCustomIo {
            output_buffer: buffer,
            total_out: Some(0),
            output: Some(w),
            state: BrotliEncoderStateStruct::new(alloc),
            error_if_invalid_data: Some(invalid_data_error_type),
            error_if_zero_bytes_written: Some(error_if_zero_bytes_written),
        };
        ret.state
            .set_parameter(BrotliEncoderParameter::BROTLI_PARAM_QUALITY, q);
        ret.state
            .set_parameter(BrotliEncoderParameter::BROTLI_PARAM_LGWIN, lgwin);

        ret
    }
    fn flush_or_close(&mut self, op: BrotliEncoderOperation) -> Result<(), ErrType> {
        let mut nop_callback =
            |_data: &mut interface::PredictionModeContextMap<interface::InputReferenceMut>,
             _cmds: &mut [interface::StaticCommand],
             _mb: interface::InputPair,
             _mfv: &mut Alloc| ();

        loop {
            let mut avail_in: usize = 0;
            let mut input_offset: usize = 0;
            let mut avail_out: usize = self.output_buffer.slice_mut().len();
            let mut output_offset: usize = 0;
            let ret = self.state.compress_stream(
                op,
                &mut avail_in,
                &[],
                &mut input_offset,
                &mut avail_out,
                self.output_buffer.slice_mut(),
                &mut output_offset,
                &mut self.total_out,
                &mut nop_callback,
            );
            if output_offset > 0 {
                let zero_err = &mut self.error_if_zero_bytes_written;
                let fallback = &mut self.error_if_invalid_data;
                match write_all(
                    self.output.as_mut().unwrap(),
                    &self.output_buffer.slice_mut()[..output_offset],
                    || {
                        if let Some(err) = zero_err.take() {
                            return Some(err);
                        }
                        fallback.take()
                    },
                ) {
                    Ok(_) => {}
                    Err(e) => return Err(e),
                }
            }
            if !ret {
                return Err(self.error_if_invalid_data.take().unwrap());
            }
            if let BrotliEncoderOperation::BROTLI_OPERATION_FLUSH = op {
                if self.state.has_more_output() {
                    continue;
                }
                return Ok(());
            }
            if self.state.is_finished() {
                return Ok(());
            }
        }
    }

    pub fn get_ref(&self) -> &W {
        self.output.as_ref().unwrap()
    }
    pub fn get_mut(&mut self) -> &mut W {
        self.output.as_mut().unwrap()
    }
    pub fn into_inner(mut self) -> W {
        match self.flush_or_close(BrotliEncoderOperation::BROTLI_OPERATION_FINISH) {
            Ok(_) => {}
            Err(_) => {}
        }
        self.output.take().unwrap()
    }
}

impl<ErrType, W: CustomWrite<ErrType>, BufferType: SliceWrapperMut<u8>, Alloc: BrotliAlloc> Drop
    for CompressorWriterCustomIo<ErrType, W, BufferType, Alloc>
{
    fn drop(&mut self) {
        if self.output.is_some() {
            match self.flush_or_close(BrotliEncoderOperation::BROTLI_OPERATION_FINISH) {
                Ok(_) => {}
                Err(_) => {}
            }
        }
        BrotliEncoderDestroyInstance(&mut self.state);
    }
}
impl<ErrType, W: CustomWrite<ErrType>, BufferType: SliceWrapperMut<u8>, Alloc: BrotliAlloc>
    CustomWrite<ErrType> for CompressorWriterCustomIo<ErrType, W, BufferType, Alloc>
{
    fn write(&mut self, buf: &[u8]) -> Result<usize, ErrType> {
        let mut nop_callback =
            |_data: &mut interface::PredictionModeContextMap<interface::InputReferenceMut>,
             _cmds: &mut [interface::StaticCommand],
             _mb: interface::InputPair,
             _mfv: &mut Alloc| ();
        let mut avail_in = buf.len();
        let mut input_offset: usize = 0;
        while avail_in != 0 {
            let mut output_offset = 0;
            let mut avail_out = self.output_buffer.slice_mut().len();
            let ret = self.state.compress_stream(
                BrotliEncoderOperation::BROTLI_OPERATION_PROCESS,
                &mut avail_in,
                buf,
                &mut input_offset,
                &mut avail_out,
                self.output_buffer.slice_mut(),
                &mut output_offset,
                &mut self.total_out,
                &mut nop_callback,
            );
            if output_offset > 0 {
                let zero_err = &mut self.error_if_zero_bytes_written;
                let fallback = &mut self.error_if_invalid_data;
                match write_all(
                    self.output.as_mut().unwrap(),
                    &self.output_buffer.slice_mut()[..output_offset],
                    || {
                        if let Some(err) = zero_err.take() {
                            return Some(err);
                        }
                        fallback.take()
                    },
                ) {
                    Ok(_) => {}
                    Err(e) => return Err(e),
                }
            }
            if !ret {
                return Err(self.error_if_invalid_data.take().unwrap());
            }
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> Result<(), ErrType> {
        match self.flush_or_close(BrotliEncoderOperation::BROTLI_OPERATION_FLUSH) {
            Ok(_) => {}
            Err(e) => return Err(e),
        }
        self.output.as_mut().unwrap().flush()
    }
}
