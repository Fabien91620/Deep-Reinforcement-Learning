use std::slice::from_raw_parts;

pub struct WrappedData {
    pub byte_count: usize,
    pub data: *mut u8,
}

pub fn free_wrapped_bytes(unsafe_wrapped_data: *mut WrappedData) {
    unsafe {
        let wrapped_data = Box::from_raw(unsafe_wrapped_data);
        Vec::from_raw_parts(wrapped_data.data, wrapped_data.byte_count, wrapped_data.byte_count);
    }
}

pub fn get_bytes<'a>(pointer_to_wrapped_data: *mut WrappedData) -> &'a [u8] {
    unsafe {
        let wrapped_data = pointer_to_wrapped_data.as_ref().unwrap();
        from_raw_parts(wrapped_data.data, wrapped_data.byte_count)
    }
}