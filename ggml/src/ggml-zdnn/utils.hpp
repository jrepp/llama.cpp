#ifndef GGML_ZDNN_UTILITIES_HPP
#define GGML_ZDNN_UTILITIES_HPP

#include "common.hpp"

zdnn_data_types ggml_zdnn_type_mapping(ggml_type type);

// Check if a type can be directly mapped to ZDNN (without dequantization)
bool ggml_zdnn_type_is_native(ggml_type type);

// Check if a type requires dequantization (is a quantized type we support)
bool ggml_zdnn_type_needs_dequant(ggml_type type);

// Dequantize data from quantized format to F32
void ggml_zdnn_dequantize(const void * src, float * dst, ggml_type type, int64_t nelements);

void ggml_zdnn_create_tensor(zdnn_tensor_desc & pre_tfm_desc,
                             zdnn_tensor_desc & tfm_desc,
                             zdnn_ztensor     & ztensor,
                      const ggml_tensor       * src,
                      const int64_t           * ne,
                      const zdnn_data_layouts   layout);

void ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, void * buffer);

void ggml_zdnn_init_tensor(ggml_backend_zdnn_buffer * buffer, const ggml_tensor * tensor);

// ============================================================================
// Lazy Unstickification Helpers
// ============================================================================
// These functions manage the validity tracking for tensor representations,
// allowing tensors to stay in stickified (NNPA) format between operations.

/**
 * Ensure the ztensor contains valid stickified data.
 * If the float data is more current, this will transform it to stickified format.
 * Call this before any NNPA operation that reads from the tensor.
 *
 * @param buffer  The zdnn buffer to ensure is stickified
 * @param tensor  The ggml tensor (for accessing float data)
 */
void ggml_zdnn_ensure_stickified(ggml_backend_zdnn_buffer * buffer, const ggml_tensor * tensor);

/**
 * Ensure the float data contains valid values.
 * If the ztensor is more current, this will unstickify it.
 * Call this before any CPU operation that reads the float data.
 *
 * @param buffer  The zdnn buffer to ensure has valid float data
 * @param tensor  The ggml tensor (for accessing float data destination)
 */
void ggml_zdnn_ensure_float_data(ggml_backend_zdnn_buffer * buffer, ggml_tensor * tensor);

/**
 * Mark the ztensor as the current representation (float data may be stale).
 * Call this after an NNPA operation writes to the ztensor.
 *
 * @param buffer  The zdnn buffer to mark
 */
void ggml_zdnn_mark_ztensor_current(ggml_backend_zdnn_buffer * buffer);

/**
 * Mark the float data as the current representation (ztensor may be stale).
 * Call this after a CPU operation writes to the float data.
 *
 * @param buffer  The zdnn buffer to mark
 */
void ggml_zdnn_mark_float_current(ggml_backend_zdnn_buffer * buffer);

/**
 * Mark both representations as current (synchronized).
 * Call this after explicitly synchronizing both representations.
 *
 * @param buffer  The zdnn buffer to mark
 */
void ggml_zdnn_mark_both_current(ggml_backend_zdnn_buffer * buffer);

/**
 * Check if the ztensor representation is valid (current or both current).
 *
 * @param buffer  The zdnn buffer to check
 * @return true if ztensor is valid, false otherwise
 */
bool ggml_zdnn_is_ztensor_valid(const ggml_backend_zdnn_buffer * buffer);

/**
 * Check if the float data representation is valid (current or both current).
 *
 * @param buffer  The zdnn buffer to check
 * @return true if float data is valid, false otherwise
 */
bool ggml_zdnn_is_float_valid(const ggml_backend_zdnn_buffer * buffer);

#endif  // GGML_ZDNN_UTILITIES_HPP
