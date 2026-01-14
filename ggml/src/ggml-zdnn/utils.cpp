#include "ggml.h"
#include "utils.hpp"

// Check if a type requires dequantization for MUL_MAT
bool ggml_zdnn_type_needs_dequant(ggml_type type) {
    switch (type) {
        // K-quants (most common in GGUF models)
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        // Basic quants
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        // IQ quants
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ1_M:
            return true;
        default:
            return false;
    }
}

// Dequantize data from quantized format to F32
void ggml_zdnn_dequantize(const void * src, float * dst, ggml_type type, int64_t nelements) {
    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    GGML_ASSERT(traits != nullptr && "Unknown type");
    GGML_ASSERT(traits->to_float != nullptr && "Type has no dequantization function");
    traits->to_float(src, dst, nelements);
}

// Check if a type can be directly mapped to a ZDNN type (without dequantization)
// Only floating-point types can be transformed and used with NNPA
// Integer types (I8, I32) are used for indices but cannot be transformed
// Q8_0 is a quantized type that should be dequantized, not treated as native INT8
bool ggml_zdnn_type_is_native(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

zdnn_data_types ggml_zdnn_type_mapping(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return FP32;
        case GGML_TYPE_F16:
            return FP16;
        case GGML_TYPE_BF16:
            return BFLOAT;
        case GGML_TYPE_Q8_0:
            return INT8;
        case GGML_TYPE_I8:
            return INT8;
        case GGML_TYPE_I32:
            return INT32;
        default:
            GGML_ABORT("%s: fatal: unable to determine zTensor data type for type %d",
                       __func__, type);
            break;
    }
}

void ggml_zdnn_create_tensor(zdnn_tensor_desc  & pre_tfm_desc,
                             zdnn_tensor_desc  & tfm_desc,
                             zdnn_ztensor      & ztensor,
                       const ggml_tensor       * src,
                       const int64_t           * ne,
                       const zdnn_data_layouts   layout) {
    zdnn_init_pre_transformed_desc(
        layout,
        ggml_zdnn_type_mapping(src->type),
        &pre_tfm_desc,
        ne[3], ne[2], ne[1], ne[0]
    );

    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_tfm_desc, &tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_tfm_desc, &tfm_desc, &ztensor));
}

void ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, void * buffer) {
    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor, buffer));
}

void ggml_zdnn_init_tensor(ggml_backend_zdnn_buffer * buffer, const ggml_tensor * tensor) {
    // Initialize dequantization fields
    buffer->dequant_data = nullptr;
    buffer->original_type = tensor->type;

    // Determine zdnn type and whether we need dequantization
    zdnn_data_types zdnn_type;
    bool needs_dequant = false;
    bool skip_ztensor = false;

    if (ggml_zdnn_type_is_native(tensor->type)) {
        // Native type - use directly
        zdnn_type = ggml_zdnn_type_mapping(tensor->type);
    } else if (ggml_zdnn_type_needs_dequant(tensor->type)) {
        // Quantized type we can dequantize - will convert to F32
        zdnn_type = FP32;
        needs_dequant = true;
    } else {
        // Unsupported type - skip ztensor creation
        // This tensor will fall back to CPU
        skip_ztensor = true;
    }

    if (skip_ztensor) {
        // Mark as not having a ztensor
        buffer->ztensor.buffer_size = 0;
        buffer->ztensor.buffer = nullptr;
        return;
    }

    // Allocate dequantization buffer if needed
    if (needs_dequant) {
        const int64_t nelements = ggml_nelements(tensor);
        buffer->dequant_data = ggml_aligned_malloc(nelements * sizeof(float));
        GGML_ASSERT(buffer->dequant_data != nullptr);
    }

    // Create the tensor descriptor based on tensor shape
    if (ggml_is_matrix(tensor)) {
        // 2D tensor (typical for weights)
        zdnn_init_pre_transformed_desc(
            ZDNN_2D,
            zdnn_type,
            &buffer->pre_tfm_desc,
            tensor->ne[1], tensor->ne[0]
        );
    } else {
        // 4D tensor - use NHWC layout
        zdnn_init_pre_transformed_desc(
            ZDNN_NHWC,
            zdnn_type,
            &buffer->pre_tfm_desc,
            tensor->ne[3], tensor->ne[2], tensor->ne[1], tensor->ne[0]
        );
    }

    ZDNN_CHECK(zdnn_generate_transformed_desc(&buffer->pre_tfm_desc, &buffer->tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&buffer->pre_tfm_desc, &buffer->tfm_desc, &buffer->ztensor));

    // Initialize validity tracking - data is current (not yet stickified)
    buffer->valid_repr = ZDNN_REPR_DATA_CURRENT;
}

// ============================================================================
// Lazy Unstickification Implementation
// ============================================================================

// Disable lazy optimization to test if it's causing issues
#define DISABLE_LAZY_UNSTICKIFY 1

void ggml_zdnn_ensure_stickified(ggml_backend_zdnn_buffer * buffer, const ggml_tensor * tensor) {
    if (!buffer || !tensor) return;

#if DISABLE_LAZY_UNSTICKIFY
    // When lazy optimization is disabled, always re-sync from float data
    // First ensure float data is valid (unstickify if needed)
    if (buffer->valid_repr == ZDNN_REPR_ZTENSOR_CURRENT) {
        // ztensor is current but float data is stale - unstickify first
        if (buffer->ztensor.is_transformed) {
            ZDNN_CHECK(zdnn_transform_origtensor(&buffer->ztensor, const_cast<ggml_tensor*>(tensor)->data));
            buffer->valid_repr = ZDNN_REPR_BOTH_CURRENT;
        }
    }
#else
    // Check if ztensor is already valid (lazy optimization enabled)
    if (buffer->valid_repr == ZDNN_REPR_ZTENSOR_CURRENT ||
        buffer->valid_repr == ZDNN_REPR_BOTH_CURRENT) {
        // ztensor is already valid, nothing to do
        return;
    }
#endif

    // Float data is current (or both are stale) - need to stickify
    // Reset ztensor if it was previously transformed
    if (buffer->ztensor.is_transformed) {
        zdnn_reset_ztensor(&buffer->ztensor);
    }

    // Determine the data source
    void * src_data = const_cast<ggml_tensor*>(tensor)->data;
    if (buffer->dequant_data) {
        // Use dequantized data for quantized types
        src_data = buffer->dequant_data;
    }

    // Transform to stickified format
    ZDNN_CHECK(zdnn_transform_ztensor(&buffer->ztensor, src_data));

    // Now both are current (we just synced ztensor from float data)
    buffer->valid_repr = ZDNN_REPR_BOTH_CURRENT;
}

// Debug counter for ensure_float_data
static int ensure_float_count = 0;

void ggml_zdnn_ensure_float_data(ggml_backend_zdnn_buffer * buffer, ggml_tensor * tensor) {
    if (!buffer || !tensor) return;

    ensure_float_count++;
    if (ensure_float_count <= 5 || ensure_float_count % 1000 == 0) {
        GGML_LOG_INFO("ensure_float_data #%d: %s valid_repr=%d is_transformed=%d\n",
                      ensure_float_count, tensor->name, buffer->valid_repr, buffer->ztensor.is_transformed);
    }

    // Check if float data is already valid
    if (buffer->valid_repr == ZDNN_REPR_DATA_CURRENT ||
        buffer->valid_repr == ZDNN_REPR_BOTH_CURRENT) {
        // Float data is already valid, nothing to do
        return;
    }

    // ztensor is current - need to unstickify
    if (!buffer->ztensor.is_transformed) {
        // ztensor hasn't been transformed yet - this shouldn't happen
        // if valid_repr is ZTENSOR_CURRENT, but handle gracefully
        GGML_LOG_WARN("ggml_zdnn_ensure_float_data: ztensor not transformed but marked current\n");
        buffer->valid_repr = ZDNN_REPR_NONE;
        return;
    }

    // Actually unstickifying
    GGML_LOG_INFO("ensure_float_data: UNSTICKIFYING %s\n", tensor->name);

    // Unstickify to float data
    ZDNN_CHECK(zdnn_transform_origtensor(&buffer->ztensor, tensor->data));

    // Now both are current (we just synced float data from ztensor)
    buffer->valid_repr = ZDNN_REPR_BOTH_CURRENT;
}

void ggml_zdnn_mark_ztensor_current(ggml_backend_zdnn_buffer * buffer) {
    if (buffer) {
        buffer->valid_repr = ZDNN_REPR_ZTENSOR_CURRENT;
    }
}

void ggml_zdnn_sync_output(ggml_backend_zdnn_buffer * buffer, ggml_tensor * tensor) {
    if (!buffer || !tensor) return;

    // Mark ztensor as current since we just computed it
    buffer->valid_repr = ZDNN_REPR_ZTENSOR_CURRENT;

    // Immediately unstickify to ensure float data is valid for CPU fallback
    if (buffer->ztensor.is_transformed) {
        ZDNN_CHECK(zdnn_transform_origtensor(&buffer->ztensor, tensor->data));
        buffer->valid_repr = ZDNN_REPR_BOTH_CURRENT;
    }
}

void ggml_zdnn_mark_float_current(ggml_backend_zdnn_buffer * buffer) {
    if (buffer) {
        buffer->valid_repr = ZDNN_REPR_DATA_CURRENT;
    }
}

void ggml_zdnn_mark_both_current(ggml_backend_zdnn_buffer * buffer) {
    if (buffer) {
        buffer->valid_repr = ZDNN_REPR_BOTH_CURRENT;
    }
}

bool ggml_zdnn_is_ztensor_valid(const ggml_backend_zdnn_buffer * buffer) {
    if (!buffer) return false;
    return buffer->valid_repr == ZDNN_REPR_ZTENSOR_CURRENT ||
           buffer->valid_repr == ZDNN_REPR_BOTH_CURRENT;
}

bool ggml_zdnn_is_float_valid(const ggml_backend_zdnn_buffer * buffer) {
    if (!buffer) return false;
    return buffer->valid_repr == ZDNN_REPR_DATA_CURRENT ||
           buffer->valid_repr == ZDNN_REPR_BOTH_CURRENT;
}
