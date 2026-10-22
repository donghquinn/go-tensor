// C interface to the Metal backend, implemented in mps.m.
#ifndef GO_TENSOR_MPS_H
#define GO_TENSOR_MPS_H

#include <stddef.h>

// Strided view; must match struct View in kernels.metal.
typedef struct {
    int ndim;
    int offset;
    int shape[8];
    int strides[8];
} mps_view;

// Convolution geometry; must match struct Geom in kernels.metal.
typedef struct {
    int N, C, H, W, KH, KW, OH, OW, stride, pad;
} mps_geom;

// Returns NULL on success, or an error message the caller must free.
char* mps_init(const char* kernel_source);
const char* mps_device_name(void);

// Buffers are retained MTLBuffer pointers holding float32 values.
void* mps_alloc(size_t bytes);                         // uninitialized
void* mps_alloc_copy(const void* src, size_t bytes);   // filled from host memory
void mps_zero(void* buf, size_t bytes);                // encoded in the command stream
void mps_release(void* buf);
void* mps_contents(void* buf);        // shared CPU-visible memory

// Returns NULL on success, or an error message the caller must free.
char* mps_sync(void);

// Command buffers are numbered in submission order. mps_pending_id is the
// number of the buffer currently being encoded; mps_completed_id is the
// highest number the GPU has finished.
unsigned long long mps_pending_id(void);
unsigned long long mps_completed_id(void);

void mps_unary(int op, float s, void* x, void* y, unsigned n);
void mps_binary(int op, void* a, const mps_view* va, void* b, const mps_view* vb, void* out, unsigned n);
void mps_copy(void* src, const mps_view* v, void* out, unsigned n);
void mps_reduce(int op, void* x, void* out, unsigned outer, unsigned n, unsigned inner);
void mps_matmul(void* a, void* b, void* c, int m, int k, int n, int trans_a, int trans_b);
void mps_im2col(void* x, void* cols, const mps_geom* g, unsigned n);
void mps_col2im(void* cols, void* x, const mps_geom* g, unsigned n);
void mps_maxpool_fwd(void* x, void* out, const mps_geom* g, unsigned n);
void mps_maxpool_bwd(void* x, void* grad, void* dx, const mps_geom* g, unsigned n);
void mps_write(void* dst, size_t dst_offset, void* src, size_t bytes);

#endif
