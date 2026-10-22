//go:build darwin && mps

// Objective-C bridge to Metal. Work is encoded into one open command buffer
// and committed in batches; mps_sync commits and waits. Command buffers retain
// the buffers they use, so Go may release a buffer while work is in flight.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <stdatomic.h>
#include <string.h>
#include "mps.h"

// Commit the open command buffer after this many operations so the GPU starts
// working while the CPU keeps encoding.
#define BATCH 64

static id<MTLDevice> dev;
static id<MTLCommandQueue> queue;
static id<MTLCommandBuffer> cmd;
static id<MTLCommandBuffer> last_committed;
static int pending;
static unsigned long long next_id = 1;     // number of the buffer being encoded
static _Atomic unsigned long long completed_id;
static id<MTLComputePipelineState> p_unary, p_binary, p_copy, p_reduce, p_im2col, p_col2im, p_pool_fwd, p_pool_bwd;

static char* copy_error(NSString* msg) { return strdup(msg.UTF8String); }

char* mps_init(const char* kernel_source) {
    @autoreleasepool {
        dev = MTLCreateSystemDefaultDevice();
        if (!dev) return copy_error(@"no Metal device found");
        if (!MPSSupportsMTLDevice(dev)) return copy_error(@"Metal Performance Shaders do not support this GPU");
        queue = [dev newCommandQueue];

        MTLCompileOptions* opts = [MTLCompileOptions new];
        if (@available(macOS 15.0, *)) {
            opts.mathMode = MTLMathModeSafe; // IEEE-conformant math, close to the CPU backend
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kernel_source] options:opts error:&err];
        if (!lib) return copy_error([NSString stringWithFormat:@"compiling kernels: %@", err.localizedDescription]);

        id<MTLComputePipelineState> __strong* slots[] = {&p_unary, &p_binary, &p_copy, &p_reduce, &p_im2col, &p_col2im, &p_pool_fwd, &p_pool_bwd};
        NSArray* names = @[@"unary_op", @"binary_op", @"copy_view", @"reduce_op", @"im2col", @"col2im", @"maxpool_fwd", @"maxpool_bwd"];
        for (NSUInteger i = 0; i < names.count; i++) {
            id<MTLFunction> fn = [lib newFunctionWithName:names[i]];
            *slots[i] = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!*slots[i]) return copy_error([NSString stringWithFormat:@"pipeline %@: %@", names[i], err.localizedDescription]);
        }
        return NULL;
    }
}

const char* mps_device_name(void) { return dev.name.UTF8String; }

static id<MTLCommandBuffer> current(void) {
    if (!cmd) cmd = [queue commandBuffer];
    return cmd;
}

static void commit(void) {
    if (!cmd) return;
    unsigned long long num = next_id++;
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        unsigned long long seen = atomic_load(&completed_id);
        while (seen < num && !atomic_compare_exchange_weak(&completed_id, &seen, num)) {
        }
    }];
    [cmd commit];
    last_committed = cmd;
    cmd = nil;
    pending = 0;
}

static void encoded(void) {
    if (++pending >= BATCH) commit();
}

char* mps_sync(void) {
    @autoreleasepool {
        commit();
        if (!last_committed) return NULL;
        [last_committed waitUntilCompleted];
        NSError* err = last_committed.error;
        last_committed = nil;
        if (err) return copy_error([NSString stringWithFormat:@"GPU command failed: %@", err.localizedDescription]);
        return NULL;
    }
}

unsigned long long mps_pending_id(void) { return next_id; }
unsigned long long mps_completed_id(void) { return atomic_load(&completed_id); }

void* mps_alloc(size_t bytes) {
    @autoreleasepool {
        id<MTLBuffer> buf = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        return buf ? (__bridge_retained void*)buf : NULL;
    }
}

void* mps_alloc_copy(const void* src, size_t bytes) {
    @autoreleasepool {
        id<MTLBuffer> buf = [dev newBufferWithBytes:src length:bytes options:MTLResourceStorageModeShared];
        return buf ? (__bridge_retained void*)buf : NULL;
    }
}

void mps_release(void* buf) {
    if (buf) CFRelease(buf);
}

void* mps_contents(void* buf) { return ((__bridge id<MTLBuffer>)buf).contents; }

#define BUF(p) ((__bridge id<MTLBuffer>)(p))

// dispatch encodes one compute pass of n threads.
static void dispatch(id<MTLComputePipelineState> p, NSUInteger n, void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (n == 0) return;
    id<MTLComputeCommandEncoder> enc = [current() computeCommandEncoder];
    [enc setComputePipelineState:p];
    bind(enc);
    NSUInteger tg = MIN(p.maxTotalThreadsPerThreadgroup, 256);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    encoded();
}

void mps_unary(int op, float s, void* x, void* y, unsigned n) {
    @autoreleasepool {
        dispatch(p_unary, n, ^(id<MTLComputeCommandEncoder> e) {
            [e setBuffer:BUF(x) offset:0 atIndex:0];
            [e setBuffer:BUF(y) offset:0 atIndex:1];
            [e setBytes:&op length:sizeof(op) atIndex:2];
            [e setBytes:&s length:sizeof(s) atIndex:3];
            [e setBytes:&n length:sizeof(n) atIndex:4];
        });
    }
}

void mps_binary(int op, void* a, const mps_view* va, void* b, const mps_view* vb, void* out, unsigned n) {
    @autoreleasepool {
        dispatch(p_binary, n, ^(id<MTLComputeCommandEncoder> e) {
            [e setBuffer:BUF(a) offset:0 atIndex:0];
            [e setBuffer:BUF(b) offset:0 atIndex:1];
            [e setBuffer:BUF(out) offset:0 atIndex:2];
            [e setBytes:va length:sizeof(mps_view) atIndex:3];
            [e setBytes:vb length:sizeof(mps_view) atIndex:4];
            [e setBytes:&op length:sizeof(op) atIndex:5];
            [e setBytes:&n length:sizeof(n) atIndex:6];
        });
    }
}

void mps_copy(void* src, const mps_view* v, void* out, unsigned n) {
    @autoreleasepool {
        dispatch(p_copy, n, ^(id<MTLComputeCommandEncoder> e) {
            [e setBuffer:BUF(src) offset:0 atIndex:0];
            [e setBuffer:BUF(out) offset:0 atIndex:1];
            [e setBytes:v length:sizeof(mps_view) atIndex:2];
            [e setBytes:&n length:sizeof(n) atIndex:3];
        });
    }
}

void mps_reduce(int op, void* x, void* out, unsigned outer, unsigned n, unsigned inner) {
    @autoreleasepool {
        NSUInteger groups = (NSUInteger)outer * inner;
        if (groups == 0) return;
        NSUInteger tg = 1;
        while (tg < n && tg < 256) tg <<= 1; // power of two for the tree reduction
        id<MTLComputeCommandEncoder> enc = [current() computeCommandEncoder];
        [enc setComputePipelineState:p_reduce];
        [enc setBuffer:BUF(x) offset:0 atIndex:0];
        [enc setBuffer:BUF(out) offset:0 atIndex:1];
        [enc setBytes:&op length:sizeof(op) atIndex:2];
        unsigned dims[4] = {outer, n, inner, 0}; // uint3 occupies 16 bytes
        [enc setBytes:dims length:sizeof(dims) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
        encoded();
    }
}

void mps_matmul(void* a, void* b, void* c, int m, int k, int n, int trans_a, int trans_b) {
    @autoreleasepool {
        if (m == 0 || n == 0) return;
        int a_rows = trans_a ? k : m, a_cols = trans_a ? m : k;
        int b_rows = trans_b ? n : k, b_cols = trans_b ? k : n;
        MPSMatrixDescriptor* da = [MPSMatrixDescriptor matrixDescriptorWithRows:a_rows columns:a_cols rowBytes:a_cols * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* db = [MPSMatrixDescriptor matrixDescriptorWithRows:b_rows columns:b_cols rowBytes:b_cols * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* dc = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:n * sizeof(float) dataType:MPSDataTypeFloat32];
        MPSMatrix* ma = [[MPSMatrix alloc] initWithBuffer:BUF(a) descriptor:da];
        MPSMatrix* mb = [[MPSMatrix alloc] initWithBuffer:BUF(b) descriptor:db];
        MPSMatrix* mc = [[MPSMatrix alloc] initWithBuffer:BUF(c) descriptor:dc];
        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev
                                                                        transposeLeft:trans_a != 0
                                                                       transposeRight:trans_b != 0
                                                                           resultRows:m
                                                                        resultColumns:n
                                                                      interiorColumns:k
                                                                                alpha:1.0
                                                                                 beta:0.0];
        [mm encodeToCommandBuffer:current() leftMatrix:ma rightMatrix:mb resultMatrix:mc];
        encoded();
    }
}

static void geom_pass(id<MTLComputePipelineState> p, void* in, void* out, const mps_geom* g, unsigned n) {
    dispatch(p, n, ^(id<MTLComputeCommandEncoder> e) {
        [e setBuffer:BUF(in) offset:0 atIndex:0];
        [e setBuffer:BUF(out) offset:0 atIndex:1];
        [e setBytes:g length:sizeof(mps_geom) atIndex:2];
        [e setBytes:&n length:sizeof(n) atIndex:3];
    });
}

void mps_im2col(void* x, void* cols, const mps_geom* g, unsigned n) {
    @autoreleasepool { geom_pass(p_im2col, x, cols, g, n); }
}

void mps_col2im(void* cols, void* x, const mps_geom* g, unsigned n) {
    @autoreleasepool { geom_pass(p_col2im, cols, x, g, n); }
}

void mps_maxpool_fwd(void* x, void* out, const mps_geom* g, unsigned n) {
    @autoreleasepool { geom_pass(p_pool_fwd, x, out, g, n); }
}

void mps_maxpool_bwd(void* x, void* grad, void* dx, const mps_geom* g, unsigned n) {
    @autoreleasepool {
        dispatch(p_pool_bwd, n, ^(id<MTLComputeCommandEncoder> e) {
            [e setBuffer:BUF(x) offset:0 atIndex:0];
            [e setBuffer:BUF(grad) offset:0 atIndex:1];
            [e setBuffer:BUF(dx) offset:0 atIndex:2];
            [e setBytes:g length:sizeof(mps_geom) atIndex:3];
            [e setBytes:&n length:sizeof(n) atIndex:4];
        });
    }
}

void mps_zero(void* buf, size_t bytes) {
    @autoreleasepool {
        if (bytes == 0) return;
        id<MTLBlitCommandEncoder> blit = [current() blitCommandEncoder];
        [blit fillBuffer:BUF(buf) range:NSMakeRange(0, bytes) value:0];
        [blit endEncoding];
        encoded();
    }
}

void mps_write(void* dst, size_t dst_offset, void* src, size_t bytes) {
    @autoreleasepool {
        if (bytes == 0) return;
        id<MTLBlitCommandEncoder> blit = [current() blitCommandEncoder];
        [blit copyFromBuffer:BUF(src) sourceOffset:0 toBuffer:BUF(dst) destinationOffset:dst_offset size:bytes];
        [blit endEncoding];
        encoded();
    }
}
