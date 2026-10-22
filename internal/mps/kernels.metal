// Metal compute kernels for the MPS backend. Every kernel computes one output
// element per thread (reductions use one threadgroup per output), gathering
// its inputs, so no kernel needs atomics and results are deterministic.

#include <metal_stdlib>
using namespace metal;

// A strided view: element i (row-major over shape) is at
// offset + sum(idx[d] * strides[d]). Matches mps_view in mps.h.
struct View {
    int ndim;
    int offset;
    int shape[8];
    int strides[8];
};

// Matches mps_geom in mps.h.
struct Geom {
    int N, C, H, W, KH, KW, OH, OW, stride, pad;
};

inline int offset_of(uint i, constant View& v) {
    int off = v.offset;
    int rem = int(i);
    for (int d = v.ndim - 1; d >= 0; --d) {
        int s = v.shape[d];
        off += (rem % s) * v.strides[d];
        rem /= s;
    }
    return off;
}

// Op codes match backend.UnaryOp.
kernel void unary_op(device const float* x [[buffer(0)]],
                     device float* y [[buffer(1)]],
                     constant int& op [[buffer(2)]],
                     constant float& s [[buffer(3)]],
                     constant uint& n [[buffer(4)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float v = x[i];
    float r = 0;
    switch (op) {
    case 0: r = -v; break;
    case 1: r = exp(v); break;
    case 2: r = log(v); break;
    case 3: r = sqrt(v); break;
    case 4: r = fabs(v); break;
    case 5: r = max(v, 0.0f); break;
    case 6: r = 1.0f / (1.0f + exp(-v)); break;
    case 7: r = tanh(v); break;
    case 8: r = v > 0 ? 1.0f : (v < 0 ? -1.0f : 0.0f); break;
    case 9: r = v > 0 ? 1.0f : 0.0f; break;
    case 10: r = v + s; break;
    case 11: r = v * s; break;
    case 12: {
        // Metal's pow is undefined for negative bases; follow C's pow instead.
        r = pow(fabs(v), s);
        if (v < 0) {
            if (floor(s) != s) r = NAN;
            else if (fmod(fabs(s), 2.0f) == 1.0f) r = -r;
        }
        break;
    }
    }
    y[i] = r;
}

// Op codes match backend.BinaryOp.
kernel void binary_op(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* out [[buffer(2)]],
                      constant View& va [[buffer(3)]],
                      constant View& vb [[buffer(4)]],
                      constant int& op [[buffer(5)]],
                      constant uint& n [[buffer(6)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float x = a[offset_of(i, va)];
    float y = b[offset_of(i, vb)];
    float r;
    switch (op) {
    case 0: r = x + y; break;
    case 1: r = x - y; break;
    case 2: r = x * y; break;
    default: r = x / y; break;
    }
    out[i] = r;
}

kernel void copy_view(device const float* src [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant View& v [[buffer(2)]],
                      constant uint& n [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    out[i] = src[offset_of(i, v)];
}

// Reduces the middle axis of an (outer, n, inner) array. One threadgroup per
// output; the threadgroup size is a power of two no larger than 256.
kernel void reduce_op(device const float* x [[buffer(0)]],
                      device float* out [[buffer(1)]],
                      constant int& op [[buffer(2)]],
                      constant uint3& dims [[buffer(3)]],
                      uint gid [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint tsize [[threads_per_threadgroup]]) {
    uint n = dims.y, inner = dims.z;
    uint o = gid / inner, in = gid % inner;
    float acc = op == 0 ? 0.0f : -INFINITY;
    for (uint k = tid; k < n; k += tsize) {
        float v = x[(o * n + k) * inner + in];
        acc = op == 0 ? acc + v : max(acc, v);
    }
    threadgroup float buf[256];
    buf[tid] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tsize / 2; s > 0; s >>= 1) {
        if (tid < s) {
            buf[tid] = op == 0 ? buf[tid] + buf[tid + s] : max(buf[tid], buf[tid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) out[gid] = buf[0];
}

// One thread per element of the (N*OH*OW, C*KH*KW) column matrix.
kernel void im2col(device const float* x [[buffer(0)]],
                   device float* cols [[buffer(1)]],
                   constant Geom& g [[buffer(2)]],
                   constant uint& n [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    int K = g.C * g.KH * g.KW;
    int row = int(i) / K, col = int(i) % K;
    int ow = row % g.OW, t = row / g.OW;
    int oh = t % g.OH, b = t / g.OH;
    int kw = col % g.KW, t2 = col / g.KW;
    int kh = t2 % g.KH, c = t2 / g.KH;
    int ih = oh * g.stride - g.pad + kh;
    int iw = ow * g.stride - g.pad + kw;
    bool inside = ih >= 0 && ih < g.H && iw >= 0 && iw < g.W;
    cols[i] = inside ? x[((b * g.C + c) * g.H + ih) * g.W + iw] : 0.0f;
}

// One thread per input element: sums every column entry read from it.
kernel void col2im(device const float* cols [[buffer(0)]],
                   device float* x [[buffer(1)]],
                   constant Geom& g [[buffer(2)]],
                   constant uint& n [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    int iw = int(i) % g.W, t = int(i) / g.W;
    int ih = t % g.H; t /= g.H;
    int c = t % g.C, b = t / g.C;
    int K = g.C * g.KH * g.KW;
    float sum = 0;
    for (int kh = 0; kh < g.KH; ++kh) {
        int ohs = ih + g.pad - kh;
        if (ohs < 0 || ohs % g.stride != 0) continue;
        int oh = ohs / g.stride;
        if (oh >= g.OH) continue;
        for (int kw = 0; kw < g.KW; ++kw) {
            int ows = iw + g.pad - kw;
            if (ows < 0 || ows % g.stride != 0) continue;
            int ow = ows / g.stride;
            if (ow >= g.OW) continue;
            sum += cols[((b * g.OH + oh) * g.OW + ow) * K + (c * g.KH + kh) * g.KW + kw];
        }
    }
    x[i] = sum;
}

// Index (within its plane) of the first maximum of window (oh, ow).
inline int window_argmax(device const float* x, int base, int oh, int ow, constant Geom& g) {
    float best = -INFINITY;
    int best_idx = -1;
    for (int kh = 0; kh < g.KH; ++kh) {
        for (int kw = 0; kw < g.KW; ++kw) {
            int idx = (oh * g.stride + kh) * g.W + ow * g.stride + kw;
            float v = x[base + idx];
            if (v > best || best_idx < 0) {
                best = v;
                best_idx = idx;
            }
        }
    }
    return best_idx;
}

kernel void maxpool_fwd(device const float* x [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        constant Geom& g [[buffer(2)]],
                        constant uint& n [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    int ow = int(i) % g.OW, t = int(i) / g.OW;
    int oh = t % g.OH, plane = t / g.OH;
    int base = plane * g.H * g.W;
    out[i] = x[base + window_argmax(x, base, oh, ow, g)];
}

// One thread per input element: sums the gradients of the windows whose
// maximum it is, matching the CPU backend's first-maximum tie-breaking.
kernel void maxpool_bwd(device const float* x [[buffer(0)]],
                        device const float* grad [[buffer(1)]],
                        device float* dx [[buffer(2)]],
                        constant Geom& g [[buffer(3)]],
                        constant uint& n [[buffer(4)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    int iw = int(i) % g.W, t = int(i) / g.W;
    int ih = t % g.H, plane = t / g.H;
    int base = plane * g.H * g.W;
    int me = ih * g.W + iw;
    int lo_h = ih - g.KH + 1, lo_w = iw - g.KW + 1;
    int oh0 = lo_h <= 0 ? 0 : (lo_h + g.stride - 1) / g.stride;
    int ow0 = lo_w <= 0 ? 0 : (lo_w + g.stride - 1) / g.stride;
    int oh1 = min(ih / g.stride, g.OH - 1);
    int ow1 = min(iw / g.stride, g.OW - 1);
    float sum = 0;
    for (int oh = oh0; oh <= oh1; ++oh) {
        for (int ow = ow0; ow <= ow1; ++ow) {
            if (window_argmax(x, base, oh, ow, g) == me) {
                sum += grad[(plane * g.OH + oh) * g.OW + ow];
            }
        }
    }
    dx[i] = sum;
}
