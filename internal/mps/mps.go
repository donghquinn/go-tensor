//go:build darwin && mps

package mps

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -O2
#cgo LDFLAGS: -framework Foundation -framework Metal -framework MetalPerformanceShaders
#include <stdlib.h>
#include "mps.h"
*/
import "C"

import (
	_ "embed"
	"errors"
	"fmt"
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/donghquinn/go-tensor/internal/backend"
)

//go:embed kernels.metal
var kernelSource string

const (
	// gcThreshold is how many bytes of new GPU buffers may be created between
	// forced garbage collections. Go's collector cannot see GPU memory, so
	// without this, unreferenced buffers would pile up between collections.
	gcThreshold = 256 << 20
	// maxPooled bounds the bytes of idle buffers kept for reuse.
	maxPooled = 512 << 20
	minClass  = 256
)

var (
	initOnce sync.Once
	initErr  error

	mu               sync.Mutex // serializes encoding into the shared command buffer
	allocatedSinceGC atomic.Int64

	// Freed buffers are kept by size class and reused, like PyTorch's caching
	// allocator: creating Metal buffers is far slower than running most kernels.
	// Reuse is safe because the GPU executes commands in submission order, so
	// work encoded later never overtakes earlier work on the same buffer.
	poolMu      sync.Mutex
	pool        = map[int][]pooled{}
	pooledBytes int
)

// pooled is an idle buffer. stamp is the command buffer that was being
// encoded when it was freed: once the GPU completes it, no queued work can
// still touch the buffer, so the CPU may write it directly.
type pooled struct {
	ptr   unsafe.Pointer
	stamp uint64
}

var device = backend.Device{Type: backend.MPS}

// New initializes Metal and returns the backend.
func New() (backend.Backend, error) {
	initOnce.Do(func() {
		src := C.CString(kernelSource)
		defer C.free(unsafe.Pointer(src))
		if msg := C.mps_init(src); msg != nil {
			initErr = errors.New(C.GoString(msg))
			C.free(unsafe.Pointer(msg))
		}
	})
	if initErr != nil {
		return nil, initErr
	}
	return Backend{}, nil
}

// DeviceName returns the GPU's name, e.g. "Apple M5".
func DeviceName() string { return C.GoString(C.mps_device_name()) }

// buffer is a float32 MTLBuffer of class bytes holding n elements.
type buffer struct {
	ptr   unsafe.Pointer
	n     int
	class int
}

func (b *buffer) Device() backend.Device { return device }
func (b *buffer) DType() backend.DType   { return backend.Float32 }
func (b *buffer) Len() int               { return b.n }

// sizeClass rounds a byte count up to a power of two.
func sizeClass(bytes int) int {
	c := minClass
	for c < bytes {
		c <<= 1
	}
	return c
}

// alloc returns an uninitialized buffer for n float32 values, reusing a
// pooled buffer when one is free.
func alloc(n int) *buffer {
	if n > math.MaxInt32 {
		panic(fmt.Sprintf("mps: tensor of %d elements is too large", n))
	}
	class := sizeClass(n * 4)
	p := take(class, func(pooled) bool { return true })
	if p == nil {
		p = fresh(class, func() unsafe.Pointer { return C.mps_alloc(C.size_t(class)) })
	}
	return track(&buffer{ptr: p, n: n, class: class})
}

// fresh creates a new Metal buffer, collecting garbage first if many bytes
// were created since the last collection.
func fresh(class int, create func() unsafe.Pointer) unsafe.Pointer {
	if allocatedSinceGC.Add(int64(class)) > gcThreshold {
		allocatedSinceGC.Store(0)
		runtime.GC() // run finalizers so unreachable buffers return to the pool
	}
	p := create()
	if p == nil {
		panic(fmt.Sprintf("mps: failed to allocate %d bytes of GPU memory", class))
	}
	return p
}

func track(b *buffer) *buffer {
	runtime.SetFinalizer(b, recycle)
	return b
}

// take removes and returns a pooled buffer of the class that satisfies ok,
// or nil. The most recently freed buffers are tried first.
func take(class int, ok func(pooled) bool) unsafe.Pointer {
	poolMu.Lock()
	defer poolMu.Unlock()
	free := pool[class]
	for i := len(free) - 1; i >= 0; i-- {
		if ok(free[i]) {
			p := free[i].ptr
			pool[class] = append(free[:i], free[i+1:]...)
			pooledBytes -= class
			return p
		}
	}
	return nil
}

func recycle(b *buffer) {
	stamp := uint64(C.mps_pending_id())
	poolMu.Lock()
	defer poolMu.Unlock()
	if pooledBytes+b.class > maxPooled {
		C.mps_release(b.ptr)
		return
	}
	pool[b.class] = append(pool[b.class], pooled{b.ptr, stamp})
	pooledBytes += b.class
}

func buf(s backend.Storage) *buffer {
	b, ok := s.(*buffer)
	if !ok {
		panic(fmt.Sprintf("mps: storage %T is not on the MPS device", s))
	}
	return b
}

// Backend implements backend.Backend with Metal compute kernels and
// MPSMatrixMultiplication. It supports float32 only.
type Backend struct{}

func (Backend) Device() backend.Device { return device }

func (Backend) Zeros(dt backend.DType, n int) backend.Storage {
	checkDType(dt)
	b := alloc(n)
	mu.Lock()
	C.mps_zero(b.ptr, C.size_t(n*4))
	mu.Unlock()
	return b
}

func checkDType(dt backend.DType) {
	if dt != backend.Float32 {
		panic("mps: only float32 is supported")
	}
}

func (Backend) FromHost(data any) backend.Storage {
	src, ok := data.([]float32)
	if !ok {
		panic(fmt.Sprintf("mps: only float32 is supported, got %T", data))
	}
	// The CPU writes the buffer directly, so a pooled buffer is only reused
	// once the GPU has finished all work that was queued when it was freed.
	class := sizeClass(len(src) * 4)
	done := uint64(C.mps_completed_id())
	p := take(class, func(b pooled) bool { return b.stamp <= done })
	if p == nil {
		p = fresh(class, func() unsafe.Pointer { return C.mps_alloc(C.size_t(class)) })
	}
	if len(src) > 0 {
		copy(unsafe.Slice((*float32)(C.mps_contents(p)), len(src)), src)
	}
	return track(&buffer{ptr: p, n: len(src), class: class})
}

func (be Backend) ToHost(s backend.Storage) any {
	b := buf(s)
	be.Sync()
	out := make([]float32, b.n)
	if b.n > 0 {
		copy(out, unsafe.Slice((*float32)(C.mps_contents(b.ptr)), b.n))
	}
	runtime.KeepAlive(b)
	return out
}

func (be Backend) Cast(s backend.Storage, dt backend.DType) backend.Storage {
	checkDType(dt)
	b := buf(s)
	return be.Copy(b, backend.View{Shape: []int{b.n}, Strides: []int{1}})
}

func (Backend) Write(dst backend.Storage, offset int, src backend.Storage) {
	d, s := buf(dst), buf(src)
	mu.Lock()
	C.mps_write(d.ptr, C.size_t(offset*4), s.ptr, C.size_t(s.n*4))
	mu.Unlock()
	runtime.KeepAlive(d)
	runtime.KeepAlive(s)
}

func (Backend) Sync() {
	mu.Lock()
	msg := C.mps_sync()
	mu.Unlock()
	if msg != nil {
		err := C.GoString(msg)
		C.free(unsafe.Pointer(msg))
		panic("mps: " + err)
	}
}

func (Backend) Unary(op backend.UnaryOp, s float64, x backend.Storage) backend.Storage {
	in := buf(x)
	out := alloc(in.n)
	mu.Lock()
	C.mps_unary(C.int(op), C.float(s), in.ptr, out.ptr, C.uint(in.n))
	mu.Unlock()
	runtime.KeepAlive(in)
	return out
}

// cview converts a view to the C struct, collapsing contiguous views to one
// dimension so kernels skip the index arithmetic.
func cview(v backend.View) C.mps_view {
	var cv C.mps_view
	n := backend.Numel(v.Shape)
	if isContiguous(v) {
		cv.ndim, cv.offset = 1, C.int(v.Offset)
		cv.shape[0], cv.strides[0] = C.int(n), 1
		return cv
	}
	if len(v.Shape) > backend.MaxDims {
		panic(fmt.Sprintf("mps: at most %d dimensions are supported", backend.MaxDims))
	}
	cv.ndim, cv.offset = C.int(len(v.Shape)), C.int(v.Offset)
	for d := range v.Shape {
		cv.shape[d], cv.strides[d] = C.int(v.Shape[d]), C.int(v.Strides[d])
	}
	return cv
}

func isContiguous(v backend.View) bool {
	want := backend.ContiguousStrides(v.Shape)
	for d := range want {
		if v.Shape[d] != 1 && v.Strides[d] != want[d] {
			return false
		}
	}
	return true
}

func (Backend) Binary(op backend.BinaryOp, a backend.Storage, av backend.View, b backend.Storage, bv backend.View) backend.Storage {
	ab, bb := buf(a), buf(b)
	n := backend.Numel(av.Shape)
	out := alloc(n)
	ca, cb := cview(av), cview(bv)
	mu.Lock()
	C.mps_binary(C.int(op), ab.ptr, &ca, bb.ptr, &cb, out.ptr, C.uint(n))
	mu.Unlock()
	runtime.KeepAlive(ab)
	runtime.KeepAlive(bb)
	return out
}

func (Backend) Copy(src backend.Storage, v backend.View) backend.Storage {
	sb := buf(src)
	n := backend.Numel(v.Shape)
	out := alloc(n)
	cv := cview(v)
	mu.Lock()
	C.mps_copy(sb.ptr, &cv, out.ptr, C.uint(n))
	mu.Unlock()
	runtime.KeepAlive(sb)
	return out
}

func (Backend) Reduce(op backend.ReduceOp, x backend.Storage, outer, n, inner int) backend.Storage {
	xb := buf(x)
	out := alloc(outer * inner)
	mu.Lock()
	C.mps_reduce(C.int(op), xb.ptr, out.ptr, C.uint(outer), C.uint(n), C.uint(inner))
	mu.Unlock()
	runtime.KeepAlive(xb)
	return out
}

func (Backend) MatMul(a, b backend.Storage, m, k, n int, transA, transB bool) backend.Storage {
	ab, bb := buf(a), buf(b)
	out := alloc(m * n)
	mu.Lock()
	C.mps_matmul(ab.ptr, bb.ptr, out.ptr, C.int(m), C.int(k), C.int(n), cbool(transA), cbool(transB))
	mu.Unlock()
	runtime.KeepAlive(ab)
	runtime.KeepAlive(bb)
	return out
}

func cbool(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

func cgeom(g backend.ConvGeom) C.mps_geom {
	return C.mps_geom{N: C.int(g.N), C: C.int(g.C), H: C.int(g.H), W: C.int(g.W), KH: C.int(g.KH), KW: C.int(g.KW),
		OH: C.int(g.OH), OW: C.int(g.OW), stride: C.int(g.Stride), pad: C.int(g.Padding)}
}

func (Backend) Im2col(x backend.Storage, g backend.ConvGeom) backend.Storage {
	xb := buf(x)
	n := g.N * g.OH * g.OW * g.C * g.KH * g.KW
	out := alloc(n)
	cg := cgeom(g)
	mu.Lock()
	C.mps_im2col(xb.ptr, out.ptr, &cg, C.uint(n))
	mu.Unlock()
	runtime.KeepAlive(xb)
	return out
}

func (Backend) Col2im(cols backend.Storage, g backend.ConvGeom) backend.Storage {
	cb := buf(cols)
	n := g.N * g.C * g.H * g.W
	out := alloc(n)
	cg := cgeom(g)
	mu.Lock()
	C.mps_col2im(cb.ptr, out.ptr, &cg, C.uint(n))
	mu.Unlock()
	runtime.KeepAlive(cb)
	return out
}

func (Backend) MaxPool2d(x backend.Storage, g backend.ConvGeom) backend.Storage {
	xb := buf(x)
	n := g.N * g.C * g.OH * g.OW
	out := alloc(n)
	cg := cgeom(g)
	mu.Lock()
	C.mps_maxpool_fwd(xb.ptr, out.ptr, &cg, C.uint(n))
	mu.Unlock()
	runtime.KeepAlive(xb)
	return out
}

func (Backend) MaxPool2dBackward(x, grad backend.Storage, g backend.ConvGeom) backend.Storage {
	xb, gb := buf(x), buf(grad)
	n := g.N * g.C * g.H * g.W
	out := alloc(n)
	cg := cgeom(g)
	mu.Lock()
	C.mps_maxpool_bwd(xb.ptr, gb.ptr, out.ptr, &cg, C.uint(n))
	mu.Unlock()
	runtime.KeepAlive(xb)
	runtime.KeepAlive(gb)
	return out
}
