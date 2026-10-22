// Package gguf reads GGUF files, the format used by llama.cpp and ggml, and
// dequantizes their tensors to float64.
package gguf

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/donghquinn/go-tensor/internal/dtype"
	"github.com/donghquinn/go-tensor/tensor"
)

const magic = "GGUF"

// ggml tensor types.
const (
	typeF32  = 0
	typeF16  = 1
	typeQ4_0 = 2
	typeQ4_1 = 3
	typeQ5_0 = 6
	typeQ5_1 = 7
	typeQ8_0 = 8
	typeQ4_K = 12
	typeQ5_K = 13
	typeQ6_K = 14
	typeI8   = 24
	typeI16  = 25
	typeI32  = 26
	typeI64  = 27
	typeF64  = 28
	typeBF16 = 30
)

// format describes how a ggml type is laid out: blockSize elements are
// stored in blockBytes bytes and decoded by dequant (nil for plain types).
type format struct {
	blockSize, blockBytes int
	plain                 dtype.Kind
	dequant               func(block []byte, out []float32)
}

var formats = map[uint32]format{
	typeF32:  {1, 4, dtype.Float32, nil},
	typeF16:  {1, 2, dtype.Float16, nil},
	typeBF16: {1, 2, dtype.BFloat16, nil},
	typeF64:  {1, 8, dtype.Float64, nil},
	typeI8:   {1, 1, dtype.Int8, nil},
	typeI16:  {1, 2, dtype.Int16, nil},
	typeI32:  {1, 4, dtype.Int32, nil},
	typeI64:  {1, 8, dtype.Int64, nil},
	typeQ4_0: {32, 18, 0, dequantQ4_0},
	typeQ4_1: {32, 20, 0, dequantQ4_1},
	typeQ5_0: {32, 22, 0, dequantQ5_0},
	typeQ5_1: {32, 24, 0, dequantQ5_1},
	typeQ8_0: {32, 34, 0, dequantQ8_0},
	typeQ4_K: {256, 144, 0, dequantQ4_K},
	typeQ5_K: {256, 176, 0, dequantQ5_K},
	typeQ6_K: {256, 210, 0, dequantQ6_K},
}

// Load reads every tensor in a GGUF file.
func Load(path string) (map[string]*tensor.Tensor, error) {
	tensors, _, err := LoadWithMetadata(path)
	return tensors, err
}

// LoadWithMetadata reads every tensor and the key-value metadata. Metadata
// values are uint8…uint64, int8…int64, float32, float64, bool, string or
// []any for arrays.
func LoadWithMetadata(path string) (map[string]*tensor.Tensor, map[string]any, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	tensors, meta, err := read(f)
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", path, err)
	}
	return tensors, meta, nil
}

type tensorInfo struct {
	name   string
	shape  []int
	typ    uint32
	offset uint64
}

// countingReader tracks the position so the data section can be aligned.
type countingReader struct {
	r *bufio.Reader
	n int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

func read(f *os.File) (map[string]*tensor.Tensor, map[string]any, error) {
	cr := &countingReader{r: bufio.NewReader(f)}
	h := &header{r: cr}

	if m := h.bytes(4); string(m) != magic {
		return nil, nil, errors.New("gguf: not a GGUF file")
	}
	version := h.u32()
	if h.err == nil && (version < 2 || version > 3) {
		return nil, nil, fmt.Errorf("gguf: unsupported version %d", version)
	}
	nTensors, nKV := h.u64(), h.u64()
	if h.err == nil && (nTensors > 1<<24 || nKV > 1<<24) {
		return nil, nil, errors.New("gguf: implausible tensor or metadata count")
	}

	meta := map[string]any{}
	for i := uint64(0); i < nKV && h.err == nil; i++ {
		key := h.str()
		meta[key] = h.value(h.u32(), 0)
	}
	alignment := int64(32)
	if a, ok := meta["general.alignment"].(uint32); ok && a > 0 {
		alignment = int64(a)
	}

	infos := make([]tensorInfo, 0, nTensors)
	for i := uint64(0); i < nTensors && h.err == nil; i++ {
		ti := tensorInfo{name: h.str()}
		nDims := h.u32()
		if nDims > 8 {
			return nil, nil, fmt.Errorf("gguf: tensor %q has %d dimensions", ti.name, nDims)
		}
		dims := make([]int, nDims)
		for d := range dims {
			dims[d] = int(h.u64())
		}
		// ggml lists the fastest-varying dimension first.
		for d := range dims {
			ti.shape = append(ti.shape, dims[len(dims)-1-d])
		}
		ti.typ = h.u32()
		ti.offset = h.u64()
		infos = append(infos, ti)
	}
	if h.err != nil {
		return nil, nil, fmt.Errorf("gguf: header: %w", h.err)
	}
	dataStart := (cr.n + alignment - 1) / alignment * alignment

	tensors := make(map[string]*tensor.Tensor, len(infos))
	for _, ti := range infos {
		t, err := readTensor(f, dataStart, ti)
		if err != nil {
			return nil, nil, fmt.Errorf("gguf: tensor %q: %w", ti.name, err)
		}
		tensors[ti.name] = t
	}
	return tensors, meta, nil
}

func readTensor(f *os.File, dataStart int64, ti tensorInfo) (*tensor.Tensor, error) {
	fm, ok := formats[ti.typ]
	if !ok {
		return nil, fmt.Errorf("unsupported ggml type %d", ti.typ)
	}
	numel, err := dtype.Numel(ti.shape)
	if err != nil {
		return nil, err
	}
	if numel%fm.blockSize != 0 {
		return nil, fmt.Errorf("%d elements is not a multiple of the block size %d", numel, fm.blockSize)
	}
	size := int64(numel / fm.blockSize * fm.blockBytes)
	start := dataStart + int64(ti.offset)
	if st, err := f.Stat(); err != nil {
		return nil, err
	} else if ti.offset > uint64(st.Size()) || start+size > st.Size() {
		return nil, fmt.Errorf("data (%d bytes at %d) extends past the end of the file", size, start)
	}
	raw := make([]byte, size)
	if _, err := f.ReadAt(raw, start); err != nil {
		return nil, err
	}

	var data []float64
	if fm.dequant == nil {
		if data, err = dtype.Decode(fm.plain, raw, binary.LittleEndian); err != nil {
			return nil, err
		}
	} else {
		// ggml dequantizes in float32; doing the same reproduces its values exactly.
		data = make([]float64, numel)
		block := make([]float32, fm.blockSize)
		for b := 0; b < numel/fm.blockSize; b++ {
			fm.dequant(raw[b*fm.blockBytes:(b+1)*fm.blockBytes], block)
			for i, v := range block {
				data[b*fm.blockSize+i] = float64(v)
			}
		}
	}
	if len(ti.shape) == 0 {
		return tensor.Scalar(data[0]), nil
	}
	return tensor.New(data, ti.shape...), nil
}

// header reads little-endian GGUF header fields, remembering the first error.
type header struct {
	r   io.Reader
	err error
}

func (h *header) bytes(n uint64) []byte {
	if h.err != nil {
		return nil
	}
	if n > 1<<30 {
		h.err = fmt.Errorf("field of %d bytes is too large", n)
		return nil
	}
	b := make([]byte, n)
	_, h.err = io.ReadFull(h.r, b)
	return b
}

func (h *header) u8() uint8 {
	if b := h.bytes(1); b != nil {
		return b[0]
	}
	return 0
}

func (h *header) u16() uint16 {
	if b := h.bytes(2); b != nil {
		return binary.LittleEndian.Uint16(b)
	}
	return 0
}

func (h *header) u32() uint32 {
	if b := h.bytes(4); b != nil {
		return binary.LittleEndian.Uint32(b)
	}
	return 0
}

func (h *header) u64() uint64 {
	if b := h.bytes(8); b != nil {
		return binary.LittleEndian.Uint64(b)
	}
	return 0
}

func (h *header) str() string { return string(h.bytes(h.u64())) }

// value reads one metadata value of the given GGUF value type.
func (h *header) value(typ uint32, depth int) any {
	switch typ {
	case 0:
		return h.u8()
	case 1:
		return int8(h.u8())
	case 2:
		return h.u16()
	case 3:
		return int16(h.u16())
	case 4:
		return h.u32()
	case 5:
		return int32(h.u32())
	case 6:
		return math.Float32frombits(h.u32())
	case 7:
		return h.u8() != 0
	case 8:
		return h.str()
	case 9:
		elem, n := h.u32(), h.u64()
		if depth > 4 || n > 1<<28 {
			h.err = errors.New("metadata array too large or too deeply nested")
			return nil
		}
		arr := make([]any, 0, min(n, 1<<16))
		for i := uint64(0); i < n && h.err == nil; i++ {
			arr = append(arr, h.value(elem, depth+1))
		}
		return arr
	case 10:
		return h.u64()
	case 11:
		return int64(h.u64())
	case 12:
		return math.Float64frombits(h.u64())
	}
	if h.err == nil {
		h.err = fmt.Errorf("unknown metadata value type %d", typ)
	}
	return nil
}
