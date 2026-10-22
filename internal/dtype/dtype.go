// Package dtype converts the element types found in model files to float64.
package dtype

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Kind is an element type as stored on disk.
type Kind int

const (
	Invalid Kind = iota
	Float64
	Float32
	Float16
	BFloat16
	Int64
	Int32
	Int16
	Int8
	Uint64
	Uint32
	Uint16
	Uint8
	Bool
)

var names = [...]string{"invalid", "float64", "float32", "float16", "bfloat16",
	"int64", "int32", "int16", "int8", "uint64", "uint32", "uint16", "uint8", "bool"}

func (k Kind) String() string {
	if k < 0 || int(k) >= len(names) {
		return fmt.Sprintf("Kind(%d)", int(k))
	}
	return names[k]
}

// Size returns the number of bytes per element.
func (k Kind) Size() int {
	switch k {
	case Float64, Int64, Uint64:
		return 8
	case Float32, Int32, Uint32:
		return 4
	case Float16, BFloat16, Int16, Uint16:
		return 2
	case Int8, Uint8, Bool:
		return 1
	}
	return 0
}

// Decode converts raw bytes holding elements of kind k to float64 values.
// Integers beyond ±2^53 lose precision.
func Decode(k Kind, b []byte, order binary.ByteOrder) ([]float64, error) {
	sz := k.Size()
	if sz == 0 {
		return nil, fmt.Errorf("dtype: unsupported element type %v", k)
	}
	if len(b)%sz != 0 {
		return nil, fmt.Errorf("dtype: %d bytes is not a whole number of %v elements", len(b), k)
	}
	out := make([]float64, len(b)/sz)
	for i := range out {
		e := b[i*sz : (i+1)*sz]
		switch k {
		case Float64:
			out[i] = math.Float64frombits(order.Uint64(e))
		case Float32:
			out[i] = float64(math.Float32frombits(order.Uint32(e)))
		case Float16:
			out[i] = HalfToFloat64(order.Uint16(e))
		case BFloat16:
			out[i] = BFloat16ToFloat64(order.Uint16(e))
		case Int64:
			out[i] = float64(int64(order.Uint64(e)))
		case Int32:
			out[i] = float64(int32(order.Uint32(e)))
		case Int16:
			out[i] = float64(int16(order.Uint16(e)))
		case Int8:
			out[i] = float64(int8(e[0]))
		case Uint64:
			out[i] = float64(order.Uint64(e))
		case Uint32:
			out[i] = float64(order.Uint32(e))
		case Uint16:
			out[i] = float64(order.Uint16(e))
		case Uint8:
			out[i] = float64(e[0])
		case Bool:
			if e[0] != 0 {
				out[i] = 1
			}
		}
	}
	return out, nil
}

// HalfToFloat64 converts IEEE 754 half-precision bits to float64.
func HalfToFloat64(h uint16) float64 {
	exp := int(h>>10) & 0x1f
	frac := float64(h & 0x3ff)
	var v float64
	switch exp {
	case 0: // zero or subnormal
		v = math.Ldexp(frac, -24)
	case 0x1f:
		if frac == 0 {
			v = math.Inf(1)
		} else {
			v = math.NaN()
		}
	default:
		v = math.Ldexp(frac+1024, exp-25)
	}
	if h&0x8000 != 0 {
		v = -v
	}
	return v
}

// BFloat16ToFloat64 converts bfloat16 bits (the top half of a float32) to float64.
func BFloat16ToFloat64(b uint16) float64 {
	return float64(math.Float32frombits(uint32(b) << 16))
}

// MaxElements bounds the size of a tensor read from a file (2^34 elements,
// 128 GiB as float64), so corrupt headers cannot trigger huge allocations.
const MaxElements = 1 << 34

// Numel returns the number of elements in shape, or an error if a dimension
// is negative or the total exceeds MaxElements (including integer overflow).
func Numel(shape []int) (int, error) {
	n := 1
	for _, d := range shape {
		if d < 0 {
			return 0, fmt.Errorf("negative dimension in shape %v", shape)
		}
		if d != 0 && n > MaxElements/d {
			return 0, fmt.Errorf("shape %v has more than %d elements", shape, MaxElements)
		}
		n *= d
	}
	return n, nil
}
