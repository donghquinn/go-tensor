package onnx

import (
	"errors"
	"fmt"
	"math"
)

// Protocol-buffer wire types.
const (
	wireVarint  = 0
	wireFixed64 = 1
	wireBytes   = 2
	wireFixed32 = 5
)

var errTruncated = errors.New("onnx: truncated protobuf message")

// decoder walks the fields of one protobuf message.
type decoder struct{ b []byte }

func (d *decoder) done() bool { return len(d.b) == 0 }

func (d *decoder) varint() (uint64, error) {
	var v uint64
	for i := 0; i < 10; i++ {
		if i >= len(d.b) {
			return 0, errTruncated
		}
		c := d.b[i]
		v |= uint64(c&0x7f) << (7 * i)
		if c < 0x80 {
			d.b = d.b[i+1:]
			return v, nil
		}
	}
	return 0, errors.New("onnx: varint overflow")
}

func (d *decoder) field() (num int, wire int, err error) {
	tag, err := d.varint()
	if err != nil {
		return 0, 0, err
	}
	return int(tag >> 3), int(tag & 7), nil
}

func (d *decoder) take(n uint64) ([]byte, error) {
	if n > uint64(len(d.b)) {
		return nil, errTruncated
	}
	b := d.b[:n]
	d.b = d.b[n:]
	return b, nil
}

func (d *decoder) bytes() ([]byte, error) {
	n, err := d.varint()
	if err != nil {
		return nil, err
	}
	return d.take(n)
}

func (d *decoder) fixed32() (uint32, error) {
	b, err := d.take(4)
	if err != nil {
		return 0, err
	}
	return uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24, nil
}

func (d *decoder) fixed64() (uint64, error) {
	lo, err := d.fixed32()
	if err != nil {
		return 0, err
	}
	hi, err := d.fixed32()
	return uint64(lo) | uint64(hi)<<32, err
}

func (d *decoder) skip(wire int) error {
	var err error
	switch wire {
	case wireVarint:
		_, err = d.varint()
	case wireFixed64:
		_, err = d.take(8)
	case wireBytes:
		_, err = d.bytes()
	case wireFixed32:
		_, err = d.take(4)
	default:
		err = fmt.Errorf("onnx: unsupported wire type %d", wire)
	}
	return err
}

// repeated decodes a repeated scalar field in either packed or unpacked form,
// appending each element converted to float64 by conv.
func (d *decoder) repeated(wire int, elemWire int, dst []float64, conv func(uint64) float64) ([]float64, error) {
	read := func(r *decoder) (uint64, error) {
		switch elemWire {
		case wireFixed32:
			v, err := r.fixed32()
			return uint64(v), err
		case wireFixed64:
			return r.fixed64()
		default:
			return r.varint()
		}
	}
	if wire == wireBytes {
		b, err := d.bytes()
		if err != nil {
			return nil, err
		}
		packed := &decoder{b}
		for !packed.done() {
			v, err := read(packed)
			if err != nil {
				return nil, err
			}
			dst = append(dst, conv(v))
		}
		return dst, nil
	}
	if wire != elemWire {
		return nil, fmt.Errorf("onnx: wire type %d for repeated field of type %d", wire, elemWire)
	}
	v, err := read(d)
	if err != nil {
		return nil, err
	}
	return append(dst, conv(v)), nil
}

func asFloat32(v uint64) float64 { return float64(math.Float32frombits(uint32(v))) }
func asFloat64(v uint64) float64 { return math.Float64frombits(v) }
func asInt64(v uint64) float64   { return float64(int64(v)) }
func asUint64(v uint64) float64  { return float64(v) }
