// Package onnx reads the weights (graph initializers) stored in ONNX models,
// including weights saved as external data next to the model file.
package onnx

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"

	"github.com/donghquinn/go-tensor/internal/dtype"
	"github.com/donghquinn/go-tensor/tensor"
)

// ONNX TensorProto.DataType values.
var kinds = map[uint64]dtype.Kind{
	1: dtype.Float32, 2: dtype.Uint8, 3: dtype.Int8, 4: dtype.Uint16, 5: dtype.Int16,
	6: dtype.Int32, 7: dtype.Int64, 9: dtype.Bool, 10: dtype.Float16, 11: dtype.Float64,
	12: dtype.Uint32, 13: dtype.Uint64, 16: dtype.BFloat16,
}

// Field numbers from onnx.proto.
const (
	modelGraph = 7

	graphInitializer = 5

	tensorDims         = 1
	tensorDataType     = 2
	tensorFloatData    = 4
	tensorInt32Data    = 5
	tensorInt64Data    = 7
	tensorName         = 8
	tensorRawData      = 9
	tensorDoubleData   = 10
	tensorUint64Data   = 11
	tensorExternalData = 13
	tensorDataLocation = 14

	locationExternal = 1
)

// Load reads every initializer in the model at path, keyed by name. External
// data files are resolved relative to the model's directory.
func Load(path string) (map[string]*tensor.Tensor, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	out, err := Decode(b, filepath.Dir(path))
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return out, nil
}

// Decode parses a serialized ModelProto. baseDir is where external data files
// are looked up; pass "" to reject models with external data.
func Decode(model []byte, baseDir string) (map[string]*tensor.Tensor, error) {
	out := map[string]*tensor.Tensor{}
	m := &decoder{model}
	for !m.done() {
		num, wire, err := m.field()
		if err != nil {
			return nil, err
		}
		if num != modelGraph || wire != wireBytes {
			if err := m.skip(wire); err != nil {
				return nil, err
			}
			continue
		}
		graph, err := m.bytes()
		if err != nil {
			return nil, err
		}
		g := &decoder{graph}
		for !g.done() {
			num, wire, err := g.field()
			if err != nil {
				return nil, err
			}
			if num != graphInitializer || wire != wireBytes {
				if err := g.skip(wire); err != nil {
					return nil, err
				}
				continue
			}
			tb, err := g.bytes()
			if err != nil {
				return nil, err
			}
			name, t, err := decodeTensor(tb, baseDir)
			if err != nil {
				return nil, fmt.Errorf("onnx: initializer %q: %w", name, err)
			}
			out[name] = t
		}
	}
	return out, nil
}

func decodeTensor(b []byte, baseDir string) (string, *tensor.Tensor, error) {
	var (
		name     string
		dims     []int
		dataType uint64
		raw      []byte
		typed    []float64
		external = map[string]string{}
		location uint64
	)
	d := &decoder{b}
	for !d.done() {
		num, wire, err := d.field()
		if err != nil {
			return name, nil, err
		}
		switch num {
		case tensorDims:
			var vals []float64
			if vals, err = d.repeated(wire, wireVarint, nil, asInt64); err == nil {
				for _, v := range vals {
					if v < 0 {
						return name, nil, fmt.Errorf("negative dimension %v", v)
					}
					dims = append(dims, int(v))
				}
			}
		case tensorDataType:
			dataType, err = d.varint()
		case tensorFloatData:
			typed, err = d.repeated(wire, wireFixed32, typed, asFloat32)
		case tensorInt32Data:
			typed, err = d.repeated(wire, wireVarint, typed, asInt64)
		case tensorInt64Data:
			typed, err = d.repeated(wire, wireVarint, typed, asInt64)
		case tensorUint64Data:
			typed, err = d.repeated(wire, wireVarint, typed, asUint64)
		case tensorDoubleData:
			typed, err = d.repeated(wire, wireFixed64, typed, asFloat64)
		case tensorName:
			var s []byte
			s, err = d.bytes()
			name = string(s)
		case tensorRawData:
			raw, err = d.bytes()
		case tensorExternalData:
			var kv []byte
			if kv, err = d.bytes(); err == nil {
				err = decodeKeyValue(kv, external)
			}
		case tensorDataLocation:
			location, err = d.varint()
		default:
			err = d.skip(wire)
		}
		if err != nil {
			return name, nil, err
		}
	}

	kind, ok := kinds[dataType]
	if !ok {
		return name, nil, fmt.Errorf("unsupported data type %d", dataType)
	}
	numel, err := dtype.Numel(dims)
	if err != nil {
		return name, nil, err
	}

	var data []float64
	switch {
	case location == locationExternal:
		raw, err := readExternal(external, baseDir)
		if err != nil {
			return name, nil, err
		}
		if data, err = dtype.Decode(kind, raw, binary.LittleEndian); err != nil {
			return name, nil, err
		}
	case raw != nil:
		if data, err = dtype.Decode(kind, raw, binary.LittleEndian); err != nil {
			return name, nil, err
		}
	default:
		data = typed
		// 16-bit floats are stored as their bit patterns in int32_data.
		for i, v := range data {
			switch kind {
			case dtype.Float16:
				data[i] = dtype.HalfToFloat64(uint16(v))
			case dtype.BFloat16:
				data[i] = dtype.BFloat16ToFloat64(uint16(v))
			}
		}
	}
	if len(data) != numel {
		return name, nil, fmt.Errorf("has %d values for shape %v", len(data), dims)
	}
	if len(dims) == 0 {
		return name, tensor.Scalar(data[0]), nil
	}
	return name, tensor.New(data, dims...), nil
}

func decodeKeyValue(b []byte, into map[string]string) error {
	var key, value string
	d := &decoder{b}
	for !d.done() {
		num, wire, err := d.field()
		if err != nil {
			return err
		}
		if wire != wireBytes || (num != 1 && num != 2) {
			if err := d.skip(wire); err != nil {
				return err
			}
			continue
		}
		s, err := d.bytes()
		if err != nil {
			return err
		}
		if num == 1 {
			key = string(s)
		} else {
			value = string(s)
		}
	}
	into[key] = value
	return nil
}

// readExternal reads the byte range described by a tensor's external_data
// entries. The location must stay inside baseDir.
func readExternal(info map[string]string, baseDir string) ([]byte, error) {
	loc := info["location"]
	if baseDir == "" {
		return nil, fmt.Errorf("external data %q but no base directory", loc)
	}
	if !filepath.IsLocal(loc) {
		return nil, fmt.Errorf("external data location %q escapes the model directory", loc)
	}
	f, err := os.Open(filepath.Join(baseDir, loc))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	offset, length := int64(0), int64(-1)
	if s, ok := info["offset"]; ok {
		if offset, err = strconv.ParseInt(s, 10, 64); err != nil || offset < 0 {
			return nil, fmt.Errorf("bad external data offset %q", s)
		}
	}
	if s, ok := info["length"]; ok {
		if length, err = strconv.ParseInt(s, 10, 64); err != nil || length < 0 {
			return nil, fmt.Errorf("bad external data length %q", s)
		}
	}
	if _, err := f.Seek(offset, io.SeekStart); err != nil {
		return nil, err
	}
	if length < 0 {
		return io.ReadAll(f)
	}
	if st, err := f.Stat(); err != nil {
		return nil, err
	} else if offset+length > st.Size() {
		return nil, fmt.Errorf("external data %q: range %d+%d exceeds file size %d", loc, offset, length, st.Size())
	}
	b := make([]byte, length)
	if _, err := io.ReadFull(f, b); err != nil {
		return nil, fmt.Errorf("external data %q: %w", loc, err)
	}
	return b, nil
}
