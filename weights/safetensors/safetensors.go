// Package safetensors reads and writes the safetensors format used by Hugging
// Face (https://github.com/huggingface/safetensors).
package safetensors

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/donghquinn/go-tensor/internal/dtype"
	"github.com/donghquinn/go-tensor/tensor"
)

// maxHeaderSize bounds the JSON header, as the reference implementation does.
const maxHeaderSize = 100 << 20

var kinds = map[string]dtype.Kind{
	"F64": dtype.Float64, "F32": dtype.Float32, "F16": dtype.Float16, "BF16": dtype.BFloat16,
	"I64": dtype.Int64, "I32": dtype.Int32, "I16": dtype.Int16, "I8": dtype.Int8,
	"U64": dtype.Uint64, "U32": dtype.Uint32, "U16": dtype.Uint16, "U8": dtype.Uint8,
	"BOOL": dtype.Bool,
}

type headerEntry struct {
	DType       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

// Load reads a .safetensors file.
func Load(path string) (map[string]*tensor.Tensor, error) {
	tensors, _, err := LoadWithMetadata(path)
	return tensors, err
}

// LoadWithMetadata reads a .safetensors file and its "__metadata__" strings.
func LoadWithMetadata(path string) (map[string]*tensor.Tensor, map[string]string, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	tensors, meta, err := Decode(b)
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", path, err)
	}
	return tensors, meta, nil
}

// Decode parses safetensors data held in memory.
func Decode(b []byte) (map[string]*tensor.Tensor, map[string]string, error) {
	if len(b) < 8 {
		return nil, nil, fmt.Errorf("safetensors: file too short")
	}
	n := binary.LittleEndian.Uint64(b)
	if n > maxHeaderSize || n > uint64(len(b)-8) {
		return nil, nil, fmt.Errorf("safetensors: invalid header length %d", n)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(b[8:8+n], &raw); err != nil {
		return nil, nil, fmt.Errorf("safetensors: header: %w", err)
	}
	body := b[8+n:]

	var meta map[string]string
	tensors := make(map[string]*tensor.Tensor, len(raw))
	for name, msg := range raw {
		if name == "__metadata__" {
			if err := json.Unmarshal(msg, &meta); err != nil {
				return nil, nil, fmt.Errorf("safetensors: __metadata__: %w", err)
			}
			continue
		}
		var h headerEntry
		if err := json.Unmarshal(msg, &h); err != nil {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}
		kind, ok := kinds[h.DType]
		if !ok {
			return nil, nil, fmt.Errorf("safetensors: tensor %q has unsupported dtype %s", name, h.DType)
		}
		numel, err := dtype.Numel(h.Shape)
		if err != nil {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}
		start, end := h.DataOffsets[0], h.DataOffsets[1]
		if start < 0 || end < start || end > len(body) || end-start != numel*kind.Size() {
			return nil, nil, fmt.Errorf("safetensors: tensor %q has invalid data offsets %v", name, h.DataOffsets)
		}
		data, err := dtype.Decode(kind, body[start:end], binary.LittleEndian)
		if err != nil {
			return nil, nil, fmt.Errorf("safetensors: tensor %q: %w", name, err)
		}
		if len(h.Shape) == 0 {
			tensors[name] = tensor.FromFloat64(data[:1]).Reshape()
		} else {
			tensors[name] = tensor.FromFloat64(data, h.Shape...)
		}
	}
	return tensors, meta, nil
}

// DType selects the element type written by Save.
type DType string

const (
	F32 DType = "F32"
	F64 DType = "F64"
)

// Save writes tensors to path. dtype is F32 (what PyTorch and Hugging Face
// models normally use; values are rounded) or F64 (lossless).
func Save(path string, tensors map[string]*tensor.Tensor, dt DType, metadata map[string]string) error {
	b, err := Encode(tensors, dt, metadata)
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// Encode serializes tensors in the safetensors format.
func Encode(tensors map[string]*tensor.Tensor, dt DType, metadata map[string]string) ([]byte, error) {
	if dt != F32 && dt != F64 {
		return nil, fmt.Errorf("safetensors: unsupported output dtype %q", dt)
	}
	size := 4
	if dt == F64 {
		size = 8
	}
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		if name == "__metadata__" {
			return nil, fmt.Errorf("safetensors: %q is a reserved name", name)
		}
		names = append(names, name)
	}
	sort.Strings(names)

	header := map[string]any{}
	if len(metadata) > 0 {
		header["__metadata__"] = metadata
	}
	var body []byte
	for _, name := range names {
		t := tensors[name]
		start := len(body)
		for _, v := range t.Float64s() {
			if dt == F64 {
				body = binary.LittleEndian.AppendUint64(body, math.Float64bits(v))
			} else {
				body = binary.LittleEndian.AppendUint32(body, math.Float32bits(float32(v)))
			}
		}
		shape := t.Shape()
		if shape == nil {
			shape = []int{}
		}
		header[name] = headerEntry{DType: string(dt), Shape: shape, DataOffsets: [2]int{start, start + t.Numel()*size}}
	}

	h, err := json.Marshal(header)
	if err != nil {
		return nil, err
	}
	// Pad the header with spaces so the data starts 8-byte aligned.
	if pad := (8 - len(h)%8) % 8; pad > 0 {
		h = append(h, bytes.Repeat([]byte(" "), pad)...)
	}
	out := binary.LittleEndian.AppendUint64(nil, uint64(len(h)))
	out = append(out, h...)
	return append(out, body...), nil
}
