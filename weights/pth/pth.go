// Package pth reads PyTorch checkpoints written by torch.save (.pth, .pt, .bin)
// and TorchScript archives written by torch.jit.save.
//
// Unlike torch.load, reading a file never executes code from it: the pickle
// stream is interpreted as data, and only the tensor-rebuilding functions that
// torch.save emits are recognized.
package pth

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/donghquinn/go-tensor/internal/dtype"
	"github.com/donghquinn/go-tensor/tensor"
)

var storageKinds = map[string]dtype.Kind{
	"DoubleStorage":   dtype.Float64,
	"FloatStorage":    dtype.Float32,
	"HalfStorage":     dtype.Float16,
	"BFloat16Storage": dtype.BFloat16,
	"LongStorage":     dtype.Int64,
	"IntStorage":      dtype.Int32,
	"ShortStorage":    dtype.Int16,
	"CharStorage":     dtype.Int8,
	"ByteStorage":     dtype.Uint8,
	"BoolStorage":     dtype.Bool,
}

type storage struct{ data []float64 }

// Load reads a checkpoint and returns every tensor in it, keyed by its path
// through nested dicts, lists and objects joined with "." — for a state_dict
// this is exactly the parameter name, e.g. "0.weight"; for a checkpoint like
// {"model": state_dict, "epoch": 3} it is "model.0.weight". A file holding a
// single bare tensor yields the key "".
func Load(path string) (map[string]*tensor.Tensor, error) {
	obj, err := LoadObject(path)
	if err != nil {
		return nil, err
	}
	return Tensors(obj), nil
}

// LoadObject reads a checkpoint and returns the unpickled object tree. Values
// are *Dict, *List, Tuple, *tensor.Tensor, string, []byte, bool, int64,
// *big.Int, float64, nil, Global, or *Object for unrecognized classes.
func LoadObject(path string) (any, error) {
	zr, err := zip.OpenReader(path)
	if err != nil {
		if isLegacy(path) {
			return nil, fmt.Errorf("pth: %s uses the legacy (pre-1.6) torch.save format; re-save it with a newer PyTorch", path)
		}
		return nil, fmt.Errorf("pth: %s: %w", path, err)
	}
	defer zr.Close()

	files := map[string]*zip.File{}
	prefix := ""
	for _, f := range zr.File {
		files[f.Name] = f
		if dir, ok := strings.CutSuffix(f.Name, "data.pkl"); ok && strings.Count(dir, "/") <= 1 {
			prefix = dir
		}
	}
	pkl, ok := files[prefix+"data.pkl"]
	if !ok {
		return nil, fmt.Errorf("pth: %s has no data.pkl; not a PyTorch checkpoint", path)
	}

	var order binary.ByteOrder = binary.LittleEndian
	if f, ok := files[prefix+"byteorder"]; ok {
		b, err := readZip(f)
		if err != nil {
			return nil, err
		}
		if strings.TrimSpace(string(b)) == "big" {
			order = binary.BigEndian
		}
	}

	storages := map[string]*storage{}
	loadStorage := func(pid any) (any, error) {
		t, ok := pid.(Tuple)
		if !ok || len(t) < 3 || t[0] != "storage" {
			return nil, fmt.Errorf("pth: unexpected persistent id %v", pid)
		}
		cls, _ := t[1].(Global)
		key, _ := t[2].(string)
		if s, ok := storages[key]; ok {
			return s, nil
		}
		kind, ok := storageKinds[cls.Name]
		if !ok {
			return nil, fmt.Errorf("pth: unsupported storage type %v", cls)
		}
		f, ok := files[prefix+"data/"+key]
		if !ok {
			return nil, fmt.Errorf("pth: storage %q missing from archive", key)
		}
		raw, err := readZip(f)
		if err != nil {
			return nil, err
		}
		data, err := dtype.Decode(kind, raw, order)
		if err != nil {
			return nil, fmt.Errorf("pth: storage %q: %w", key, err)
		}
		s := &storage{data: data}
		storages[key] = s
		return s, nil
	}

	r, err := pkl.Open()
	if err != nil {
		return nil, err
	}
	defer r.Close()
	obj, err := unpickle(r, loadStorage, reduce)
	if err != nil {
		return nil, fmt.Errorf("pth: %s: %w", path, err)
	}
	return obj, nil
}

// reduce evaluates the callables torch.save emits.
func reduce(fn Global, args Tuple) (any, bool, error) {
	switch fn.Module + "." + fn.Name {
	case "torch._utils._rebuild_tensor_v2", "torch._utils._rebuild_tensor":
		t, err := rebuildTensor(args)
		return t, true, err
	case "torch._utils._rebuild_parameter", "torch._utils._rebuild_parameter_with_state":
		if len(args) == 0 {
			return nil, true, errors.New("pth: _rebuild_parameter without data")
		}
		return args[0], true, nil
	case "torch._tensor._rebuild_from_type_v2":
		// (func, type, args, state): rebuild with func(*args), dropping the subclass.
		if len(args) < 3 {
			return nil, true, errors.New("pth: malformed _rebuild_from_type_v2")
		}
		inner, _ := args[0].(Global)
		innerArgs, _ := args[2].(Tuple)
		return reduce(inner, innerArgs)
	case "collections.OrderedDict", "builtins.dict":
		d := newDict()
		for _, item := range asSlice(firstArg(args)) {
			if kv := asSlice(item); len(kv) == 2 {
				d.set(kv[0], kv[1])
			}
		}
		return d, true, nil
	case "torch.Size", "builtins.tuple":
		return Tuple(asSlice(firstArg(args))), true, nil
	}
	return nil, false, nil
}

func firstArg(args Tuple) any {
	if len(args) == 0 {
		return nil
	}
	return args[0]
}

// rebuildTensor implements _rebuild_tensor_v2(storage, offset, size, stride, ...).
func rebuildTensor(args Tuple) (*tensor.Tensor, error) {
	if len(args) < 4 {
		return nil, fmt.Errorf("pth: _rebuild_tensor got %d arguments", len(args))
	}
	s, ok := args[0].(*storage)
	if !ok {
		return nil, fmt.Errorf("pth: tensor storage is %T", args[0])
	}
	offset, ok1 := args[1].(int64)
	size, ok2 := ints(args[2])
	stride, ok3 := ints(args[3])
	if !ok1 || !ok2 || !ok3 || len(size) != len(stride) {
		return nil, fmt.Errorf("pth: malformed tensor metadata %v", args[1:4])
	}

	n, err := dtype.Numel(size)
	if err != nil {
		return nil, fmt.Errorf("pth: %w", err)
	}
	// Check the view fits in its storage before allocating the result.
	if n > 0 {
		last := offset
		for d := range size {
			last += int64(size[d]-1) * int64(stride[d])
		}
		if offset < 0 || last >= int64(len(s.data)) {
			return nil, fmt.Errorf("pth: tensor view (offset %d, size %v, stride %v) exceeds storage of %d elements", offset, size, stride, len(s.data))
		}
	}
	out := make([]float64, n)
	idx := make([]int, len(size))
	for k := range out {
		pos := int(offset)
		for d, i := range idx {
			pos += i * stride[d]
		}
		out[k] = s.data[pos]
		for d := len(idx) - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < size[d] {
				break
			}
			idx[d] = 0
		}
	}
	if len(size) == 0 {
		return tensor.FromFloat64(out[:1]).Reshape(), nil
	}
	return tensor.FromFloat64(out, size...), nil
}

func ints(v any) ([]int, bool) {
	items := asSlice(v)
	out := make([]int, len(items))
	for i, x := range items {
		n, ok := x.(int64)
		if !ok || n < 0 {
			return nil, false
		}
		out[i] = int(n)
	}
	return out, true
}

// Tensors collects every tensor reachable from obj, keyed by its dotted path.
func Tensors(obj any) map[string]*tensor.Tensor {
	out := map[string]*tensor.Tensor{}
	collect("", obj, out, map[any]bool{})
	return out
}

func collect(prefix string, v any, out map[string]*tensor.Tensor, seen map[any]bool) {
	join := func(k string) string {
		if prefix == "" {
			return k
		}
		return prefix + "." + k
	}
	switch v := v.(type) {
	case *tensor.Tensor:
		out[prefix] = v
	case *Dict:
		if seen[v] {
			return
		}
		seen[v] = true
		for _, k := range v.Keys {
			collect(join(keyString(k)), v.Values[k], out, seen)
		}
	case *List:
		if seen[v] {
			return
		}
		seen[v] = true
		for i, x := range v.Items {
			collect(join(strconv.Itoa(i)), x, out, seen)
		}
	case Tuple:
		for i, x := range v {
			collect(join(strconv.Itoa(i)), x, out, seen)
		}
	case *Object:
		if seen[v] {
			return
		}
		seen[v] = true
		collect(prefix, v.State, out, seen)
	}
}

func keyString(k any) string {
	if s, ok := k.(string); ok {
		return s
	}
	return fmt.Sprint(k)
}

func readZip(f *zip.File) ([]byte, error) {
	r, err := f.Open()
	if err != nil {
		return nil, err
	}
	defer r.Close()
	return io.ReadAll(r)
}

// isLegacy reports whether path starts like a pre-1.6 torch.save file: a raw
// pickle stream beginning with the PROTO opcode.
func isLegacy(path string) bool {
	f, err := os.Open(path)
	if err != nil {
		return false
	}
	defer f.Close()
	head := make([]byte, 2)
	if _, err := io.ReadFull(f, head); err != nil {
		return false
	}
	return bytes.Equal(head, []byte{0x80, 0x02})
}
