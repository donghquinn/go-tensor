// Package numpy reads and writes NumPy .npy arrays and .npz archives, the
// format used by np.save/np.savez and by many JAX and Keras checkpoints.
package numpy

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/donghquinn/go-tensor/internal/dtype"
	"github.com/donghquinn/go-tensor/tensor"
)

const magic = "\x93NUMPY"

var kinds = map[string]dtype.Kind{
	"f8": dtype.Float64, "f4": dtype.Float32, "f2": dtype.Float16,
	"i8": dtype.Int64, "i4": dtype.Int32, "i2": dtype.Int16, "i1": dtype.Int8,
	"u8": dtype.Uint64, "u4": dtype.Uint32, "u2": dtype.Uint16, "u1": dtype.Uint8,
	"b1": dtype.Bool,
}

var (
	descrRe   = regexp.MustCompile(`'descr'\s*:\s*'([<>|=])([a-z]\d+)'`)
	fortranRe = regexp.MustCompile(`'fortran_order'\s*:\s*(True|False)`)
	shapeRe   = regexp.MustCompile(`'shape'\s*:\s*\(([^)]*)\)`)
)

// LoadNpy reads a single .npy file.
func LoadNpy(path string) (*tensor.Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	t, err := ReadNpy(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return t, nil
}

// ReadNpy parses an .npy stream.
func ReadNpy(r io.Reader) (*tensor.Tensor, error) {
	pre := make([]byte, len(magic)+2)
	if _, err := io.ReadFull(r, pre); err != nil {
		return nil, fmt.Errorf("numpy: %w", err)
	}
	if string(pre[:len(magic)]) != magic {
		return nil, errors.New("numpy: not an .npy file")
	}
	var hlen int
	switch major := pre[len(magic)]; major {
	case 1:
		var n uint16
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return nil, err
		}
		hlen = int(n)
	case 2, 3:
		var n uint32
		if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
			return nil, err
		}
		hlen = int(n)
	default:
		return nil, fmt.Errorf("numpy: unsupported .npy version %d", major)
	}
	if hlen > 1<<20 {
		return nil, fmt.Errorf("numpy: header of %d bytes is too large", hlen)
	}
	header := make([]byte, hlen)
	if _, err := io.ReadFull(r, header); err != nil {
		return nil, err
	}

	m := descrRe.FindSubmatch(header)
	if m == nil {
		return nil, fmt.Errorf("numpy: unsupported dtype in header %q", header)
	}
	kind, ok := kinds[string(m[2])]
	if !ok {
		return nil, fmt.Errorf("numpy: unsupported dtype %s", m[2])
	}
	var order binary.ByteOrder = binary.LittleEndian
	if m[1][0] == '>' {
		order = binary.BigEndian
	}
	fm := fortranRe.FindSubmatch(header)
	sm := shapeRe.FindSubmatch(header)
	if fm == nil || sm == nil {
		return nil, fmt.Errorf("numpy: malformed header %q", header)
	}
	var shape []int
	for _, s := range strings.Split(string(sm[1]), ",") {
		if s = strings.TrimSpace(s); s == "" {
			continue
		}
		d, err := strconv.Atoi(s)
		if err != nil || d < 0 {
			return nil, fmt.Errorf("numpy: bad shape %q", sm[1])
		}
		shape = append(shape, d)
	}
	numel, err := dtype.Numel(shape)
	if err != nil {
		return nil, fmt.Errorf("numpy: %w", err)
	}

	// ReadAll only allocates as much as the stream really holds.
	want := int64(numel * kind.Size())
	raw, err := io.ReadAll(io.LimitReader(r, want))
	if err != nil {
		return nil, fmt.Errorf("numpy: data: %w", err)
	}
	if int64(len(raw)) != want {
		return nil, fmt.Errorf("numpy: data: %w", io.ErrUnexpectedEOF)
	}
	data, err := dtype.Decode(kind, raw, order)
	if err != nil {
		return nil, err
	}
	if len(shape) == 0 {
		return tensor.Scalar(data[0]), nil
	}
	if string(fm[1]) == "True" {
		data = fromFortran(data, shape)
	}
	return tensor.New(data, shape...), nil
}

// fromFortran reorders column-major data into row-major order.
func fromFortran(data []float64, shape []int) []float64 {
	out := make([]float64, len(data))
	// Column-major strides: the first dimension varies fastest.
	strides := make([]int, len(shape))
	s := 1
	for d := range shape {
		strides[d] = s
		s *= shape[d]
	}
	idx := make([]int, len(shape))
	for k := range out {
		src := 0
		for d, i := range idx {
			src += i * strides[d]
		}
		out[k] = data[src]
		for d := len(idx) - 1; d >= 0; d-- {
			idx[d]++
			if idx[d] < shape[d] {
				break
			}
			idx[d] = 0
		}
	}
	return out
}

// LoadNpz reads every array in an .npz archive (compressed or not), keyed by
// the name passed to np.savez.
func LoadNpz(path string) (map[string]*tensor.Tensor, error) {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("numpy: %s: %w", path, err)
	}
	defer zr.Close()
	out := map[string]*tensor.Tensor{}
	for _, f := range zr.File {
		name, ok := strings.CutSuffix(f.Name, ".npy")
		if !ok {
			continue
		}
		r, err := f.Open()
		if err != nil {
			return nil, err
		}
		t, err := ReadNpy(r)
		r.Close()
		if err != nil {
			return nil, fmt.Errorf("%s: %s: %w", path, f.Name, err)
		}
		out[name] = t
	}
	return out, nil
}

// WriteNpy writes t as a little-endian float64 .npy stream.
func WriteNpy(w io.Writer, t *tensor.Tensor) error {
	dims := make([]string, 0, t.Dim())
	for _, d := range t.Shape() {
		dims = append(dims, strconv.Itoa(d))
	}
	shape := strings.Join(dims, ", ")
	if len(dims) == 1 {
		shape += ","
	}
	header := fmt.Sprintf("{'descr': '<f8', 'fortran_order': False, 'shape': (%s), }", shape)
	// Pad with spaces and a newline so the data is 64-byte aligned.
	total := len(magic) + 4 + len(header) + 1
	header += strings.Repeat(" ", (64-total%64)%64) + "\n"

	var buf bytes.Buffer
	buf.WriteString(magic)
	buf.Write([]byte{1, 0})
	binary.Write(&buf, binary.LittleEndian, uint16(len(header)))
	buf.WriteString(header)
	binary.Write(&buf, binary.LittleEndian, t.Data())
	_, err := w.Write(buf.Bytes())
	return err
}

// SaveNpz writes tensors to an uncompressed .npz archive readable by np.load.
func SaveNpz(path string, tensors map[string]*tensor.Tensor) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	zw := zip.NewWriter(f)
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		w, err := zw.CreateHeader(&zip.FileHeader{Name: name + ".npy", Method: zip.Store})
		if err == nil {
			err = WriteNpy(w, tensors[name])
		}
		if err != nil {
			f.Close()
			return err
		}
	}
	if err := zw.Close(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
