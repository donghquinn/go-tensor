package nn

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"

	"github.com/donghquinn/go-tensor/tensor"
)

// File layout (little-endian): magic, version, parameter count, then for each
// parameter its rank, dimensions and float64 values. Parameters are matched by
// position in Module.Parameters, like a torch state_dict without names.
const (
	fileMagic   = "GOTN"
	fileVersion = uint32(1)
)

// Save writes the values of m's parameters to w.
func Save(w io.Writer, m Module) error {
	bw := bufio.NewWriter(w)
	params := m.Parameters()
	bw.WriteString(fileMagic)
	write := func(v any) { binary.Write(bw, binary.LittleEndian, v) }
	write(fileVersion)
	write(uint32(len(params)))
	for _, p := range params {
		shape := p.Shape()
		write(uint32(len(shape)))
		for _, s := range shape {
			write(uint32(s))
		}
		write(p.Float64s())
	}
	return bw.Flush()
}

// Load reads parameter values saved by Save into m. The module must have the
// same architecture as the one that was saved.
func Load(r io.Reader, m Module) error {
	br := bufio.NewReader(r)
	magic := make([]byte, len(fileMagic))
	if _, err := io.ReadFull(br, magic); err != nil {
		return fmt.Errorf("nn: read header: %w", err)
	}
	if string(magic) != fileMagic {
		return fmt.Errorf("nn: not a go-tensor model file")
	}
	var version, count uint32
	if err := binary.Read(br, binary.LittleEndian, &version); err != nil {
		return err
	}
	if version != fileVersion {
		return fmt.Errorf("nn: unsupported model file version %d", version)
	}
	if err := binary.Read(br, binary.LittleEndian, &count); err != nil {
		return err
	}
	params := m.Parameters()
	if int(count) != len(params) {
		return fmt.Errorf("nn: file has %d parameters, module has %d", count, len(params))
	}
	for i, p := range params {
		var rank uint32
		if err := binary.Read(br, binary.LittleEndian, &rank); err != nil {
			return err
		}
		dims := make([]uint32, rank)
		if err := binary.Read(br, binary.LittleEndian, dims); err != nil {
			return err
		}
		want := p.Shape()
		match := len(dims) == len(want)
		for j := 0; match && j < len(dims); j++ {
			match = int(dims[j]) == want[j]
		}
		if !match {
			return fmt.Errorf("nn: parameter %d has shape %v in file, %v in module", i, dims, want)
		}
		vals := make([]float64, p.Numel())
		if err := binary.Read(br, binary.LittleEndian, vals); err != nil {
			return fmt.Errorf("nn: parameter %d: %w", i, err)
		}
		p.CopyFrom(tensor.FromFloat64(vals))
	}
	return nil
}

// SaveFile writes m's parameters to the file at path.
func SaveFile(path string, m Module) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	if err := Save(f, m); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// LoadFile reads parameters from the file at path into m.
func LoadFile(path string, m Module) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return Load(f, m)
}
