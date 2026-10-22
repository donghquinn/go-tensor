// Package weights loads pretrained weights from common model file formats into
// a name → tensor map that nn.LoadStateDict can apply to a model.
//
// Supported formats, detected from the file contents:
//
//   - PyTorch checkpoints (.pth, .pt, .bin) and TorchScript archives — package pth
//   - safetensors (.safetensors) — package safetensors
//   - ONNX initializers (.onnx), including external data — package onnx
//   - NumPy arrays and archives (.npy, .npz) — package numpy
//   - GGUF (.gguf), including quantized tensors — package gguf
//
// Every format is decoded to float64. No format can execute code while loading.
package weights

import (
	"archive/zip"
	"bytes"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/donghquinn/go-tensor/tensor"
	"github.com/donghquinn/go-tensor/weights/gguf"
	"github.com/donghquinn/go-tensor/weights/numpy"
	"github.com/donghquinn/go-tensor/weights/onnx"
	"github.com/donghquinn/go-tensor/weights/pth"
	"github.com/donghquinn/go-tensor/weights/safetensors"
)

// Format identifies a weight file format.
type Format string

const (
	PyTorch     Format = "pytorch"
	Safetensors Format = "safetensors"
	ONNX        Format = "onnx"
	NPY         Format = "npy"
	NPZ         Format = "npz"
	GGUF        Format = "gguf"
)

// Load detects the format of the file at path and reads all of its tensors.
// A file holding a single unnamed array (.npy, or torch.save of one tensor)
// yields the key "".
func Load(path string) (map[string]*tensor.Tensor, error) {
	f, err := Detect(path)
	if err != nil {
		return nil, err
	}
	switch f {
	case PyTorch:
		return pth.Load(path)
	case Safetensors:
		return safetensors.Load(path)
	case ONNX:
		return onnx.Load(path)
	case NPZ:
		return numpy.LoadNpz(path)
	case GGUF:
		return gguf.Load(path)
	case NPY:
		t, err := numpy.LoadNpy(path)
		if err != nil {
			return nil, err
		}
		return map[string]*tensor.Tensor{"": t}, nil
	}
	return nil, fmt.Errorf("weights: unhandled format %q", f)
}

// Detect identifies the format of the file at path from its first bytes,
// falling back to the file extension for ONNX, which has no magic number.
func Detect(path string) (Format, error) {
	fh, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer fh.Close()
	head := make([]byte, 16)
	n, err := io.ReadFull(fh, head)
	if err != nil && err != io.ErrUnexpectedEOF {
		return "", fmt.Errorf("weights: %s: %w", path, err)
	}
	head = head[:n]

	switch {
	case bytes.HasPrefix(head, []byte("GGUF")):
		return GGUF, nil
	case bytes.HasPrefix(head, []byte("\x93NUMPY")):
		return NPY, nil
	case bytes.HasPrefix(head, []byte("PK\x03\x04")):
		return detectZip(path)
	case len(head) > 8 && head[8] == '{':
		// Checked before pickle: the 8-byte header length may start with 0x80.
		return Safetensors, nil
	case bytes.HasPrefix(head, []byte{0x80}):
		return PyTorch, nil // legacy torch.save pickle; pth reports a clear error
	case strings.EqualFold(filepath.Ext(path), ".onnx"):
		return ONNX, nil
	}
	return "", fmt.Errorf("weights: %s: unrecognized file format", path)
}

// detectZip tells PyTorch archives (which contain data.pkl) from .npz files.
func detectZip(path string) (Format, error) {
	zr, err := zip.OpenReader(path)
	if err != nil {
		return "", fmt.Errorf("weights: %s: %w", path, err)
	}
	defer zr.Close()
	for _, f := range zr.File {
		switch {
		case strings.HasSuffix(f.Name, "data.pkl"):
			return PyTorch, nil
		case strings.HasSuffix(f.Name, ".npy"):
			return NPZ, nil
		}
	}
	return "", fmt.Errorf("weights: %s: zip archive is neither a PyTorch checkpoint nor an .npz file", path)
}
