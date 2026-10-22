// Package mnist loads the MNIST handwritten digit dataset, like
// torchvision.datasets.MNIST.
package mnist

import (
	"compress/gzip"
	"crypto/md5"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/donghquinn/go-tensor/data"
	"github.com/donghquinn/go-tensor/tensor"
)

// BaseURL is the mirror the dataset files are downloaded from.
var BaseURL = "https://ossci-datasets.s3.amazonaws.com/mnist/"

var files = []struct{ name, md5 string }{
	{"train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"},
	{"train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"},
	{"t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"},
	{"t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"},
}

const (
	imagesMagic = 0x00000803
	labelsMagic = 0x00000801
)

// Load reads the training (60,000 images) or test (10,000 images) split from
// root. Images have shape (N, 1, 28, 28) with pixel values scaled to [0, 1].
// If download is true, missing files are fetched from BaseURL first.
func Load(root string, train, download bool) (*data.TensorDataset, error) {
	if download {
		if err := Download(root); err != nil {
			return nil, err
		}
	}
	prefix := "t10k"
	if train {
		prefix = "train"
	}
	x, err := readGzip(filepath.Join(root, prefix+"-images-idx3-ubyte.gz"), ReadImages)
	if err != nil {
		return nil, err
	}
	y, err := readGzip(filepath.Join(root, prefix+"-labels-idx1-ubyte.gz"), ReadLabels)
	if err != nil {
		return nil, err
	}
	if x.Shape()[0] != len(y) {
		return nil, fmt.Errorf("mnist: %d images but %d labels", x.Shape()[0], len(y))
	}
	return data.NewTensorDataset(x, y), nil
}

// Download fetches any dataset files missing from root and verifies their
// checksums.
func Download(root string) error {
	if err := os.MkdirAll(root, 0o755); err != nil {
		return err
	}
	for _, f := range files {
		path := filepath.Join(root, f.name)
		if _, err := os.Stat(path); err == nil {
			continue
		}
		if err := fetch(BaseURL+f.name, path, f.md5); err != nil {
			return err
		}
	}
	return nil
}

func fetch(url, path, wantMD5 string) error {
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("mnist: download %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("mnist: download %s: %s", url, resp.Status)
	}

	tmp, err := os.CreateTemp(filepath.Dir(path), ".download-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())

	h := md5.New()
	_, err = io.Copy(io.MultiWriter(tmp, h), resp.Body)
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return fmt.Errorf("mnist: download %s: %w", url, err)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != wantMD5 {
		return fmt.Errorf("mnist: checksum mismatch for %s: got %s, want %s", url, got, wantMD5)
	}
	return os.Rename(tmp.Name(), path)
}

func readGzip[T any](path string, read func(io.Reader) (T, error)) (T, error) {
	var zero T
	f, err := os.Open(path)
	if err != nil {
		return zero, err
	}
	defer f.Close()
	zr, err := gzip.NewReader(f)
	if err != nil {
		return zero, fmt.Errorf("mnist: %s: %w", path, err)
	}
	v, err := read(zr)
	if err != nil {
		return zero, fmt.Errorf("mnist: %s: %w", path, err)
	}
	return v, nil
}

// ReadImages parses an uncompressed IDX image file into a tensor of shape
// (N, 1, rows, cols) with values in [0, 1].
func ReadImages(r io.Reader) (*tensor.Tensor, error) {
	var hdr [4]uint32
	if err := binary.Read(r, binary.BigEndian, &hdr); err != nil {
		return nil, err
	}
	if hdr[0] != imagesMagic {
		return nil, fmt.Errorf("bad image file magic %#x", hdr[0])
	}
	n, rows, cols := int(hdr[1]), int(hdr[2]), int(hdr[3])
	raw := make([]byte, n*rows*cols)
	if _, err := io.ReadFull(r, raw); err != nil {
		return nil, err
	}
	px := make([]float64, len(raw))
	for i, b := range raw {
		px[i] = float64(b) / 255
	}
	return tensor.New(px, n, 1, rows, cols), nil
}

// ReadLabels parses an uncompressed IDX label file.
func ReadLabels(r io.Reader) ([]int, error) {
	var hdr [2]uint32
	if err := binary.Read(r, binary.BigEndian, &hdr); err != nil {
		return nil, err
	}
	if hdr[0] != labelsMagic {
		return nil, fmt.Errorf("bad label file magic %#x", hdr[0])
	}
	raw := make([]byte, hdr[1])
	if _, err := io.ReadFull(r, raw); err != nil {
		return nil, err
	}
	labels := make([]int, len(raw))
	for i, b := range raw {
		labels[i] = int(b)
	}
	return labels, nil
}
