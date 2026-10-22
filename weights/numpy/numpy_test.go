package numpy

import (
	"path/filepath"
	"testing"

	"github.com/donghquinn/go-tensor/internal/weighttest"
	"github.com/donghquinn/go-tensor/tensor"
)

func TestLoadNpz(t *testing.T) {
	got, err := LoadNpz(weighttest.Testdata("zoo.npz"))
	if err != nil {
		t.Fatal(err)
	}
	weighttest.Check(t, got, weighttest.Expected(t, "npz"), 0)
}

func TestLoadFortranNpy(t *testing.T) {
	got, err := LoadNpy(weighttest.Testdata("fortran.npy"))
	if err != nil {
		t.Fatal(err)
	}
	weighttest.Check(t, map[string]*tensor.Tensor{"fortran": got}, weighttest.Expected(t, "npy"), 0)
}

func TestRoundTrip(t *testing.T) {
	in := map[string]*tensor.Tensor{
		"vec":    tensor.New([]float64{1.0 / 3, -2}, 2),
		"mat":    tensor.Randn(3, 4),
		"scalar": tensor.Scalar(9),
	}
	path := filepath.Join(t.TempDir(), "rt.npz")
	if err := SaveNpz(path, in); err != nil {
		t.Fatal(err)
	}
	out, err := LoadNpz(path)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]weighttest.Entry{}
	for k, v := range in {
		want[k] = weighttest.Entry{Shape: v.Shape(), Values: v.Float64s()}
	}
	weighttest.Check(t, out, want, 0)
}
