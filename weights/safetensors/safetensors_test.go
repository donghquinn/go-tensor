package safetensors

import (
	"encoding/binary"
	"path/filepath"
	"testing"

	"github.com/donghquinn/go-tensor/internal/weighttest"
	"github.com/donghquinn/go-tensor/tensor"
)

func TestLoadZoo(t *testing.T) {
	got, meta, err := LoadWithMetadata(weighttest.Testdata("zoo.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	weighttest.Check(t, got, weighttest.Expected(t, "safetensors"), 0)
	if meta["note"] != "zoo" || meta["format"] != "pt" {
		t.Fatalf("metadata = %v", meta)
	}
}

func TestRoundTrip(t *testing.T) {
	in := map[string]*tensor.Tensor{
		"w":      tensor.New([]float64{1.0 / 3, -2, 1e-40}, 3),
		"m":      tensor.Randn(2, 3),
		"scalar": tensor.Scalar(4),
	}
	for _, dt := range []DType{F64, F32} {
		path := filepath.Join(t.TempDir(), "rt.safetensors")
		if err := Save(path, in, dt, map[string]string{"k": "v"}); err != nil {
			t.Fatal(err)
		}
		out, meta, err := LoadWithMetadata(path)
		if err != nil {
			t.Fatal(err)
		}
		if meta["k"] != "v" {
			t.Fatalf("%s: metadata %v", dt, meta)
		}
		want := map[string]weighttest.Entry{}
		for k, v := range in {
			e := weighttest.Entry{Shape: v.Shape(), Values: append([]float64(nil), v.Float64s()...)}
			if dt == F32 {
				for i, x := range e.Values {
					e.Values[i] = float64(float32(x))
				}
			}
			want[k] = e
		}
		weighttest.Check(t, out, want, 0)
	}
}

func TestRejectsCorruptFiles(t *testing.T) {
	good, err := Encode(map[string]*tensor.Tensor{"a": tensor.Ones(4)}, F32, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := Decode(good[:len(good)-1]); err == nil {
		t.Error("truncated body accepted")
	}
	huge := append([]byte(nil), good...)
	binary.LittleEndian.PutUint64(huge, 1<<40)
	if _, _, err := Decode(huge); err == nil {
		t.Error("oversized header length accepted")
	}
	if _, _, err := Decode([]byte{1, 2}); err == nil {
		t.Error("short file accepted")
	}
}

func FuzzDecode(f *testing.F) {
	good, _ := Encode(map[string]*tensor.Tensor{"a": tensor.Ones(2, 2)}, F32, map[string]string{"k": "v"})
	f.Add(good)
	f.Fuzz(func(t *testing.T, b []byte) { Decode(b) })
}
