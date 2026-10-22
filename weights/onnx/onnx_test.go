package onnx

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/donghquinn/go-tensor/internal/weighttest"
)

func TestLoadZoo(t *testing.T) {
	got, err := Load(weighttest.Testdata("zoo.onnx"))
	if err != nil {
		t.Fatal(err)
	}
	weighttest.Check(t, got, weighttest.Expected(t, "onnx"), 0)
}

func TestExternalDataMatchesEmbedded(t *testing.T) {
	embedded, err := Load(weighttest.Testdata("cnn.onnx"))
	if err != nil {
		t.Fatal(err)
	}
	external, err := Load(weighttest.Testdata("cnn_external.onnx"))
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]weighttest.Entry{}
	for k, v := range embedded {
		want[k] = weighttest.Entry{Shape: v.Shape(), Values: v.Data()}
	}
	weighttest.Check(t, external, want, 0)
	if len(want) != 4 {
		t.Fatalf("got %d initializers, want 4", len(want))
	}
}

func TestExternalDataCannotEscapeDirectory(t *testing.T) {
	for _, loc := range []string{"../secret", "/etc/passwd"} {
		_, err := readExternal(map[string]string{"location": loc}, t.TempDir())
		if err == nil {
			t.Errorf("location %q was accepted", loc)
		}
	}
}

func TestTruncatedModel(t *testing.T) {
	b, err := os.ReadFile(weighttest.Testdata("zoo.onnx"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Decode(b[:len(b)/2], filepath.Dir(weighttest.Testdata("zoo.onnx"))); err == nil {
		t.Fatal("truncated model accepted")
	}
}

func FuzzDecode(f *testing.F) {
	b, _ := os.ReadFile(weighttest.Testdata("zoo.onnx"))
	f.Add(b)
	f.Fuzz(func(t *testing.T, b []byte) { Decode(b, "") })
}
