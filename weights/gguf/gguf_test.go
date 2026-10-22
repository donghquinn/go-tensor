package gguf

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/donghquinn/go-tensor/internal/weighttest"
)

func TestLoadZoo(t *testing.T) {
	got, meta, err := LoadWithMetadata(weighttest.Testdata("zoo.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	weighttest.Check(t, got, weighttest.Expected(t, "gguf"), 0)

	if meta["zoo.answer"] != uint32(42) || meta["zoo.name"] != "zoo" || meta["general.architecture"] != "go-tensor-test" {
		t.Fatalf("metadata = %v", meta)
	}
	if l, ok := meta["zoo.list"].([]any); !ok || len(l) != 3 || l[2] != int32(3) {
		t.Fatalf("zoo.list = %#v", meta["zoo.list"])
	}
}

func TestTruncatedFile(t *testing.T) {
	b, err := os.ReadFile(weighttest.Testdata("zoo.gguf"))
	if err != nil {
		t.Fatal(err)
	}
	// The last tensor (q6_k) holds 420 bytes, so len-300 cuts into its data.
	for _, n := range []int{3, 30, 200, len(b) - 300} {
		path := filepath.Join(t.TempDir(), "cut.gguf")
		os.WriteFile(path, b[:n], 0o644)
		if _, err := Load(path); err == nil {
			t.Errorf("file truncated to %d bytes was accepted", n)
		}
	}
}
