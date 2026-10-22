// Package weighttest checks loaded weights against the values recorded by
// weights/testdata/gen_fixtures.py.
package weighttest

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	"github.com/donghquinn/go-tensor/tensor"
)

// Entry is one expected tensor.
type Entry struct {
	Shape  []int     `json:"shape"`
	Values []float64 `json:"values"`
}

// Testdata returns the path of a file in weights/testdata.
func Testdata(name string) string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "weights", "testdata", name)
}

// Expected returns the expected tensors for one section of zoo_expected.json.
func Expected(t *testing.T, section string) map[string]Entry {
	t.Helper()
	var all map[string]map[string]Entry
	b, err := os.ReadFile(Testdata("zoo_expected.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &all); err != nil {
		t.Fatal(err)
	}
	want, ok := all[section]
	if !ok {
		t.Fatalf("no section %q in zoo_expected.json", section)
	}
	return want
}

// Check asserts that got holds exactly the tensors in want, with values equal
// to within a relative tolerance of tol.
func Check(t *testing.T, got map[string]*tensor.Tensor, want map[string]Entry, tol float64) {
	t.Helper()
	for name, w := range want {
		g, ok := got[name]
		if !ok {
			t.Errorf("missing tensor %q", name)
			continue
		}
		if !slices.Equal(g.Shape(), w.Shape) && !(len(g.Shape()) == 0 && len(w.Shape) == 0) {
			t.Errorf("%s: shape %v, want %v", name, g.Shape(), w.Shape)
			continue
		}
		for i, v := range g.Data() {
			if d := math.Abs(v - w.Values[i]); d > tol*math.Max(1, math.Abs(w.Values[i])) {
				t.Errorf("%s[%d] = %v, want %v", name, i, v, w.Values[i])
				break
			}
		}
	}
	for name := range got {
		if _, ok := want[name]; !ok {
			t.Errorf("unexpected tensor %q", name)
		}
	}
}
