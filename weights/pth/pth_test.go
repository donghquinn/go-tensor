package pth

import (
	"bytes"
	"math/big"
	"os"
	"path/filepath"
	"testing"

	"github.com/donghquinn/go-tensor/internal/weighttest"
)

func TestLoadZoo(t *testing.T) {
	got, err := Load(weighttest.Testdata("zoo.pth"))
	if err != nil {
		t.Fatal(err)
	}
	weighttest.Check(t, got, weighttest.Expected(t, "pth"), 0)
}

func TestLoadTorchScript(t *testing.T) {
	got, err := Load(weighttest.Testdata("scripted.pt"))
	if err != nil {
		t.Fatal(err)
	}
	weighttest.Check(t, got, weighttest.Expected(t, "torchscript"), 0)
}

func TestLoadObjectKeepsNonTensorValues(t *testing.T) {
	obj, err := LoadObject(weighttest.Testdata("zoo.pth"))
	if err != nil {
		t.Fatal(err)
	}
	d, ok := obj.(*Dict)
	if !ok {
		t.Fatalf("top-level object is %T, want *Dict", obj)
	}
	if d.Values["epoch"] != int64(7) || d.Values["lr"] != 0.01 || d.Values["name"] != "zoo" {
		t.Fatalf("epoch=%v lr=%v name=%v", d.Values["epoch"], d.Values["lr"], d.Values["name"])
	}
	if want := []any{"model", "epoch", "lr", "history", "name"}; len(d.Keys) != len(want) || d.Keys[0] != "model" {
		t.Fatalf("keys %v, want insertion order %v", d.Keys, want)
	}
}

func TestUnknownGlobalsAreNotExecuted(t *testing.T) {
	// pickle.dumps(os.system("echo pwned")) would REDUCE os.system; it must
	// come back as an inert Object instead.
	p := []byte("\x80\x02cos\nsystem\nq\x00X\x0a\x00\x00\x00echo pwnedq\x01\x85q\x02Rq\x03.")
	v, err := unpickle(bytes.NewReader(p), nil, reduce)
	if err != nil {
		t.Fatal(err)
	}
	o, ok := v.(*Object)
	if !ok || o.Class != (Global{"os", "system"}) || o.Args[0] != "echo pwned" {
		t.Fatalf("got %#v, want inert os.system Object", v)
	}
}

func TestDecodeLong(t *testing.T) {
	cases := []struct {
		in   []byte
		want int64
	}{
		{nil, 0},
		{[]byte{0xff}, -1},
		{[]byte{0x00, 0x80}, -32768},
		{[]byte{0x00, 0x80, 0x00}, 32768},
		{[]byte{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80}, -1 << 63},
	}
	for _, c := range cases {
		if got := decodeLong(c.in); got != c.want {
			t.Errorf("decodeLong(%x) = %v, want %d", c.in, got, c.want)
		}
	}
	big9 := decodeLong([]byte{0, 0, 0, 0, 0, 0, 0, 0, 1})
	if b, ok := big9.(*big.Int); !ok || b.Cmp(new(big.Int).Lsh(big.NewInt(1), 64)) != 0 {
		t.Fatalf("decodeLong 9 bytes = %v, want 2^64", big9)
	}
}

func TestRejectsNonCheckpoints(t *testing.T) {
	dir := t.TempDir()
	legacy := filepath.Join(dir, "legacy.pth")
	os.WriteFile(legacy, []byte("\x80\x02\x8a\x0a"), 0o644)
	if _, err := Load(legacy); err == nil || !bytes.Contains([]byte(err.Error()), []byte("legacy")) {
		t.Fatalf("legacy file error = %v", err)
	}
	if _, err := Load(weighttest.Testdata("zoo.npz")); err == nil {
		t.Fatal("expected error for a zip without data.pkl")
	}
}

func FuzzUnpickle(f *testing.F) {
	f.Add([]byte("\x80\x02}q\x00(X\x01\x00\x00\x00aK\x01u."))
	f.Add([]byte("\x80\x02cos\nsystem\nq\x00X\x02\x00\x00\x00lsq\x01\x85q\x02Rq\x03."))
	f.Fuzz(func(t *testing.T, b []byte) {
		unpickle(bytes.NewReader(b), func(any) (any, error) { return &storage{data: make([]float64, 4)}, nil }, reduce)
	})
}
