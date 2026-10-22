package mnist

import (
	"bytes"
	"encoding/binary"
	"testing"
)

func TestReadIDX(t *testing.T) {
	var img bytes.Buffer
	binary.Write(&img, binary.BigEndian, []uint32{imagesMagic, 2, 2, 3})
	img.Write([]byte{0, 255, 51, 0, 0, 0, 102, 0, 0, 0, 0, 255})
	x, err := ReadImages(&img)
	if err != nil {
		t.Fatal(err)
	}
	if s := x.Shape(); len(s) != 4 || s[0] != 2 || s[1] != 1 || s[2] != 2 || s[3] != 3 {
		t.Fatalf("shape = %v, want [2 1 2 3]", s)
	}
	if x.At(0, 0, 0, 1) != 1 || x.At(0, 0, 0, 2) != float64(float32(0.2)) || x.At(1, 0, 1, 2) != 1 {
		t.Fatalf("unexpected pixels %v", x)
	}

	var lbl bytes.Buffer
	binary.Write(&lbl, binary.BigEndian, []uint32{labelsMagic, 2})
	lbl.Write([]byte{7, 3})
	y, err := ReadLabels(&lbl)
	if err != nil {
		t.Fatal(err)
	}
	if len(y) != 2 || y[0] != 7 || y[1] != 3 {
		t.Fatalf("labels = %v, want [7 3]", y)
	}

	if _, err := ReadLabels(bytes.NewReader(make([]byte, 8))); err == nil {
		t.Fatal("expected error for bad magic")
	}
}
