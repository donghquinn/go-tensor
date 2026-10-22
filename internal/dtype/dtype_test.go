package dtype

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestHalf(t *testing.T) {
	cases := map[uint16]float64{
		0x0000: 0,
		0x3c00: 1,
		0xc000: -2,
		0x3555: 0.333251953125,
		0x7bff: 65504,              // largest normal
		0x0001: math.Ldexp(1, -24), // smallest subnormal
		0x0400: math.Ldexp(1, -14), // smallest normal
	}
	for bits, want := range cases {
		if got := HalfToFloat64(bits); got != want {
			t.Errorf("HalfToFloat64(%#04x) = %v, want %v", bits, got, want)
		}
	}
	if !math.IsInf(HalfToFloat64(0xfc00), -1) || !math.IsNaN(HalfToFloat64(0x7e00)) {
		t.Error("half infinity/NaN not decoded")
	}
}

func TestBFloat16(t *testing.T) {
	if got := BFloat16ToFloat64(0x3fc0); got != 1.5 {
		t.Fatalf("BFloat16ToFloat64(0x3fc0) = %v, want 1.5", got)
	}
}

func TestDecode(t *testing.T) {
	b := []byte{0xff, 0xff, 0x02, 0x00}
	got, err := Decode(Int16, b, binary.LittleEndian)
	if err != nil || got[0] != -1 || got[1] != 2 {
		t.Fatalf("Decode int16 = %v, %v", got, err)
	}
	got, _ = Decode(Int16, b, binary.BigEndian)
	if got[0] != -1 || got[1] != 512 {
		t.Fatalf("Decode int16 big-endian = %v", got)
	}
	if _, err := Decode(Float32, b[:3], binary.LittleEndian); err == nil {
		t.Fatal("expected error for truncated data")
	}
}

func TestNumel(t *testing.T) {
	if n, err := Numel([]int{2, 3, 4}); n != 24 || err != nil {
		t.Fatalf("Numel = %d, %v", n, err)
	}
	if n, err := Numel(nil); n != 1 || err != nil {
		t.Fatalf("Numel(scalar) = %d, %v", n, err)
	}
	if n, err := Numel([]int{0, 1 << 40}); n != 0 || err != nil {
		t.Fatalf("Numel with zero dim = %d, %v", n, err)
	}
	for _, bad := range [][]int{{-1}, {1 << 32, 1 << 32}, {1 << 20, 1 << 20, 1 << 20}} {
		if _, err := Numel(bad); err == nil {
			t.Errorf("Numel(%v) accepted", bad)
		}
	}
}
