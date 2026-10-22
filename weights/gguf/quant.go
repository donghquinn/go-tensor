package gguf

import (
	"encoding/binary"

	"github.com/donghquinn/go-tensor/internal/dtype"
)

// Block dequantizers, ported from ggml's dequantize_row_* reference code.
// Arithmetic is float32 like ggml's, and products are explicitly rounded with
// float32(...) so the compiler cannot fuse them into FMA instructions.

func f16(b []byte) float32 { return float32(dtype.HalfToFloat64(binary.LittleEndian.Uint16(b))) }

// Q4_0: f16 scale d, then 16 bytes of 4-bit values offset by 8.
func dequantQ4_0(b []byte, y []float32) {
	d, qs := f16(b), b[2:18]
	for j, q := range qs {
		y[j] = float32(int(q&0x0f)-8) * d
		y[j+16] = float32(int(q>>4)-8) * d
	}
}

// Q4_1: f16 scale d and minimum m, then 16 bytes of 4-bit values.
func dequantQ4_1(b []byte, y []float32) {
	d, m, qs := f16(b), f16(b[2:]), b[4:20]
	for j, q := range qs {
		y[j] = float32(float32(q&0x0f)*d) + m
		y[j+16] = float32(float32(q>>4)*d) + m
	}
}

// Q5_0: f16 scale d, 32 high bits, then 16 bytes of low nibbles; offset by 16.
func dequantQ5_0(b []byte, y []float32) {
	d, qh, qs := f16(b), binary.LittleEndian.Uint32(b[2:]), b[6:22]
	for j, q := range qs {
		h0 := byte((qh>>uint(j))<<4) & 0x10
		h1 := byte(qh>>uint(j+12)) & 0x10
		y[j] = float32(int((q&0x0f)|h0)-16) * d
		y[j+16] = float32(int((q>>4)|h1)-16) * d
	}
}

// Q5_1: f16 scale d and minimum m, 32 high bits, then 16 bytes of low nibbles.
func dequantQ5_1(b []byte, y []float32) {
	d, m, qh, qs := f16(b), f16(b[2:]), binary.LittleEndian.Uint32(b[4:]), b[8:24]
	for j, q := range qs {
		h0 := byte((qh>>uint(j))<<4) & 0x10
		h1 := byte(qh>>uint(j+12)) & 0x10
		y[j] = float32(float32((q&0x0f)|h0)*d) + m
		y[j+16] = float32(float32((q>>4)|h1)*d) + m
	}
}

// Q8_0: f16 scale d, then 32 signed bytes.
func dequantQ8_0(b []byte, y []float32) {
	d := f16(b)
	for j, q := range b[2:34] {
		y[j] = float32(int8(q)) * d
	}
}

// scaleMinK4 unpacks the j-th 6-bit scale and minimum of a Q4_K/Q5_K block.
func scaleMinK4(j int, q []byte) (sc, m float32) {
	if j < 4 {
		return float32(q[j] & 63), float32(q[j+4] & 63)
	}
	return float32((q[j+4] & 0x0f) | ((q[j-4] >> 6) << 4)), float32((q[j+4] >> 4) | ((q[j] >> 6) << 4))
}

// Q4_K: f16 d and dmin, 12 bytes of packed 6-bit scales/mins, 128 bytes of
// 4-bit values; 8 sub-blocks of 32.
func dequantQ4_K(b []byte, y []float32) {
	d, dmin, scales, q := f16(b), f16(b[2:]), b[4:16], b[16:144]
	for j, is := 0, 0; j < 256; j, is = j+64, is+2 {
		sc, m := scaleMinK4(is, scales)
		d1, m1 := float32(d*sc), float32(dmin*m)
		sc, m = scaleMinK4(is+1, scales)
		d2, m2 := float32(d*sc), float32(dmin*m)
		for l := 0; l < 32; l++ {
			y[j+l] = float32(d1*float32(q[l]&0x0f)) - m1
			y[j+32+l] = float32(d2*float32(q[l]>>4)) - m2
		}
		q = q[32:]
	}
}

// Q5_K: like Q4_K with a fifth bit per value stored in 32 bytes of qh.
func dequantQ5_K(b []byte, y []float32) {
	d, dmin, scales, qh, ql := f16(b), f16(b[2:]), b[4:16], b[16:48], b[48:176]
	u1, u2 := byte(1), byte(2)
	for j, is := 0, 0; j < 256; j, is = j+64, is+2 {
		sc, m := scaleMinK4(is, scales)
		d1, m1 := float32(d*sc), float32(dmin*m)
		sc, m = scaleMinK4(is+1, scales)
		d2, m2 := float32(d*sc), float32(dmin*m)
		for l := 0; l < 32; l++ {
			lo, hi := ql[l]&0x0f, ql[l]>>4
			if qh[l]&u1 != 0 {
				lo += 16
			}
			if qh[l]&u2 != 0 {
				hi += 16
			}
			y[j+l] = float32(d1*float32(lo)) - m1
			y[j+32+l] = float32(d2*float32(hi)) - m2
		}
		ql = ql[32:]
		u1 <<= 2
		u2 <<= 2
	}
}

// Q6_K: 128 bytes of low 4 bits, 64 bytes of high 2 bits, 16 signed 8-bit
// scales, then f16 d; values are offset by 32.
func dequantQ6_K(b []byte, y []float32) {
	ql, qh, sc, d := b[0:128], b[128:192], b[192:208], f16(b[208:])
	for n := 0; n < 256; n += 128 {
		for l := 0; l < 32; l++ {
			is := l / 16
			q1 := int((ql[l]&0x0f)|((qh[l]>>0)&3)<<4) - 32
			q2 := int((ql[l+32]&0x0f)|((qh[l]>>2)&3)<<4) - 32
			q3 := int((ql[l]>>4)|((qh[l]>>4)&3)<<4) - 32
			q4 := int((ql[l+32]>>4)|((qh[l]>>6)&3)<<4) - 32
			y[n+l] = float32(d*float32(int8(sc[is]))) * float32(q1)
			y[n+l+32] = float32(d*float32(int8(sc[is+2]))) * float32(q2)
			y[n+l+64] = float32(d*float32(int8(sc[is+4]))) * float32(q3)
			y[n+l+96] = float32(d*float32(int8(sc[is+6]))) * float32(q4)
		}
		ql, qh, sc = ql[64:], qh[32:], sc[8:]
	}
}
