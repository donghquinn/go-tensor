package tensor

import (
	"fmt"

	"github.com/donghquinn/go-tensor/internal/backend"
)

func convGeom(op string, shape []int, kh, kw, stride, padding int) backend.ConvGeom {
	if len(shape) != 4 {
		panic(fmt.Sprintf("tensor: %s expects input of shape (N, C, H, W), got %v", op, shape))
	}
	if stride < 1 || padding < 0 {
		panic(fmt.Sprintf("tensor: %s needs stride >= 1 and padding >= 0, got %d and %d", op, stride, padding))
	}
	g := backend.ConvGeom{N: shape[0], C: shape[1], H: shape[2], W: shape[3], KH: kh, KW: kw, Stride: stride, Padding: padding}
	g.OH = (g.H+2*padding-kh)/stride + 1
	g.OW = (g.W+2*padding-kw)/stride + 1
	if g.OH <= 0 || g.OW <= 0 {
		panic(fmt.Sprintf("tensor: %s kernel %dx%d is larger than padded input %v", op, kh, kw, shape))
	}
	return g
}

// im2col unrolls every receptive field of x into a row: (N*OH*OW, C*KH*KW).
func (t *Tensor) im2col(g backend.ConvGeom) *Tensor {
	b := t.be()
	shape := []int{g.N * g.OH * g.OW, g.C * g.KH * g.KW}
	return newResult(b.Im2col(t.st, g), shape, "Im2col", []*Tensor{t}, func(grad *Tensor) []*Tensor {
		return []*Tensor{fromStorage(b.Col2im(grad.st, g), t.shape)}
	})
}

// Conv2d applies a 2-D convolution (cross-correlation, as in PyTorch) to x of
// shape (N, C, H, W) with weight of shape (O, C, KH, KW) and an optional bias
// of shape (O). The result has shape (N, O, OH, OW).
//
// It is computed as im2col followed by a matrix product, so it runs on any
// device and uses the device's fastest matrix multiplication.
func Conv2d(x, weight, bias *Tensor, stride, padding int) *Tensor {
	checkSame("Conv2d", x, weight)
	if len(weight.shape) != 4 || len(x.shape) != 4 || weight.shape[1] != x.shape[1] {
		panic(fmt.Sprintf("tensor: Conv2d weight %v does not match input %v", weight.shape, x.shape))
	}
	o := weight.shape[0]
	if bias != nil && (len(bias.shape) != 1 || bias.shape[0] != o) {
		panic(fmt.Sprintf("tensor: Conv2d bias shape %v, want [%d]", bias.shape, o))
	}
	g := convGeom("Conv2d", x.shape, weight.shape[2], weight.shape[3], stride, padding)
	k := g.C * g.KH * g.KW

	cols := x.im2col(g)                        // (N*L, K)
	y := cols.MatMul(weight.Reshape(o, k).T()) // (N*L, O)
	out := y.Reshape(g.N, g.OH*g.OW, o).Permute(0, 2, 1).Reshape(g.N, o, g.OH, g.OW)
	if bias != nil {
		out = out.Add(bias.Reshape(o, 1, 1))
	}
	return out
}

// MaxPool2d takes the maximum over kernel×kernel windows of x, which has shape
// (N, C, H, W), moving the window by stride.
func MaxPool2d(x *Tensor, kernel, stride int) *Tensor {
	g := convGeom("MaxPool2d", x.shape, kernel, kernel, stride, 0)
	b := x.be()
	shape := []int{g.N, g.C, g.OH, g.OW}
	return newResult(b.MaxPool2d(x.st, g), shape, "MaxPool2d", []*Tensor{x}, func(grad *Tensor) []*Tensor {
		return []*Tensor{fromStorage(b.MaxPool2dBackward(x.st, grad.st, g), x.shape)}
	})
}
