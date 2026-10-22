package tensor

import (
	"fmt"
	"math"
)

// convGeom describes a 2-D sliding-window operation over a batch of images.
type convGeom struct {
	n, c, h, w      int // input batch, channels, height, width
	kh, kw          int // kernel size
	oh, ow          int // output size
	stride, padding int
}

func newConvGeom(op string, shape []int, kh, kw, stride, padding int) convGeom {
	if len(shape) != 4 {
		panic(fmt.Sprintf("tensor: %s expects input of shape (N, C, H, W), got %v", op, shape))
	}
	if stride < 1 || padding < 0 {
		panic(fmt.Sprintf("tensor: %s needs stride >= 1 and padding >= 0, got %d and %d", op, stride, padding))
	}
	g := convGeom{n: shape[0], c: shape[1], h: shape[2], w: shape[3], kh: kh, kw: kw, stride: stride, padding: padding}
	g.oh = (g.h+2*padding-kh)/stride + 1
	g.ow = (g.w+2*padding-kw)/stride + 1
	if g.oh <= 0 || g.ow <= 0 {
		panic(fmt.Sprintf("tensor: %s kernel %dx%d is larger than padded input %v", op, kh, kw, shape))
	}
	return g
}

// patches returns the number of output positions per image and the number of
// values in each receptive field.
func (g convGeom) patches() (l, k int) { return g.oh * g.ow, g.c * g.kh * g.kw }

// im2col unrolls every receptive field of x into a row, giving a matrix of
// shape (N*OH*OW, C*KH*KW). Out-of-bounds (padded) positions are zero.
func (g convGeom) im2col(x []float64) []float64 {
	l, k := g.patches()
	cols := make([]float64, g.n*l*k)
	g.eachPatch(func(row, xOff int) {
		cols[row] = x[xOff]
	})
	return cols
}

// col2im is the adjoint of im2col: it sums each row entry back into the input
// position it was read from.
func (g convGeom) col2im(cols []float64) []float64 {
	x := make([]float64, g.n*g.c*g.h*g.w)
	g.eachPatch(func(row, xOff int) {
		x[xOff] += cols[row]
	})
	return x
}

// eachPatch calls fn with the flat im2col index and the flat input index for
// every in-bounds element of every receptive field.
func (g convGeom) eachPatch(fn func(colIdx, xIdx int)) {
	_, k := g.patches()
	row := 0
	for n := 0; n < g.n; n++ {
		for oh := 0; oh < g.oh; oh++ {
			for ow := 0; ow < g.ow; ow++ {
				idx := row * k
				for c := 0; c < g.c; c++ {
					plane := (n*g.c + c) * g.h
					for kh := 0; kh < g.kh; kh++ {
						ih := oh*g.stride - g.padding + kh
						for kw := 0; kw < g.kw; kw++ {
							iw := ow*g.stride - g.padding + kw
							if ih >= 0 && ih < g.h && iw >= 0 && iw < g.w {
								fn(idx, (plane+ih)*g.w+iw)
							}
							idx++
						}
					}
				}
				row++
			}
		}
	}
}

// Conv2d applies a 2-D convolution (cross-correlation, as in PyTorch) to x of
// shape (N, C, H, W) with weight of shape (O, C, KH, KW) and an optional bias
// of shape (O). The result has shape (N, O, OH, OW).
func Conv2d(x, weight, bias *Tensor, stride, padding int) *Tensor {
	if len(weight.shape) != 4 || len(x.shape) != 4 || weight.shape[1] != x.shape[1] {
		panic(fmt.Sprintf("tensor: Conv2d weight %v does not match input %v", weight.shape, x.shape))
	}
	o := weight.shape[0]
	if bias != nil && (len(bias.shape) != 1 || bias.shape[0] != o) {
		panic(fmt.Sprintf("tensor: Conv2d bias shape %v, want [%d]", bias.shape, o))
	}
	g := newConvGeom("Conv2d", x.shape, weight.shape[2], weight.shape[3], stride, padding)
	l, k := g.patches()
	rows := g.n * l

	// (N*L, K) @ (K, O) -> (N*L, O), then reorder to (N, O, L).
	cols := g.im2col(x.data)
	y := matmul(cols, transpose2d(weight.data, o, k), rows, k, o)
	out := make([]float64, len(y))
	for n := 0; n < g.n; n++ {
		for p := 0; p < l; p++ {
			src := y[(n*l+p)*o : (n*l+p+1)*o]
			for j, v := range src {
				if bias != nil {
					v += bias.data[j]
				}
				out[(n*o+j)*l+p] = v
			}
		}
	}

	inputs := []*Tensor{x, weight}
	if bias != nil {
		inputs = append(inputs, bias)
	}
	return newResult(out, []int{g.n, o, g.oh, g.ow}, "Conv2d", inputs, func(grad *Tensor) []*Tensor {
		// Reorder the gradient from (N, O, L) to (N*L, O) to match y.
		gy := make([]float64, len(grad.data))
		for n := 0; n < g.n; n++ {
			for j := 0; j < o; j++ {
				for p := 0; p < l; p++ {
					gy[(n*l+p)*o+j] = grad.data[(n*o+j)*l+p]
				}
			}
		}
		grads := make([]*Tensor, len(inputs))
		if x.requiresGrad {
			grads[0] = fromData(g.col2im(matmul(gy, weight.data, rows, o, k)), x.shape)
		}
		if weight.requiresGrad {
			grads[1] = fromData(matmul(transpose2d(gy, rows, o), cols, o, rows, k), weight.shape)
		}
		if bias != nil && bias.requiresGrad {
			gb := make([]float64, o)
			for r := 0; r < rows; r++ {
				for j, v := range gy[r*o : (r+1)*o] {
					gb[j] += v
				}
			}
			grads[2] = fromData(gb, bias.shape)
		}
		return grads
	})
}

// MaxPool2d takes the maximum over kernel×kernel windows of x, which has shape
// (N, C, H, W), moving the window by stride.
func MaxPool2d(x *Tensor, kernel, stride int) *Tensor {
	g := newConvGeom("MaxPool2d", x.shape, kernel, kernel, stride, 0)
	planes := g.n * g.c
	out := make([]float64, planes*g.oh*g.ow)
	argmax := make([]int, len(out))
	i := 0
	for pl := 0; pl < planes; pl++ {
		base := pl * g.h * g.w
		for oh := 0; oh < g.oh; oh++ {
			for ow := 0; ow < g.ow; ow++ {
				best, bestIdx := math.Inf(-1), -1
				for kh := 0; kh < kernel; kh++ {
					for kw := 0; kw < kernel; kw++ {
						idx := base + (oh*stride+kh)*g.w + ow*stride + kw
						if v := x.data[idx]; v > best || bestIdx < 0 {
							best, bestIdx = v, idx
						}
					}
				}
				out[i], argmax[i] = best, bestIdx
				i++
			}
		}
	}
	return newResult(out, []int{g.n, g.c, g.oh, g.ow}, "MaxPool2d", []*Tensor{x}, func(grad *Tensor) []*Tensor {
		d := make([]float64, len(x.data))
		for i, idx := range argmax {
			d[idx] += grad.data[i]
		}
		return []*Tensor{fromData(d, x.shape)}
	})
}
