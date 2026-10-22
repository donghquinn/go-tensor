package tensor

import (
	"math"
	"testing"
)

// gradCheck compares the autograd gradient of sum(f(inputs...)) against a
// central finite-difference estimate for every input element.
func gradCheck(t *testing.T, name string, f func(xs ...*Tensor) *Tensor, inputs ...*Tensor) {
	t.Helper()
	for _, x := range inputs {
		x.SetRequiresGrad(true).ZeroGrad()
	}
	f(inputs...).Sum().Backward()

	const eps = 1e-6
	eval := func() (v float64) {
		NoGrad(func() { v = f(inputs...).Sum().Item() })
		return v
	}
	for n, x := range inputs {
		for i := range x.data {
			orig := x.data[i]
			x.data[i] = orig + eps
			fp := eval()
			x.data[i] = orig - eps
			fm := eval()
			x.data[i] = orig

			want := (fp - fm) / (2 * eps)
			got := 0.0
			if x.grad != nil {
				got = x.grad.data[i]
			}
			if math.Abs(want-got) > 1e-5*math.Max(1, math.Abs(want)) {
				t.Errorf("%s: input %d element %d: autograd %.8f, numeric %.8f", name, n, i, got, want)
			}
		}
	}
}

func TestGradients(t *testing.T) {
	ManualSeed(1)
	pos := func(shape ...int) *Tensor { return Uniform(0.5, 2, shape...) }

	cases := []struct {
		name   string
		f      func(xs ...*Tensor) *Tensor
		inputs []*Tensor
	}{
		{"Add/broadcast", func(x ...*Tensor) *Tensor { return x[0].Add(x[1]) }, []*Tensor{Randn(3, 4), Randn(4)}},
		{"Sub/broadcast", func(x ...*Tensor) *Tensor { return x[0].Sub(x[1]) }, []*Tensor{Randn(3, 1), Randn(1, 4)}},
		{"Mul/broadcast", func(x ...*Tensor) *Tensor { return x[0].Mul(x[1]) }, []*Tensor{Randn(2, 3, 4), Randn(3, 1)}},
		{"Div", func(x ...*Tensor) *Tensor { return x[0].Div(x[1]) }, []*Tensor{Randn(3, 4), pos(3, 4)}},
		{"Pow", func(x ...*Tensor) *Tensor { return x[0].Pow(3) }, []*Tensor{Randn(5)}},
		{"Exp", func(x ...*Tensor) *Tensor { return x[0].Exp() }, []*Tensor{Randn(5)}},
		{"Log", func(x ...*Tensor) *Tensor { return x[0].Log() }, []*Tensor{pos(5)}},
		{"Sqrt", func(x ...*Tensor) *Tensor { return x[0].Sqrt() }, []*Tensor{pos(5)}},
		{"Sigmoid", func(x ...*Tensor) *Tensor { return x[0].Sigmoid() }, []*Tensor{Randn(5)}},
		{"Tanh", func(x ...*Tensor) *Tensor { return x[0].Tanh() }, []*Tensor{Randn(5)}},
		{"ReLU", func(x ...*Tensor) *Tensor { return x[0].ReLU() }, []*Tensor{New([]float64{-2, -0.5, 0.3, 1.5})}},
		{"MulScalar/AddScalar", func(x ...*Tensor) *Tensor { return x[0].MulScalar(3).AddScalar(1) }, []*Tensor{Randn(4)}},
		{"MatMul", func(x ...*Tensor) *Tensor { return x[0].MatMul(x[1]) }, []*Tensor{Randn(3, 4), Randn(4, 2)}},
		{"T", func(x ...*Tensor) *Tensor { return x[0].T().Mul(x[1]) }, []*Tensor{Randn(2, 3, 4), Randn(4, 3)}},
		{"Reshape", func(x ...*Tensor) *Tensor { return x[0].Reshape(-1, 2).Mul(x[1]) }, []*Tensor{Randn(2, 3), Randn(2)}},
		{"SumDim", func(x ...*Tensor) *Tensor { return x[0].SumDim(1, false).Pow(2) }, []*Tensor{Randn(2, 3, 4)}},
		{"MeanDim/keep", func(x ...*Tensor) *Tensor { return x[0].MeanDim(-1, true).Mul(x[0]) }, []*Tensor{Randn(3, 4)}},
		{"Mean", func(x ...*Tensor) *Tensor { return x[0].Mul(x[0]).Mean() }, []*Tensor{Randn(3, 4)}},
		{"LogSoftmax", func(x ...*Tensor) *Tensor { return x[0].LogSoftmax().Mul(x[1]) }, []*Tensor{Randn(3, 4), Randn(3, 4)}},
		{"Pick", func(x ...*Tensor) *Tensor { return x[0].Pick([]int{2, 0, 1}).Pow(2) }, []*Tensor{Randn(3, 4)}},
		{"Index", func(x ...*Tensor) *Tensor { return x[0].Index(1).Pow(2) }, []*Tensor{Randn(3, 2, 2)}},
		{"Stack", func(x ...*Tensor) *Tensor { return Stack([]*Tensor{x[0], x[1].Exp()}).Mul(x[0]) }, []*Tensor{Randn(3), Randn(3)}},
		{"Conv2d", func(x ...*Tensor) *Tensor { return Conv2d(x[0], x[1], x[2], 1, 0).Pow(2) }, []*Tensor{Randn(2, 3, 5, 5), Randn(4, 3, 3, 3), Randn(4)}},
		{"Conv2d/stride+pad", func(x ...*Tensor) *Tensor { return Conv2d(x[0], x[1], nil, 2, 1).Pow(2) }, []*Tensor{Randn(2, 2, 5, 4), Randn(3, 2, 3, 2)}},
		{"MaxPool2d", func(x ...*Tensor) *Tensor { return MaxPool2d(x[0], 2, 2).Pow(2) }, []*Tensor{Randn(2, 3, 4, 6)}},
		{"MaxPool2d/overlap", func(x ...*Tensor) *Tensor { return MaxPool2d(x[0], 3, 1).Pow(2) }, []*Tensor{Randn(1, 2, 5, 5)}},
		{"reuse", func(x ...*Tensor) *Tensor { return x[0].Mul(x[0]).Add(x[0].Exp()) }, []*Tensor{Randn(4)}},
	}
	for _, c := range cases {
		gradCheck(t, c.name, c.f, c.inputs...)
	}
}

func TestBroadcastShape(t *testing.T) {
	out := Ones(3, 1).Add(Ones(1, 4))
	if !equalShape(out.Shape(), []int{3, 4}) {
		t.Fatalf("shape = %v, want [3 4]", out.Shape())
	}
}

func TestMatMulValues(t *testing.T) {
	a := New([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := New([]float64{7, 8, 9, 10, 11, 12}, 3, 2)
	got := a.MatMul(b).Data()
	want := []float64{58, 64, 139, 154}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("MatMul = %v, want %v", got, want)
		}
	}
}

func TestMatMulParallelMatchesSerial(t *testing.T) {
	a, b := Randn(70, 50), Randn(50, 40)
	par := matmul(a.data, b.data, 70, 50, 40)
	for i := 0; i < 70; i++ {
		for j := 0; j < 40; j++ {
			s := 0.0
			for p := 0; p < 50; p++ {
				s += a.data[i*50+p] * b.data[p*40+j]
			}
			if math.Abs(s-par[i*40+j]) > 1e-9 {
				t.Fatalf("element (%d,%d): got %v, want %v", i, j, par[i*40+j], s)
			}
		}
	}
}

func TestGradAccumulates(t *testing.T) {
	x := New([]float64{2}).SetRequiresGrad(true)
	x.Mul(x).Sum().Backward()
	x.Mul(x).Sum().Backward()
	if g := x.Grad().Item(); g != 8 {
		t.Fatalf("accumulated grad = %v, want 8", g)
	}
}

func TestNoGrad(t *testing.T) {
	x := Randn(3).SetRequiresGrad(true)
	var y *Tensor
	NoGrad(func() { y = x.Exp() })
	if y.RequiresGrad() {
		t.Fatal("result of op inside NoGrad requires grad")
	}
	if !IsGradEnabled() {
		t.Fatal("grad not re-enabled after NoGrad")
	}
}

func TestString(t *testing.T) {
	got := New([]float64{1, 2, 3, 4}, 2, 2).String()
	want := "tensor([[1, 2],\n        [3, 4]])"
	if got != want {
		t.Fatalf("String() = %q, want %q", got, want)
	}
}

func TestConv2dValues(t *testing.T) {
	// 1x1x3x3 input, 1x1x2x2 kernel of ones, padding 0: each output is a 2x2 window sum.
	x := New([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, 1, 1, 3, 3)
	w := Ones(1, 1, 2, 2)
	got := Conv2d(x, w, New([]float64{10}), 1, 0)
	want := []float64{22, 26, 34, 38}
	if !equalShape(got.Shape(), []int{1, 1, 2, 2}) {
		t.Fatalf("shape = %v, want [1 1 2 2]", got.Shape())
	}
	for i, v := range want {
		if got.data[i] != v {
			t.Fatalf("Conv2d = %v, want %v", got.data, want)
		}
	}

	pooled := MaxPool2d(x, 2, 1).Data()
	for i, v := range []float64{5, 6, 8, 9} {
		if pooled[i] != v {
			t.Fatalf("MaxPool2d = %v", pooled)
		}
	}
}

// BenchmarkConv2d measures forward+backward of the second conv layer of the
// MNIST example (batch 64, 16->32 channels, 14x14, 3x3 kernel).
func BenchmarkConv2d(b *testing.B) {
	x := Randn(64, 16, 14, 14).SetRequiresGrad(true)
	w := Randn(32, 16, 3, 3).SetRequiresGrad(true)
	bias := Randn(32).SetRequiresGrad(true)
	for i := 0; i < b.N; i++ {
		Conv2d(x, w, bias, 1, 1).Sum().Backward()
	}
}
