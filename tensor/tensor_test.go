package tensor

import (
	"math"
	"testing"
)

type gradCase struct {
	name   string
	f      func(xs ...*Tensor) *Tensor
	inputs [][]int // input shapes
	pos    bool    // inputs must be positive (Log, Sqrt, Div)
}

// gradCases lists every differentiable operation with sample input shapes.
// The device tests reuse it to compare GPU gradients with CPU gradients.
func gradCases() []gradCase {
	return []gradCase{
		{"Add/broadcast", func(x ...*Tensor) *Tensor { return x[0].Add(x[1]) }, [][]int{{3, 4}, {4}}, false},
		{"Sub/broadcast", func(x ...*Tensor) *Tensor { return x[0].Sub(x[1]) }, [][]int{{3, 1}, {1, 4}}, false},
		{"Mul/broadcast", func(x ...*Tensor) *Tensor { return x[0].Mul(x[1]) }, [][]int{{2, 3, 4}, {3, 1}}, false},
		{"Div", func(x ...*Tensor) *Tensor { return x[0].Div(x[1]) }, [][]int{{3, 4}, {3, 4}}, true},
		{"Pow", func(x ...*Tensor) *Tensor { return x[0].Pow(3) }, [][]int{{5}}, false},
		{"Exp", func(x ...*Tensor) *Tensor { return x[0].Exp() }, [][]int{{5}}, false},
		{"Log", func(x ...*Tensor) *Tensor { return x[0].Log() }, [][]int{{5}}, true},
		{"Sqrt", func(x ...*Tensor) *Tensor { return x[0].Sqrt() }, [][]int{{5}}, true},
		{"Abs", func(x ...*Tensor) *Tensor { return x[0].Abs() }, [][]int{{6}}, false},
		{"Sigmoid", func(x ...*Tensor) *Tensor { return x[0].Sigmoid() }, [][]int{{5}}, false},
		{"Tanh", func(x ...*Tensor) *Tensor { return x[0].Tanh() }, [][]int{{5}}, false},
		{"ReLU", func(x ...*Tensor) *Tensor { return x[0].ReLU() }, [][]int{{6}}, false},
		{"MulScalar/AddScalar", func(x ...*Tensor) *Tensor { return x[0].MulScalar(3).AddScalar(1) }, [][]int{{4}}, false},
		{"MatMul", func(x ...*Tensor) *Tensor { return x[0].MatMul(x[1]) }, [][]int{{3, 4}, {4, 2}}, false},
		{"T", func(x ...*Tensor) *Tensor { return x[0].T().Mul(x[1]) }, [][]int{{2, 3, 4}, {4, 3}}, false},
		{"Permute", func(x ...*Tensor) *Tensor { return x[0].Permute(2, 0, 1).Mul(x[1]) }, [][]int{{2, 3, 4}, {4, 2, 3}}, false},
		{"Reshape", func(x ...*Tensor) *Tensor { return x[0].Reshape(-1, 2).Mul(x[1]) }, [][]int{{2, 3}, {2}}, false},
		{"Sum", func(x ...*Tensor) *Tensor { return x[0].Sum().Mul(x[0]) }, [][]int{{2, 3}}, false},
		{"SumDim", func(x ...*Tensor) *Tensor { return x[0].SumDim(1, false).Pow(2) }, [][]int{{2, 3, 4}}, false},
		{"MeanDim/keep", func(x ...*Tensor) *Tensor { return x[0].MeanDim(-1, true).Mul(x[0]) }, [][]int{{3, 4}}, false},
		{"Mean", func(x ...*Tensor) *Tensor { return x[0].Mul(x[0]).Mean() }, [][]int{{3, 4}}, false},
		{"LogSoftmax", func(x ...*Tensor) *Tensor { return x[0].LogSoftmax().Mul(x[1]) }, [][]int{{3, 4}, {3, 4}}, false},
		{"Pick", func(x ...*Tensor) *Tensor { return x[0].Pick([]int{2, 0, 1}).Pow(2) }, [][]int{{3, 4}}, false},
		{"Index", func(x ...*Tensor) *Tensor { return x[0].Index(1).Pow(2) }, [][]int{{3, 2, 2}}, false},
		{"Stack", func(x ...*Tensor) *Tensor { return Stack([]*Tensor{x[0], x[1].Exp()}).Mul(x[0]) }, [][]int{{3}, {3}}, false},
		{"Conv2d", func(x ...*Tensor) *Tensor { return Conv2d(x[0], x[1], x[2], 1, 0).Pow(2) }, [][]int{{2, 3, 5, 5}, {4, 3, 3, 3}, {4}}, false},
		{"Conv2d/stride+pad", func(x ...*Tensor) *Tensor { return Conv2d(x[0], x[1], nil, 2, 1).Pow(2) }, [][]int{{2, 2, 5, 4}, {3, 2, 3, 2}}, false},
		{"MaxPool2d", func(x ...*Tensor) *Tensor { return MaxPool2d(x[0], 2, 2).Pow(2) }, [][]int{{2, 3, 4, 6}}, false},
		{"MaxPool2d/overlap", func(x ...*Tensor) *Tensor { return MaxPool2d(x[0], 3, 1).Pow(2) }, [][]int{{1, 2, 5, 5}}, false},
		{"reuse", func(x ...*Tensor) *Tensor { return x[0].Mul(x[0]).Add(x[0].Exp()) }, [][]int{{4}}, false},
	}
}

// sampleInputs returns host data for a case's inputs. Values stay away from
// 0 so ReLU, Abs and MaxPool are differentiable at every sample.
func sampleInputs(c gradCase) [][]float64 {
	vals := make([][]float64, len(c.inputs))
	for i, shape := range c.inputs {
		v := make([]float64, numel(shape))
		rngMu.Lock()
		for j := range v {
			if c.pos {
				v[j] = 0.5 + 1.5*rng.Float64()
			} else {
				v[j] = rng.NormFloat64()
				if math.Abs(v[j]) < 0.05 {
					v[j] += 0.1
				}
			}
		}
		rngMu.Unlock()
		vals[i] = v
	}
	return vals
}

// gradCheck compares the autograd gradient of sum(f(inputs...)) against a
// central finite-difference estimate, in float64 on the CPU.
func gradCheck(t *testing.T, c gradCase) {
	t.Helper()
	vals := sampleInputs(c)
	build := func() []*Tensor {
		xs := make([]*Tensor, len(vals))
		for i, v := range vals {
			xs[i] = FromFloat64(v, c.inputs[i]...)
		}
		return xs
	}
	xs := build()
	for _, x := range xs {
		x.SetRequiresGrad(true)
	}
	c.f(xs...).Sum().Backward()

	const eps = 1e-6
	eval := func() (v float64) {
		NoGrad(func() { v = c.f(build()...).Sum().Item() })
		return v
	}
	for n, x := range xs {
		var grad []float64
		if x.Grad() != nil {
			grad = x.Grad().Float64s()
		}
		for i := range vals[n] {
			orig := vals[n][i]
			vals[n][i] = orig + eps
			fp := eval()
			vals[n][i] = orig - eps
			fm := eval()
			vals[n][i] = orig

			want := (fp - fm) / (2 * eps)
			got := 0.0
			if grad != nil {
				got = grad[i]
			}
			if math.Abs(want-got) > 1e-5*math.Max(1, math.Abs(want)) {
				t.Errorf("%s: input %d element %d: autograd %.8f, numeric %.8f", c.name, n, i, got, want)
			}
		}
	}
}

func TestGradients(t *testing.T) {
	ManualSeed(1)
	for _, c := range gradCases() {
		gradCheck(t, c)
	}
}

func TestDefaultsAndDTypes(t *testing.T) {
	x := Randn(2, 3)
	if x.DType() != Float32 || x.Device() != CPU {
		t.Fatalf("default tensor is %v on %v, want float32 on cpu", x.DType(), x.Device())
	}
	d := x.ToDType(Float64)
	if d.DType() != Float64 || d.Float64s()[0] != float64(x.Float32s()[0]) {
		t.Fatal("ToDType(Float64) did not preserve values")
	}

	SetDefaultDType(Float64)
	defer SetDefaultDType(Float32)
	if y := New([]float64{0.1}); y.DType() != Float64 || y.Item() != 0.1 {
		t.Fatalf("New under default float64: %v %v", y.DType(), y.Item())
	}

	defer func() {
		if recover() == nil {
			t.Fatal("mixing float32 and float64 did not panic")
		}
	}()
	x.Add(d)
}

func TestToGradientFlowsBack(t *testing.T) {
	x := FromFloat64([]float64{1, 2}).SetRequiresGrad(true)
	y := x.ToDType(Float32).MulScalar(3).Sum()
	y.Backward()
	if g := x.Grad(); g.DType() != Float64 || g.Float64s()[1] != 3 {
		t.Fatalf("grad through ToDType = %v", g)
	}
}

func TestCopyFromAndMoveTo(t *testing.T) {
	p := Zeros(2, 2).SetRequiresGrad(true)
	p.CopyFrom(FromFloat64([]float64{1, 2, 3, 4}, 4))
	if got := p.Float32s(); got[3] != 4 {
		t.Fatalf("CopyFrom = %v", got)
	}
	p.MoveTo(CPU)
	p.CastTo(Float64)
	if p.DType() != Float64 || !p.RequiresGrad() || p.At(1, 0) != 3 {
		t.Fatalf("CastTo: %v", p)
	}
}

func TestUnavailableDevice(t *testing.T) {
	if IsAvailable(CUDA(0)) {
		t.Skip("CUDA is compiled in")
	}
	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("To(CUDA) did not panic")
		}
		t.Log(r)
	}()
	Ones(2).To(CUDA(0))
}

func TestParseDevice(t *testing.T) {
	for s, want := range map[string]Device{"cpu": CPU, "MPS": MPS, "cuda": CUDA(0), "cuda:2": CUDA(2)} {
		if got, err := ParseDevice(s); err != nil || got != want {
			t.Errorf("ParseDevice(%q) = %v, %v", s, got, err)
		}
	}
	if _, err := ParseDevice("tpu"); err == nil {
		t.Error("ParseDevice accepted tpu")
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
	got := a.MatMul(b).Float64s()
	want := []float64{58, 64, 139, 154}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("MatMul = %v, want %v", got, want)
		}
	}
}

func TestLargeMatMul(t *testing.T) {
	// Large enough to take the parallel path; checked against a naive loop.
	a, b := FromFloat64(sampleInputs(gradCase{inputs: [][]int{{70, 50}}})[0], 70, 50),
		FromFloat64(sampleInputs(gradCase{inputs: [][]int{{50, 40}}})[0], 50, 40)
	got := a.MatMul(b).Float64s()
	av, bv := a.Float64s(), b.Float64s()
	for i := 0; i < 70; i++ {
		for j := 0; j < 40; j++ {
			s := 0.0
			for p := 0; p < 50; p++ {
				s += av[i*50+p] * bv[p*40+j]
			}
			if math.Abs(s-got[i*40+j]) > 1e-9 {
				t.Fatalf("element (%d,%d): got %v, want %v", i, j, got[i*40+j], s)
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
	if got := FromFloat64([]float64{1}).String(); got != "tensor([1], dtype=float64)" {
		t.Fatalf("String() = %q", got)
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
	for i, v := range got.Float64s() {
		if v != want[i] {
			t.Fatalf("Conv2d = %v, want %v", got, want)
		}
	}

	pooled := MaxPool2d(x, 2, 1).Float64s()
	for i, v := range []float64{5, 6, 8, 9} {
		if pooled[i] != v {
			t.Fatalf("MaxPool2d = %v", pooled)
		}
	}
}

// BenchmarkConv2d measures forward+backward of the second conv layer of the
// MNIST example (batch 64, 16->32 channels, 14x14, 3x3 kernel).
func BenchmarkConv2d(b *testing.B) {
	benchConv(b, CPU)
}

func benchConv(b *testing.B, dev Device) {
	x := Randn(64, 16, 14, 14).To(dev).SetRequiresGrad(true)
	w := Randn(32, 16, 3, 3).To(dev).SetRequiresGrad(true)
	bias := Randn(32).To(dev).SetRequiresGrad(true)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Conv2d(x, w, bias, 1, 1).Sum().Backward()
	}
	Synchronize(dev)
}
