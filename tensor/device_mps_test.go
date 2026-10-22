//go:build darwin && mps

package tensor

import (
	"math"
	"strings"
	"testing"
)

func requireMPS(t testing.TB) {
	t.Helper()
	if !IsAvailable(MPS) {
		_, err := backendFor(MPS)
		t.Fatalf("MPS unavailable: %v", err)
	}
}

func closeEnough(a, b float64) bool {
	return math.Abs(a-b) <= 1e-4*math.Max(1, math.Abs(b)) || (math.IsNaN(a) && math.IsNaN(b))
}

// TestMPSMatchesCPU runs the forward and backward pass of every
// differentiable operation on the CPU and on the GPU, in float32, and
// requires the outputs and every input gradient to agree.
func TestMPSMatchesCPU(t *testing.T) {
	requireMPS(t)
	ManualSeed(2)
	for _, c := range gradCases() {
		vals := sampleInputs(c)
		run := func(dev Device) (out []float64, grads [][]float64) {
			xs := make([]*Tensor, len(vals))
			for i, v := range vals {
				xs[i] = FromFloat64(v, c.inputs[i]...).ToDType(Float32).To(dev).SetRequiresGrad(true)
			}
			y := c.f(xs...)
			if y.Device() != dev {
				t.Fatalf("%s: output on %v, want %v", c.name, y.Device(), dev)
			}
			y.Sum().Backward()
			for _, x := range xs {
				if x.Grad().Device() != dev {
					t.Fatalf("%s: gradient on %v, want %v", c.name, x.Grad().Device(), dev)
				}
				grads = append(grads, x.Grad().Float64s())
			}
			return y.Float64s(), grads
		}
		cpuOut, cpuGrads := run(CPU)
		gpuOut, gpuGrads := run(MPS)
		for i := range cpuOut {
			if !closeEnough(gpuOut[i], cpuOut[i]) {
				t.Errorf("%s: output[%d] mps=%v cpu=%v", c.name, i, gpuOut[i], cpuOut[i])
				break
			}
		}
		for n := range cpuGrads {
			for i := range cpuGrads[n] {
				if !closeEnough(gpuGrads[n][i], cpuGrads[n][i]) {
					t.Errorf("%s: grad of input %d [%d] mps=%v cpu=%v", c.name, n, i, gpuGrads[n][i], cpuGrads[n][i])
					break
				}
			}
		}
	}
}

func TestMPSLargeOps(t *testing.T) {
	requireMPS(t)
	ManualSeed(3)
	// Sizes that exercise multi-threadgroup dispatch, the reduction tree and
	// MPS matrix multiplication with transposes.
	a, b := Randn(257, 129), Randn(129, 65)
	cases := map[string]func(a, b *Tensor) *Tensor{
		"matmul":    func(a, b *Tensor) *Tensor { return a.MatMul(b) },
		"matmul^T":  func(a, b *Tensor) *Tensor { return b.T().MatMul(a.T()) },
		"sum":       func(a, _ *Tensor) *Tensor { return a.Reshape(-1).Sum() },
		"sumdim0":   func(a, _ *Tensor) *Tensor { return a.SumDim(0, false) },
		"maxdim":    func(a, _ *Tensor) *Tensor { return a.MaxDim(1, false) },
		"softmax":   func(a, _ *Tensor) *Tensor { return a.Softmax() },
		"broadcast": func(a, b *Tensor) *Tensor { return a.Add(b.SumDim(1, true).T()) }, // (257,129) + (1,129)
	}
	for name, f := range cases {
		cpu := f(a, b).Float64s()
		gpu := f(a.To(MPS), b.To(MPS)).Float64s()
		for i := range cpu {
			if math.Abs(gpu[i]-cpu[i]) > 1e-3*math.Max(1, math.Abs(cpu[i])) {
				t.Errorf("%s[%d]: mps=%v cpu=%v", name, i, gpu[i], cpu[i])
				break
			}
		}
	}
}

func TestMPSDeviceSemantics(t *testing.T) {
	requireMPS(t)
	x := New([]float64{1, 2, 3}).To(MPS)
	if x.Device() != MPS || x.DType() != Float32 {
		t.Fatalf("To(MPS) gave %v %v", x.Device(), x.DType())
	}
	if s := x.String(); !strings.Contains(s, "device=mps") {
		t.Fatalf("String() = %q", s)
	}
	if back := x.MulScalar(2).To(CPU); back.Device() != CPU || back.Float64s()[2] != 6 {
		t.Fatalf("round trip = %v", back)
	}

	// In-place updates, as used by optimizers and LoadStateDict.
	p := Zeros(3).To(MPS)
	p.CopyFrom(FromFloat64([]float64{4, 5, 6}))
	q := Zeros(2, 2)
	q.MoveTo(MPS)
	if p.Float64s()[1] != 5 || q.Device() != MPS {
		t.Fatalf("CopyFrom/MoveTo: %v %v", p, q)
	}

	mustPanic(t, "float64 on MPS", func() { FromFloat64([]float64{1}).To(MPS) })
	mustPanic(t, "mixed devices", func() { x.Add(New([]float64{1, 2, 3})) })
}

func mustPanic(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("%s: did not panic", name)
		}
	}()
	f()
}

func BenchmarkConv2dMPS(b *testing.B) {
	requireMPS(b)
	benchConv(b, MPS)
}
