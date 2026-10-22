package nn_test

import (
	"bytes"
	"testing"

	"github.com/donghquinn/go-tensor/nn"
	"github.com/donghquinn/go-tensor/optim"
	"github.com/donghquinn/go-tensor/tensor"
)

func TestTrainXOR(t *testing.T) {
	tensor.ManualSeed(0)
	x := tensor.New([]float64{0, 0, 0, 1, 1, 0, 1, 1}, 4, 2)
	y := []int{0, 1, 1, 0}

	model := nn.NewSequential(nn.NewLinear(2, 8), nn.NewTanh(), nn.NewLinear(8, 2))
	opt := optim.NewAdam(model.Parameters(), optim.AdamConfig{LR: 0.05})

	var loss *tensor.Tensor
	for i := 0; i < 500; i++ {
		loss = nn.CrossEntropyLoss(model.Forward(x), y)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
	}
	if loss.Item() > 0.01 {
		t.Fatalf("final loss %v, want < 0.01", loss.Item())
	}
	for i, p := range model.Forward(x).Argmax() {
		if p != y[i] {
			t.Fatalf("prediction %d = %d, want %d", i, p, y[i])
		}
	}
}

func TestLinearRegressionSGD(t *testing.T) {
	tensor.ManualSeed(0)
	// y = 3x - 2
	x := tensor.Randn(64, 1)
	y := x.MulScalar(3).AddScalar(-2)

	lin := nn.NewLinear(1, 1)
	opt := optim.NewSGD(lin.Parameters(), optim.SGDConfig{LR: 0.1, Momentum: 0.9})
	for i := 0; i < 200; i++ {
		loss := nn.MSELoss(lin.Forward(x), y)
		opt.ZeroGrad()
		loss.Backward()
		opt.Step()
	}
	w, b := lin.Weight.Item(), lin.Bias.Item()
	if abs(w-3) > 1e-3 || abs(b+2) > 1e-3 {
		t.Fatalf("learned w=%v b=%v, want w=3 b=-2", w, b)
	}
}

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}

func TestSaveLoad(t *testing.T) {
	tensor.ManualSeed(0)
	src := nn.NewSequential(nn.NewLinear(3, 4), nn.NewReLU(), nn.NewLinear(4, 2))
	var buf bytes.Buffer
	if err := nn.Save(&buf, src); err != nil {
		t.Fatal(err)
	}

	dst := nn.NewSequential(nn.NewLinear(3, 4), nn.NewReLU(), nn.NewLinear(4, 2))
	if err := nn.Load(bytes.NewReader(buf.Bytes()), dst); err != nil {
		t.Fatal(err)
	}
	x := tensor.Randn(5, 3)
	a, b := src.Forward(x).Float64s(), dst.Forward(x).Float64s()
	for i := range a {
		if a[i] != b[i] {
			t.Fatalf("outputs differ after load: %v vs %v", a, b)
		}
	}

	wrong := nn.NewSequential(nn.NewLinear(3, 5), nn.NewReLU(), nn.NewLinear(5, 2))
	if err := nn.Load(bytes.NewReader(buf.Bytes()), wrong); err == nil {
		t.Fatal("expected shape mismatch error")
	}
}

func TestDropoutModes(t *testing.T) {
	tensor.ManualSeed(0)
	d := nn.NewDropout(0.5)
	model := nn.NewSequential(d)
	x := tensor.Ones(1000)

	out := model.Forward(x).Float64s()
	zeros := 0
	for _, v := range out {
		if v == 0 {
			zeros++
		} else if v != 2 {
			t.Fatalf("kept value %v, want 2 (scaled by 1/(1-p))", v)
		}
	}
	if zeros < 400 || zeros > 600 {
		t.Fatalf("dropped %d of 1000, want about 500", zeros)
	}

	nn.Eval(model)
	if got := model.Forward(x); got != x {
		t.Fatal("dropout in eval mode should return its input unchanged")
	}
	nn.Train(model)
	if got := model.Forward(x); got == x {
		t.Fatal("dropout did not return to training mode")
	}
}

func TestConvNetForwardShape(t *testing.T) {
	model := nn.NewSequential(
		nn.NewConv2d(1, 4, 3, 1, 1),
		nn.NewReLU(),
		nn.NewMaxPool2d(2, 0),
		nn.NewFlatten(),
		nn.NewLinear(4*4*4, 3),
	)
	out := model.Forward(tensor.Randn(2, 1, 8, 8))
	if s := out.Shape(); len(s) != 2 || s[0] != 2 || s[1] != 3 {
		t.Fatalf("output shape %v, want [2 3]", s)
	}
	if n := len(model.Parameters()); n != 4 {
		t.Fatalf("%d parameters, want 4", n)
	}
}
