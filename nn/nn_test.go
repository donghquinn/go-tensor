package nn_test

import (
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
