//go:build darwin && mps

package nn_test

import (
	"math"
	"testing"

	"github.com/donghquinn/go-tensor/nn"
	"github.com/donghquinn/go-tensor/optim"
	"github.com/donghquinn/go-tensor/tensor"
)

// TestTrainOnMPSMatchesCPU trains the same CNN for a few steps on each device
// and requires the loss curves to agree.
func TestTrainOnMPSMatchesCPU(t *testing.T) {
	losses := func(dev tensor.Device) []float64 {
		tensor.ManualSeed(0)
		model := nn.NewSequential(
			nn.NewConv2d(1, 4, 3, 1, 1), nn.NewReLU(), nn.NewMaxPool2d(2, 0),
			nn.NewFlatten(), nn.NewLinear(4*4*4, 3),
		)
		nn.To(model, dev)
		opt := optim.NewAdam(model.Parameters(), optim.AdamConfig{LR: 0.01})
		x := tensor.Randn(16, 1, 8, 8).To(dev)
		y := make([]int, 16)
		for i := range y {
			y[i] = i % 3
		}
		var out []float64
		for step := 0; step < 20; step++ {
			loss := nn.CrossEntropyLoss(model.Forward(x), y)
			opt.ZeroGrad()
			loss.Backward()
			opt.Step()
			out = append(out, loss.Item())
		}
		for _, p := range model.Parameters() {
			if p.Device() != dev {
				t.Fatalf("parameter moved to %v during training", p.Device())
			}
		}
		return out
	}
	cpu, gpu := losses(tensor.CPU), losses(tensor.MPS)
	for i := range cpu {
		if math.Abs(cpu[i]-gpu[i]) > 1e-3 {
			t.Fatalf("step %d: loss cpu=%v mps=%v", i, cpu[i], gpu[i])
		}
	}
	if gpu[len(gpu)-1] > gpu[0]/2 {
		t.Fatalf("loss did not decrease on MPS: %v", gpu)
	}
}
