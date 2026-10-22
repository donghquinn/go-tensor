// Command xor trains a small MLP to learn the XOR function.
package main

import (
	"fmt"

	"github.com/donghquinn/go-tensor/nn"
	"github.com/donghquinn/go-tensor/optim"
	"github.com/donghquinn/go-tensor/tensor"
)

func main() {
	tensor.ManualSeed(42)

	x := tensor.New([]float64{
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	}, 4, 2)
	y := []int{0, 1, 1, 0}

	model := nn.NewSequential(
		nn.NewLinear(2, 8),
		nn.NewTanh(),
		nn.NewLinear(8, 2),
	)
	opt := optim.NewAdam(model.Parameters(), optim.AdamConfig{LR: 0.05})

	for epoch := 0; epoch <= 500; epoch++ {
		loss := nn.CrossEntropyLoss(model.Forward(x), y)

		opt.ZeroGrad()
		loss.Backward()
		opt.Step()

		if epoch%100 == 0 {
			fmt.Printf("epoch %3d  loss %.6f\n", epoch, loss.Item())
		}
	}

	tensor.NoGrad(func() {
		probs := model.Forward(x).Softmax()
		fmt.Println("probabilities:", probs)
		fmt.Println("predictions:  ", probs.Argmax())
	})
}
