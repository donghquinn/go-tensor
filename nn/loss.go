package nn

import "github.com/donghquinn/go-tensor/tensor"

// MSELoss returns the mean squared error between pred and target.
func MSELoss(pred, target *tensor.Tensor) *tensor.Tensor {
	return pred.Sub(target).Pow(2).Mean()
}

// CrossEntropyLoss returns the mean cross-entropy between raw logits of shape
// (batch, classes) and integer class targets, like torch.nn.CrossEntropyLoss.
func CrossEntropyLoss(logits *tensor.Tensor, targets []int) *tensor.Tensor {
	return logits.LogSoftmax().Pick(targets).Mean().Neg()
}

// BCELoss returns the binary cross-entropy between probabilities pred in (0, 1)
// and targets in [0, 1].
func BCELoss(pred, target *tensor.Tensor) *tensor.Tensor {
	pos := target.Mul(pred.Log())
	neg := target.Neg().AddScalar(1).Mul(pred.Neg().AddScalar(1).Log())
	return pos.Add(neg).Mean().Neg()
}
