// Package nn provides neural-network building blocks, modeled after torch.nn.
package nn

import (
	"math"

	"github.com/donghquinn/go-tensor/tensor"
)

// Module is a layer or model that maps an input tensor to an output tensor
// and owns trainable parameters.
type Module interface {
	Forward(x *tensor.Tensor) *tensor.Tensor
	Parameters() []*tensor.Tensor
}

// Linear applies y = x @ Weightᵀ + Bias. Weight has shape (out, in) and Bias
// has shape (out), matching torch.nn.Linear.
type Linear struct {
	Weight *tensor.Tensor
	Bias   *tensor.Tensor
}

// NewLinear creates a Linear layer initialized like PyTorch, with values drawn
// uniformly from [-1/sqrt(in), 1/sqrt(in)].
func NewLinear(in, out int) *Linear {
	bound := 1 / math.Sqrt(float64(in))
	return &Linear{
		Weight: tensor.Uniform(-bound, bound, out, in).SetRequiresGrad(true),
		Bias:   tensor.Uniform(-bound, bound, out).SetRequiresGrad(true),
	}
}

func (l *Linear) Forward(x *tensor.Tensor) *tensor.Tensor {
	return x.MatMul(l.Weight.T()).Add(l.Bias)
}

func (l *Linear) Parameters() []*tensor.Tensor { return []*tensor.Tensor{l.Weight, l.Bias} }

// Sequential chains modules, feeding each output into the next.
type Sequential struct {
	Layers []Module
}

// NewSequential creates a Sequential container from layers.
func NewSequential(layers ...Module) *Sequential { return &Sequential{Layers: layers} }

func (s *Sequential) Forward(x *tensor.Tensor) *tensor.Tensor {
	for _, l := range s.Layers {
		x = l.Forward(x)
	}
	return x
}

func (s *Sequential) Parameters() []*tensor.Tensor {
	var ps []*tensor.Tensor
	for _, l := range s.Layers {
		ps = append(ps, l.Parameters()...)
	}
	return ps
}

// activation adapts a parameter-free tensor method into a Module.
type activation func(*tensor.Tensor) *tensor.Tensor

func (a activation) Forward(x *tensor.Tensor) *tensor.Tensor { return a(x) }
func (activation) Parameters() []*tensor.Tensor              { return nil }

// NewReLU returns a ReLU activation module.
func NewReLU() Module { return activation((*tensor.Tensor).ReLU) }

// NewSigmoid returns a Sigmoid activation module.
func NewSigmoid() Module { return activation((*tensor.Tensor).Sigmoid) }

// NewTanh returns a Tanh activation module.
func NewTanh() Module { return activation((*tensor.Tensor).Tanh) }

// NewSoftmax returns a module applying softmax over the last dimension.
func NewSoftmax() Module { return activation((*tensor.Tensor).Softmax) }

// NewFlatten returns a module that reshapes (batch, ...) into (batch, features).
func NewFlatten() Module { return activation((*tensor.Tensor).Flatten) }
