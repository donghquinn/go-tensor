// Package optim implements gradient-based optimizers, modeled after torch.optim.
//
// Updates are expressed as tensor operations, so they run on the device of
// the parameters, and optimizer state lives on that device too.
package optim

import (
	"math"

	"github.com/donghquinn/go-tensor/tensor"
)

// Optimizer updates parameters in place from their accumulated gradients.
type Optimizer interface {
	// Step applies one update using the current gradients.
	Step()
	// ZeroGrad clears the gradients of every parameter.
	ZeroGrad()
}

func zeroGrad(params []*tensor.Tensor) {
	for _, p := range params {
		p.ZeroGrad()
	}
}

// SGDConfig holds the hyperparameters for SGD.
type SGDConfig struct {
	LR          float64
	Momentum    float64
	WeightDecay float64
}

// SGD is stochastic gradient descent with optional momentum and weight decay.
type SGD struct {
	SGDConfig
	params   []*tensor.Tensor
	velocity []*tensor.Tensor // created on the first step
}

// NewSGD creates an SGD optimizer over params.
func NewSGD(params []*tensor.Tensor, cfg SGDConfig) *SGD {
	return &SGD{SGDConfig: cfg, params: params, velocity: make([]*tensor.Tensor, len(params))}
}

func (o *SGD) ZeroGrad() { zeroGrad(o.params) }

func (o *SGD) Step() {
	tensor.NoGrad(func() {
		for i, p := range o.params {
			g := p.Grad()
			if g == nil {
				continue
			}
			d := g
			if o.WeightDecay != 0 {
				d = d.Add(p.MulScalar(o.WeightDecay))
			}
			if o.Momentum != 0 {
				if o.velocity[i] == nil {
					o.velocity[i] = d.Clone()
				} else {
					o.velocity[i] = o.velocity[i].MulScalar(o.Momentum).Add(d)
				}
				d = o.velocity[i]
			}
			p.CopyFrom(p.Sub(d.MulScalar(o.LR)))
		}
	})
}

// AdamConfig holds the hyperparameters for Adam. Zero values are replaced by
// PyTorch's defaults (LR 1e-3, betas 0.9/0.999, eps 1e-8).
type AdamConfig struct {
	LR          float64
	Beta1       float64
	Beta2       float64
	Eps         float64
	WeightDecay float64
}

// Adam implements the Adam optimizer (Kingma & Ba, 2014).
type Adam struct {
	AdamConfig
	params []*tensor.Tensor
	m, v   []*tensor.Tensor // created on the first step
	step   int
}

// NewAdam creates an Adam optimizer over params.
func NewAdam(params []*tensor.Tensor, cfg AdamConfig) *Adam {
	if cfg.LR == 0 {
		cfg.LR = 1e-3
	}
	if cfg.Beta1 == 0 {
		cfg.Beta1 = 0.9
	}
	if cfg.Beta2 == 0 {
		cfg.Beta2 = 0.999
	}
	if cfg.Eps == 0 {
		cfg.Eps = 1e-8
	}
	return &Adam{AdamConfig: cfg, params: params,
		m: make([]*tensor.Tensor, len(params)), v: make([]*tensor.Tensor, len(params))}
}

func (o *Adam) ZeroGrad() { zeroGrad(o.params) }

func (o *Adam) Step() {
	o.step++
	bc1 := 1 - math.Pow(o.Beta1, float64(o.step))
	bc2 := 1 - math.Pow(o.Beta2, float64(o.step))
	tensor.NoGrad(func() {
		for i, p := range o.params {
			g := p.Grad()
			if g == nil {
				continue
			}
			if o.WeightDecay != 0 {
				g = g.Add(p.MulScalar(o.WeightDecay))
			}
			if o.m[i] == nil {
				o.m[i], o.v[i] = tensor.ZerosLike(p), tensor.ZerosLike(p)
			}
			o.m[i] = o.m[i].MulScalar(o.Beta1).Add(g.MulScalar(1 - o.Beta1))
			o.v[i] = o.v[i].MulScalar(o.Beta2).Add(g.Mul(g).MulScalar(1 - o.Beta2))
			denom := o.v[i].MulScalar(1 / bc2).Sqrt().AddScalar(o.Eps)
			p.CopyFrom(p.Sub(o.m[i].MulScalar(o.LR / bc1).Div(denom)))
		}
	})
}
