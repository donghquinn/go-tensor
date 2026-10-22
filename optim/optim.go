// Package optim implements gradient-based optimizers, modeled after torch.optim.
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
	velocity [][]float64
}

// NewSGD creates an SGD optimizer over params.
func NewSGD(params []*tensor.Tensor, cfg SGDConfig) *SGD {
	v := make([][]float64, len(params))
	for i, p := range params {
		v[i] = make([]float64, p.Numel())
	}
	return &SGD{SGDConfig: cfg, params: params, velocity: v}
}

func (o *SGD) ZeroGrad() { zeroGrad(o.params) }

func (o *SGD) Step() {
	for i, p := range o.params {
		g := p.Grad()
		if g == nil {
			continue
		}
		pd, gd, v := p.Data(), g.Data(), o.velocity[i]
		for j := range pd {
			d := gd[j] + o.WeightDecay*pd[j]
			if o.Momentum != 0 {
				v[j] = o.Momentum*v[j] + d
				d = v[j]
			}
			pd[j] -= o.LR * d
		}
	}
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
	m, v   [][]float64
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
	m := make([][]float64, len(params))
	v := make([][]float64, len(params))
	for i, p := range params {
		m[i] = make([]float64, p.Numel())
		v[i] = make([]float64, p.Numel())
	}
	return &Adam{AdamConfig: cfg, params: params, m: m, v: v}
}

func (o *Adam) ZeroGrad() { zeroGrad(o.params) }

func (o *Adam) Step() {
	o.step++
	bc1 := 1 - math.Pow(o.Beta1, float64(o.step))
	bc2 := 1 - math.Pow(o.Beta2, float64(o.step))
	for i, p := range o.params {
		g := p.Grad()
		if g == nil {
			continue
		}
		pd, gd, m, v := p.Data(), g.Data(), o.m[i], o.v[i]
		for j := range pd {
			d := gd[j] + o.WeightDecay*pd[j]
			m[j] = o.Beta1*m[j] + (1-o.Beta1)*d
			v[j] = o.Beta2*v[j] + (1-o.Beta2)*d*d
			pd[j] -= o.LR * (m[j] / bc1) / (math.Sqrt(v[j]/bc2) + o.Eps)
		}
	}
}
