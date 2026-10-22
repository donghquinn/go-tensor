package tensor

import (
	"fmt"
	"sync/atomic"
)

var gradDisabled atomic.Bool

// NoGrad runs fn with gradient tracking disabled, like `with torch.no_grad():`.
// The setting is process-wide, so do not train concurrently with a NoGrad block.
func NoGrad(fn func()) {
	prev := gradDisabled.Swap(true)
	defer gradDisabled.Store(prev)
	fn()
}

// IsGradEnabled reports whether operations currently record the graph.
func IsGradEnabled() bool { return !gradDisabled.Load() }

// BackwardFunc receives the gradient of the output and returns one gradient
// per input, in order. A nil entry means no gradient flows to that input.
type BackwardFunc func(grad *Tensor) []*Tensor

type node struct {
	op       string
	inputs   []*Tensor
	backward BackwardFunc
}

// CustomOp wraps an already computed result as the output of a differentiable
// operation, like subclassing torch.autograd.Function. data is used without
// copying. backward runs with gradient tracking disabled.
func CustomOp(op string, data []float64, shape []int, inputs []*Tensor, backward BackwardFunc) *Tensor {
	if numel(shape) != len(data) {
		panic(fmt.Sprintf("tensor: %s produced %d values for shape %v", op, len(data), shape))
	}
	return newResult(data, shape, op, inputs, backward)
}

func newResult(data []float64, shape []int, op string, inputs []*Tensor, backward BackwardFunc) *Tensor {
	out := fromData(data, shape)
	if !IsGradEnabled() {
		return out
	}
	for _, in := range inputs {
		if in.requiresGrad {
			out.requiresGrad = true
			out.ctx = &node{op: op, inputs: inputs, backward: backward}
			break
		}
	}
	return out
}

// Backward computes the gradient of t with respect to every leaf tensor that
// requires grad and accumulates it into their Grad. For non-scalar outputs a
// gradient of the same shape as t must be supplied.
func (t *Tensor) Backward(grad ...*Tensor) {
	if !t.requiresGrad {
		panic("tensor: Backward called on a tensor that does not require grad")
	}
	var seed *Tensor
	if len(grad) > 0 {
		seed = grad[0]
		if !equalShape(seed.shape, t.shape) {
			panic(fmt.Sprintf("tensor: Backward grad shape %v does not match output shape %v", seed.shape, t.shape))
		}
	} else {
		if len(t.data) != 1 {
			panic("tensor: grad can be implicitly created only for scalar outputs")
		}
		seed = OnesLike(t)
	}

	order := topoSort(t)
	grads := map[*Tensor]*Tensor{t: seed}

	prev := gradDisabled.Swap(true)
	defer gradDisabled.Store(prev)

	for i := len(order) - 1; i >= 0; i-- {
		v := order[i]
		g, ok := grads[v]
		if !ok {
			continue
		}
		delete(grads, v)
		if v.ctx == nil {
			if v.grad == nil {
				v.grad = g.Clone()
			} else {
				v.grad = v.grad.Add(g)
			}
			continue
		}
		inGrads := v.ctx.backward(g)
		for j, in := range v.ctx.inputs {
			if !in.requiresGrad || inGrads[j] == nil {
				continue
			}
			if acc, ok := grads[in]; ok {
				grads[in] = acc.Add(inGrads[j])
			} else {
				grads[in] = inGrads[j]
			}
		}
	}
}

// topoSort returns the tensors that require grad and lead to root, ordered so
// that every tensor appears after all of its inputs.
func topoSort(root *Tensor) []*Tensor {
	var order []*Tensor
	visited := map[*Tensor]bool{}
	var visit func(*Tensor)
	visit = func(t *Tensor) {
		if visited[t] || !t.requiresGrad {
			return
		}
		visited[t] = true
		if t.ctx != nil {
			for _, in := range t.ctx.inputs {
				visit(in)
			}
		}
		order = append(order, t)
	}
	visit(root)
	return order
}
