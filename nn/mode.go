package nn

import "github.com/donghquinn/go-tensor/tensor"

// TrainingSetter is implemented by modules that behave differently during
// training and evaluation, such as Dropout, and by containers that forward the
// mode to their children. Modules start in training mode, as in PyTorch.
type TrainingSetter interface {
	SetTraining(training bool)
}

// Train puts m and its children in training mode, like model.train().
func Train(m Module) { setTraining(m, true) }

// Eval puts m and its children in evaluation mode, like model.eval().
func Eval(m Module) { setTraining(m, false) }

func setTraining(m Module, training bool) {
	if s, ok := m.(TrainingSetter); ok {
		s.SetTraining(training)
	}
}

func (s *Sequential) SetTraining(training bool) {
	for _, l := range s.Layers {
		setTraining(l, training)
	}
}

// Dropout randomly zeroes elements with probability P during training and
// scales the rest by 1/(1-P), like torch.nn.Dropout. In evaluation mode it
// passes the input through unchanged.
type Dropout struct {
	P    float64
	eval bool
}

// NewDropout creates a Dropout layer with drop probability p in [0, 1).
func NewDropout(p float64) *Dropout {
	if p < 0 || p >= 1 {
		panic("nn: dropout probability must be in [0, 1)")
	}
	return &Dropout{P: p}
}

func (d *Dropout) SetTraining(training bool) { d.eval = !training }

func (d *Dropout) Forward(x *tensor.Tensor) *tensor.Tensor {
	if d.eval || d.P == 0 {
		return x
	}
	// The mask is drawn on the CPU so a seed gives the same mask on every device.
	m := tensor.Rand(x.Shape()...).Float64s()
	scale := 1 / (1 - d.P)
	for i, r := range m {
		if r < d.P {
			m[i] = 0
		} else {
			m[i] = scale
		}
	}
	mask := tensor.FromFloat64(m, x.Shape()...).ToDType(x.DType()).To(x.Device())
	return x.Mul(mask)
}

func (d *Dropout) Parameters() []*tensor.Tensor { return nil }
