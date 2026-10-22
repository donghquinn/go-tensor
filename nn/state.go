package nn

import (
	"fmt"
	"slices"
	"strconv"
	"strings"

	"github.com/donghquinn/go-tensor/tensor"
)

// NamedParameter is a parameter together with its state-dict name.
type NamedParameter struct {
	Name  string
	Value *tensor.Tensor
}

// Named is implemented by modules that name their parameters, like
// torch.nn.Module.named_parameters. Container modules prefix their children's
// names with the child's name or index.
type Named interface {
	NamedParameters() []NamedParameter
}

// NamedParameters returns m's parameters with PyTorch-compatible names, e.g.
// "0.weight" for the weight of the first layer of a Sequential. Modules that
// do not implement Named get their parameters numbered by position.
func NamedParameters(m Module) []NamedParameter {
	if n, ok := m.(Named); ok {
		return n.NamedParameters()
	}
	var out []NamedParameter
	for i, p := range m.Parameters() {
		out = append(out, NamedParameter{strconv.Itoa(i), p})
	}
	return out
}

func (l *Linear) NamedParameters() []NamedParameter {
	return []NamedParameter{{"weight", l.Weight}, {"bias", l.Bias}}
}

func (c *Conv2d) NamedParameters() []NamedParameter {
	return []NamedParameter{{"weight", c.Weight}, {"bias", c.Bias}}
}

func (s *Sequential) NamedParameters() []NamedParameter {
	var out []NamedParameter
	for i, l := range s.Layers {
		for _, p := range NamedParameters(l) {
			out = append(out, NamedParameter{strconv.Itoa(i) + "." + p.Name, p.Value})
		}
	}
	return out
}

// StateDict returns m's parameters keyed by name, like model.state_dict().
// The tensors share memory with the model.
func StateDict(m Module) map[string]*tensor.Tensor {
	sd := map[string]*tensor.Tensor{}
	for _, p := range NamedParameters(m) {
		sd[p.Name] = p.Value
	}
	return sd
}

// StateDictError reports keys that did not line up in LoadStateDict.
type StateDictError struct {
	Missing    []string // in the model but not in the state dict
	Unexpected []string // in the state dict but not in the model
}

func (e *StateDictError) Error() string {
	var parts []string
	if len(e.Missing) > 0 {
		parts = append(parts, "missing keys: "+strings.Join(e.Missing, ", "))
	}
	if len(e.Unexpected) > 0 {
		parts = append(parts, "unexpected keys: "+strings.Join(e.Unexpected, ", "))
	}
	return "nn: state dict does not match model: " + strings.Join(parts, "; ")
}

// LoadStateDict copies values from sd into m's parameters, matching by name,
// like model.load_state_dict. With strict, any missing or unexpected key is a
// *StateDictError; otherwise only matching keys are loaded. A shape mismatch
// is always an error. Nothing is modified unless the whole load succeeds.
func LoadStateDict(m Module, sd map[string]*tensor.Tensor, strict bool) error {
	params := NamedParameters(m)
	var missing []string
	var load []NamedParameter
	known := map[string]bool{}
	for _, p := range params {
		known[p.Name] = true
		src, ok := sd[p.Name]
		if !ok {
			missing = append(missing, p.Name)
			continue
		}
		if !slices.Equal(src.Shape(), p.Value.Shape()) {
			return fmt.Errorf("nn: %s has shape %v in the state dict but %v in the model", p.Name, src.Shape(), p.Value.Shape())
		}
		load = append(load, NamedParameter{p.Name, src})
	}
	if strict {
		var unexpected []string
		for k := range sd {
			if !known[k] {
				unexpected = append(unexpected, k)
			}
		}
		if len(missing) > 0 || len(unexpected) > 0 {
			slices.Sort(unexpected)
			return &StateDictError{Missing: missing, Unexpected: unexpected}
		}
	}

	byName := map[string]*tensor.Tensor{}
	for _, p := range params {
		byName[p.Name] = p.Value
	}
	for _, p := range load {
		byName[p.Name].CopyFrom(p.Value)
	}
	return nil
}
