package weights_test

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"testing"

	"github.com/donghquinn/go-tensor/internal/weighttest"
	"github.com/donghquinn/go-tensor/nn"
	"github.com/donghquinn/go-tensor/tensor"
	"github.com/donghquinn/go-tensor/weights"
)

// newCNN mirrors the PyTorch model in testdata/gen_fixtures.py, so parameter
// names line up: 0.weight, 0.bias, 4.weight, 4.bias.
func newCNN() nn.Module {
	return nn.NewSequential(
		nn.NewConv2d(1, 2, 3, 1, 1),
		nn.NewReLU(),
		nn.NewMaxPool2d(2, 0),
		nn.NewFlatten(),
		nn.NewLinear(2*4*4, 3),
	)
}

// TestPyTorchModelInEveryFormat loads the same trained PyTorch model from
// every supported format and checks the Go forward pass reproduces PyTorch's
// output for the same input.
func TestPyTorchModelInEveryFormat(t *testing.T) {
	var io struct{ Input, Output weighttest.Entry }
	b, err := os.ReadFile(weighttest.Testdata("cnn_io.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &io); err != nil {
		t.Fatal(err)
	}
	x := tensor.New(io.Input.Values, io.Input.Shape...)

	files := map[string]weights.Format{
		"cnn.pth":           weights.PyTorch,
		"cnn.safetensors":   weights.Safetensors,
		"cnn.npz":           weights.NPZ,
		"cnn.onnx":          weights.ONNX,
		"cnn_external.onnx": weights.ONNX,
		"cnn.gguf":          weights.GGUF,
	}
	for file, format := range files {
		t.Run(file, func(t *testing.T) {
			path := weighttest.Testdata(file)
			if f, err := weights.Detect(path); err != nil || f != format {
				t.Fatalf("Detect = %q, %v; want %q", f, err, format)
			}
			sd, err := weights.Load(path)
			if err != nil {
				t.Fatal(err)
			}
			model := newCNN()
			if err := nn.LoadStateDict(model, sd, true); err != nil {
				t.Fatal(err)
			}
			var y *tensor.Tensor
			tensor.NoGrad(func() { y = model.Forward(x) })
			for i, want := range io.Output.Values {
				// PyTorch computed in float32; Go computes in float64.
				if got := y.Float64s()[i]; math.Abs(got-want) > 1e-5 {
					t.Fatalf("output[%d] = %v, PyTorch gave %v", i, got, want)
				}
			}
		})
	}
}

func TestDetectEveryFixture(t *testing.T) {
	files := map[string]weights.Format{
		"zoo.pth":         weights.PyTorch,
		"scripted.pt":     weights.PyTorch,
		"zoo.safetensors": weights.Safetensors,
		"zoo.onnx":        weights.ONNX,
		"zoo.npz":         weights.NPZ,
		"fortran.npy":     weights.NPY,
		"zoo.gguf":        weights.GGUF,
	}
	for file, want := range files {
		got, err := weights.Detect(weighttest.Testdata(file))
		if err != nil || got != want {
			t.Errorf("Detect(%s) = %q, %v; want %q", file, got, err, want)
		}
		if _, err := weights.Load(weighttest.Testdata(file)); err != nil {
			t.Errorf("Load(%s): %v", file, err)
		}
	}
	if _, err := weights.Detect(weighttest.Testdata("gen_fixtures.py")); err == nil {
		t.Error("a Python script was detected as a weight file")
	}
}

func TestLoadStateDictStrictness(t *testing.T) {
	sd, err := weights.Load(weighttest.Testdata("cnn.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	// A model with an extra layer: "6.weight"/"6.bias" are missing from sd.
	bigger := nn.NewSequential(append(newCNN().(*nn.Sequential).Layers, nn.NewReLU(), nn.NewLinear(3, 2))...)
	before := append([]float64(nil), bigger.Layers[0].Parameters()[0].Float64s()...)

	err = nn.LoadStateDict(bigger, sd, true)
	var sde *nn.StateDictError
	if !errors.As(err, &sde) || len(sde.Missing) != 2 || sde.Missing[0] != "6.weight" {
		t.Fatalf("strict load error = %v", err)
	}
	if got := bigger.Layers[0].Parameters()[0].Float64s(); got[0] != before[0] {
		t.Fatal("a failed strict load modified the model")
	}

	if err := nn.LoadStateDict(bigger, sd, false); err != nil {
		t.Fatalf("non-strict load: %v", err)
	}
	if got := bigger.Layers[0].Parameters()[0].Float64s(); got[0] != sd["0.weight"].Float64s()[0] {
		t.Fatal("non-strict load did not copy matching parameters")
	}

	wrong := nn.NewSequential(nn.NewConv2d(1, 4, 3, 1, 1))
	if err := nn.LoadStateDict(wrong, sd, false); err == nil {
		t.Fatal("shape mismatch accepted")
	}
}
