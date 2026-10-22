//go:build darwin && mps

package weights_test

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/donghquinn/go-tensor/internal/weighttest"
	"github.com/donghquinn/go-tensor/nn"
	"github.com/donghquinn/go-tensor/tensor"
	"github.com/donghquinn/go-tensor/weights"
)

// TestPyTorchModelOnMPS loads PyTorch weights into a model on the GPU and
// checks it reproduces PyTorch's outputs.
func TestPyTorchModelOnMPS(t *testing.T) {
	var io struct{ Input, Output weighttest.Entry }
	b, err := os.ReadFile(weighttest.Testdata("cnn_io.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(b, &io); err != nil {
		t.Fatal(err)
	}
	for _, file := range []string{"cnn.pth", "cnn.safetensors", "cnn.onnx", "cnn.gguf"} {
		sd, err := weights.Load(weighttest.Testdata(file))
		if err != nil {
			t.Fatal(err)
		}
		model := newCNN()
		nn.To(model, tensor.MPS)
		if err := nn.LoadStateDict(model, sd, true); err != nil {
			t.Fatalf("%s: %v", file, err)
		}
		x := tensor.New(io.Input.Values, io.Input.Shape...).To(tensor.MPS)
		var y *tensor.Tensor
		tensor.NoGrad(func() { y = model.Forward(x) })
		if y.Device() != tensor.MPS {
			t.Fatalf("%s: output on %v", file, y.Device())
		}
		for i, got := range y.Float64s() {
			if want := io.Output.Values[i]; math.Abs(got-want) > 1e-5 {
				t.Fatalf("%s: output[%d] = %v, PyTorch gave %v", file, i, got, want)
			}
		}
	}
}
