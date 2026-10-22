// Command mnist trains a convolutional network or a multilayer perceptron on
// the MNIST handwritten digits.
//
//	go run ./examples/mnist -model cnn -epochs 3
//	go run -tags mps ./examples/mnist -device mps   # Apple-silicon GPU
package main

import (
	"flag"
	"fmt"
	"log"
	"path/filepath"
	"time"

	"github.com/donghquinn/go-tensor/data"
	"github.com/donghquinn/go-tensor/datasets/mnist"
	"github.com/donghquinn/go-tensor/nn"
	"github.com/donghquinn/go-tensor/optim"
	"github.com/donghquinn/go-tensor/tensor"
)

func main() {
	arch := flag.String("model", "cnn", "architecture: cnn or mlp")
	deviceName := flag.String("device", "cpu", "device to train on: cpu, mps or cuda")
	root := flag.String("data", ".cache/mnist", "directory for the MNIST files (downloaded if missing)")
	epochs := flag.Int("epochs", 3, "number of training epochs")
	batchSize := flag.Int("batch", 64, "mini-batch size")
	lr := flag.Float64("lr", 1e-3, "Adam learning rate")
	out := flag.String("save", "", "where to save the trained model (default: <data>/<model>.gotn)")
	flag.Parse()
	if *out == "" {
		*out = filepath.Join(*root, *arch+".gotn")
	}
	newModel, ok := models[*arch]
	if !ok {
		log.Fatalf("unknown model %q (want cnn or mlp)", *arch)
	}

	dev, err := tensor.ParseDevice(*deviceName)
	if err != nil {
		log.Fatal(err)
	}
	if !tensor.IsAvailable(dev) {
		log.Fatalf("device %v is not available (GPU backends need a build tag, e.g. go run -tags mps)", dev)
	}

	tensor.ManualSeed(0)

	train, err := mnist.Load(*root, true, true)
	if err != nil {
		log.Fatal(err)
	}
	test, err := mnist.Load(*root, false, true)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("train: %d images, test: %d images\n", train.Len(), test.Len())

	model := newModel()
	nn.To(model, dev)
	opt := optim.NewAdam(model.Parameters(), optim.AdamConfig{LR: *lr})
	loader := data.NewDataLoader(train, *batchSize, true)

	for epoch := 1; epoch <= *epochs; epoch++ {
		start := time.Now()
		total, batches := 0.0, 0
		nn.Train(model)
		for x, y := range loader.All() {
			loss := nn.CrossEntropyLoss(model.Forward(x.To(dev)), y)

			opt.ZeroGrad()
			loss.Backward()
			opt.Step()

			total += loss.Item()
			batches++
		}
		fmt.Printf("epoch %d  loss %.4f  test accuracy %.2f%%  (%s)\n",
			epoch, total/float64(batches), 100*accuracy(model, test, dev), time.Since(start).Round(time.Millisecond))
	}

	if err := nn.SaveFile(*out, model); err != nil {
		log.Fatal(err)
	}
	restored := newModel()
	nn.To(restored, dev)
	if err := nn.LoadFile(*out, restored); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("saved model to %s; reloaded copy scores %.2f%%\n", *out, 100*accuracy(restored, test, dev))
}

var models = map[string]func() nn.Module{
	// 1x28x28 -> 16x14x14 -> 32x7x7 -> 128 -> 10
	"cnn": func() nn.Module {
		return nn.NewSequential(
			nn.NewConv2d(1, 16, 3, 1, 1),
			nn.NewReLU(),
			nn.NewMaxPool2d(2, 0),
			nn.NewConv2d(16, 32, 3, 1, 1),
			nn.NewReLU(),
			nn.NewMaxPool2d(2, 0),
			nn.NewFlatten(),
			nn.NewDropout(0.25),
			nn.NewLinear(32*7*7, 128),
			nn.NewReLU(),
			nn.NewLinear(128, 10),
		)
	},
	"mlp": func() nn.Module {
		return nn.NewSequential(
			nn.NewFlatten(),
			nn.NewLinear(28*28, 128),
			nn.NewReLU(),
			nn.NewLinear(128, 10),
		)
	},
}

func accuracy(model nn.Module, ds data.Dataset, dev tensor.Device) float64 {
	nn.Eval(model)
	correct := 0
	tensor.NoGrad(func() {
		for x, y := range data.NewDataLoader(ds, 1000, false).All() {
			for i, p := range model.Forward(x.To(dev)).Argmax() {
				if p == y[i] {
					correct++
				}
			}
		}
	})
	return float64(correct) / float64(ds.Len())
}
