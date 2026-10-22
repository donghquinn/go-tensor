package tensor

import (
	"fmt"
	"strconv"
	"strings"
	"sync"

	"github.com/donghquinn/go-tensor/internal/backend"
	"github.com/donghquinn/go-tensor/internal/cpu"
)

// DType is the element type of a tensor, like torch.dtype.
type DType = backend.DType

const (
	Float32 = backend.Float32
	Float64 = backend.Float64
)

// Device is where a tensor's data lives and its operations run, like
// torch.device.
type Device = backend.Device

var (
	// CPU is the host processor. It is always available.
	CPU = Device{Type: backend.CPU}
	// MPS is the Apple-silicon GPU via Metal. It requires building with -tags mps.
	MPS = Device{Type: backend.MPS}
)

// CUDA returns the NVIDIA GPU with the given ordinal. It requires building
// with -tags cuda.
func CUDA(index int) Device { return Device{Type: backend.CUDA, Index: index} }

// ParseDevice parses a device name: "cpu", "mps", "cuda" or "cuda:N".
func ParseDevice(s string) (Device, error) {
	switch s = strings.ToLower(strings.TrimSpace(s)); {
	case s == "cpu":
		return CPU, nil
	case s == "mps":
		return MPS, nil
	case s == "cuda":
		return CUDA(0), nil
	case strings.HasPrefix(s, "cuda:"):
		i, err := strconv.Atoi(s[len("cuda:"):])
		if err == nil && i >= 0 {
			return CUDA(i), nil
		}
	}
	return Device{}, fmt.Errorf("tensor: unknown device %q", s)
}

// buildTags names the build tag that compiles in each device's backend.
var buildTags = map[backend.DeviceType]string{backend.MPS: "mps", backend.CUDA: "cuda"}

var (
	registryMu sync.Mutex
	factories  = map[backend.DeviceType]func(index int) (backend.Backend, error){
		backend.CPU: func(int) (backend.Backend, error) { return cpu.Backend{}, nil },
	}
	backends = map[Device]backend.Backend{}
	initErrs = map[Device]error{}
)

// registerBackend is called from the init function of a build-tagged file
// that compiles in a device.
func registerBackend(t backend.DeviceType, f func(index int) (backend.Backend, error)) {
	registryMu.Lock()
	defer registryMu.Unlock()
	factories[t] = f
}

func backendFor(d Device) (backend.Backend, error) {
	registryMu.Lock()
	defer registryMu.Unlock()
	if b, ok := backends[d]; ok {
		return b, nil
	}
	if err, ok := initErrs[d]; ok {
		return nil, err
	}
	f, ok := factories[d.Type]
	if !ok {
		return nil, fmt.Errorf("tensor: %v support is not compiled in; rebuild with -tags %s", d, buildTags[d.Type])
	}
	b, err := f(d.Index)
	if err != nil {
		err = fmt.Errorf("tensor: %v is unavailable: %w", d, err)
		initErrs[d] = err
		return nil, err
	}
	backends[d] = b
	return b, nil
}

func mustBackend(d Device) backend.Backend {
	b, err := backendFor(d)
	if err != nil {
		panic(err.Error())
	}
	return b
}

// IsAvailable reports whether tensors can be placed on d, like
// torch.cuda.is_available() and torch.backends.mps.is_available().
func IsAvailable(d Device) bool {
	_, err := backendFor(d)
	return err == nil
}

// Synchronize waits until all queued work on d has finished, like
// torch.cuda.synchronize(). Reading values from a tensor synchronizes
// automatically, so this is only needed for timing.
func Synchronize(d Device) { mustBackend(d).Sync() }

var (
	defaultsMu    sync.RWMutex
	defaultDType  = Float32
	defaultDevice = CPU
)

// SetDefaultDType sets the dtype of newly created tensors, like
// torch.set_default_dtype. The initial default is Float32.
func SetDefaultDType(dt DType) {
	defaultsMu.Lock()
	defaultDType = dt
	defaultsMu.Unlock()
}

// DefaultDType returns the dtype of newly created tensors.
func DefaultDType() DType {
	defaultsMu.RLock()
	defer defaultsMu.RUnlock()
	return defaultDType
}

// SetDefaultDevice sets the device of newly created tensors, like
// torch.set_default_device. It panics if d is unavailable.
func SetDefaultDevice(d Device) {
	mustBackend(d)
	defaultsMu.Lock()
	defaultDevice = d
	defaultsMu.Unlock()
}

// DefaultDevice returns the device of newly created tensors.
func DefaultDevice() Device {
	defaultsMu.RLock()
	defer defaultsMu.RUnlock()
	return defaultDevice
}
