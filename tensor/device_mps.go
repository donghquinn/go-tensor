//go:build darwin && mps

package tensor

import (
	"github.com/donghquinn/go-tensor/internal/backend"
	"github.com/donghquinn/go-tensor/internal/mps"
)

func init() {
	registerBackend(backend.MPS, func(int) (backend.Backend, error) { return mps.New() })
}
