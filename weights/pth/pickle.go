package pth

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"strconv"
	"strings"
)

// The unpickler below is a data-only interpreter for the pickle protocol (0-5).
// Unlike Python's pickle it never imports or runs code: GLOBAL references
// become Global values, and only a fixed allowlist of callables (the ones
// torch.save emits) is evaluated. Anything else becomes an inert Object.

// Global is a reference to a Python callable or class by module and name.
type Global struct{ Module, Name string }

func (g Global) String() string { return g.Module + "." + g.Name }

// Object is an instance of a class the unpickler does not understand. Its
// constructor arguments and BUILD state are kept so tensors inside it can
// still be found (e.g. TorchScript modules keep parameters in State).
type Object struct {
	Class Global
	Args  []any
	State any
}

// Dict is an insertion-ordered Python dict.
type Dict struct {
	Keys   []any
	Values map[any]any
}

func newDict() *Dict { return &Dict{Values: map[any]any{}} }

func (d *Dict) set(k, v any) {
	switch kk := k.(type) { // slices are not hashable
	case []byte:
		k = string(kk)
	case Tuple:
		k = fmt.Sprint([]any(kk))
	}
	if _, exists := d.Values[k]; !exists {
		d.Keys = append(d.Keys, k)
	}
	d.Values[k] = v
}

// Tuple and List are distinct so callers can tell them apart.
type (
	Tuple []any
	List  struct{ Items []any }
)

type mark struct{}

// persistentLoader resolves BINPERSID references (tensor storages).
type persistentLoader func(pid any) (any, error)

// reducer evaluates an allowlisted callable. ok=false means the callable is
// unknown and an Object should be built instead.
type reducer func(fn Global, args Tuple) (v any, ok bool, err error)

type unpickler struct {
	r          *bufio.Reader
	stack      []any
	metastack  [][]any
	memo       map[int]any
	persistent persistentLoader
	reduce     reducer
}

func unpickle(r io.Reader, persistent persistentLoader, reduce reducer) (any, error) {
	u := &unpickler{r: bufio.NewReader(r), memo: map[int]any{}, persistent: persistent, reduce: reduce}
	return u.run()
}

var errStackUnderflow = errors.New("pickle: stack underflow")

func (u *unpickler) push(v any) { u.stack = append(u.stack, v) }

func (u *unpickler) pop() (any, error) {
	if len(u.stack) == 0 {
		return nil, errStackUnderflow
	}
	v := u.stack[len(u.stack)-1]
	u.stack = u.stack[:len(u.stack)-1]
	return v, nil
}

func (u *unpickler) top() (any, error) {
	if len(u.stack) == 0 {
		return nil, errStackUnderflow
	}
	return u.stack[len(u.stack)-1], nil
}

// popMark returns the items pushed since the last MARK.
func (u *unpickler) popMark() ([]any, error) {
	if len(u.metastack) == 0 {
		return nil, errors.New("pickle: MARK not found")
	}
	items := u.stack
	u.stack = u.metastack[len(u.metastack)-1]
	u.metastack = u.metastack[:len(u.metastack)-1]
	return items, nil
}

func (u *unpickler) readN(n int) ([]byte, error) {
	if n < 0 || n > 1<<34 {
		return nil, fmt.Errorf("pickle: invalid length %d", n)
	}
	b := make([]byte, n)
	_, err := io.ReadFull(u.r, b)
	return b, err
}

func (u *unpickler) readUint(size int) (uint64, error) {
	b, err := u.readN(size)
	if err != nil {
		return 0, err
	}
	var v uint64
	for i := size - 1; i >= 0; i-- {
		v = v<<8 | uint64(b[i])
	}
	return v, nil
}

func (u *unpickler) readLine() (string, error) {
	s, err := u.r.ReadString('\n')
	return strings.TrimSuffix(s, "\n"), err
}

func (u *unpickler) run() (any, error) {
	for {
		op, err := u.r.ReadByte()
		if err != nil {
			return nil, fmt.Errorf("pickle: %w", err)
		}
		if op == '.' { // STOP
			return u.pop()
		}
		if err := u.step(op); err != nil {
			return nil, err
		}
	}
}

func (u *unpickler) step(op byte) error {
	switch op {
	case 0x80: // PROTO
		_, err := u.r.ReadByte()
		return err
	case 0x95: // FRAME
		_, err := u.readUint(8)
		return err

	case 'N':
		u.push(nil)
	case 0x88:
		u.push(true)
	case 0x89:
		u.push(false)
	case 'J': // BININT
		v, err := u.readUint(4)
		u.push(int64(int32(v)))
		return err
	case 'K': // BININT1
		v, err := u.readUint(1)
		u.push(int64(v))
		return err
	case 'M': // BININT2
		v, err := u.readUint(2)
		u.push(int64(v))
		return err
	case 0x8a, 0x8b: // LONG1, LONG4
		size := 1
		if op == 0x8b {
			size = 4
		}
		n, err := u.readUint(size)
		if err != nil {
			return err
		}
		b, err := u.readN(int(n))
		if err != nil {
			return err
		}
		u.push(decodeLong(b))
	case 'I', 'L': // INT, LONG (text)
		s, err := u.readLine()
		if err != nil {
			return err
		}
		s = strings.TrimSuffix(s, "L")
		switch s {
		case "00":
			u.push(false)
		case "01":
			u.push(true)
		default:
			v, err := strconv.ParseInt(s, 10, 64)
			if err != nil {
				return fmt.Errorf("pickle: bad INT %q", s)
			}
			u.push(v)
		}
	case 'G': // BINFLOAT (big-endian)
		b, err := u.readN(8)
		if err != nil {
			return err
		}
		u.push(math.Float64frombits(binary.BigEndian.Uint64(b)))
	case 'F': // FLOAT (text)
		s, err := u.readLine()
		if err != nil {
			return err
		}
		v, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return fmt.Errorf("pickle: bad FLOAT %q", s)
		}
		u.push(v)

	case 'X', 0x8c, 0x8d, 'T', 'U', 'B', 'C', 0x8e, 0x96: // strings and bytes
		size := map[byte]int{'X': 4, 0x8c: 1, 0x8d: 8, 'T': 4, 'U': 1, 'B': 4, 'C': 1, 0x8e: 8, 0x96: 8}[op]
		n, err := u.readUint(size)
		if err != nil {
			return err
		}
		b, err := u.readN(int(n))
		if err != nil {
			return err
		}
		switch op {
		case 'B', 'C', 0x8e, 0x96:
			u.push(b)
		default:
			u.push(string(b))
		}

	case ')':
		u.push(Tuple{})
	case 't':
		items, err := u.popMark()
		if err != nil {
			return err
		}
		u.push(Tuple(items))
	case 0x85, 0x86, 0x87: // TUPLE1-3
		n := int(op - 0x84)
		if len(u.stack) < n {
			return errStackUnderflow
		}
		t := append(Tuple{}, u.stack[len(u.stack)-n:]...)
		u.stack = u.stack[:len(u.stack)-n]
		u.push(t)
	case ']':
		u.push(&List{})
	case 'l':
		items, err := u.popMark()
		if err != nil {
			return err
		}
		u.push(&List{Items: items})
	case 'a', 'e': // APPEND, APPENDS
		var items []any
		if op == 'a' {
			v, err := u.pop()
			if err != nil {
				return err
			}
			items = []any{v}
		} else {
			var err error
			if items, err = u.popMark(); err != nil {
				return err
			}
		}
		top, err := u.top()
		if err != nil {
			return err
		}
		switch l := top.(type) {
		case *List:
			l.Items = append(l.Items, items...)
		case *Object:
			l.State = append(asSlice(l.State), items...)
		default:
			return fmt.Errorf("pickle: APPEND to %T", top)
		}
	case '}':
		u.push(newDict())
	case 'd':
		items, err := u.popMark()
		if err != nil {
			return err
		}
		d := newDict()
		if err := setItems(d, items); err != nil {
			return err
		}
		u.push(d)
	case 's', 'u': // SETITEM, SETITEMS
		var items []any
		if op == 's' {
			v, err := u.pop()
			if err != nil {
				return err
			}
			k, err := u.pop()
			if err != nil {
				return err
			}
			items = []any{k, v}
		} else {
			var err error
			if items, err = u.popMark(); err != nil {
				return err
			}
		}
		top, err := u.top()
		if err != nil {
			return err
		}
		switch d := top.(type) {
		case *Dict:
			return setItems(d, items)
		case *Object: // dict subclass of an unknown type
			sd, ok := d.State.(*Dict)
			if !ok {
				sd = newDict()
				d.State = sd
			}
			return setItems(sd, items)
		default:
			return fmt.Errorf("pickle: SETITEM on %T", top)
		}
	case 0x8f: // EMPTY_SET
		u.push(&List{})
	case 0x90: // ADDITEMS
		items, err := u.popMark()
		if err != nil {
			return err
		}
		top, err := u.top()
		if err != nil {
			return err
		}
		if l, ok := top.(*List); ok {
			l.Items = append(l.Items, items...)
		}
	case 0x91: // FROZENSET
		items, err := u.popMark()
		if err != nil {
			return err
		}
		u.push(&List{Items: items})

	case '(':
		u.metastack = append(u.metastack, u.stack)
		u.stack = nil
	case '0':
		_, err := u.pop()
		return err
	case '1':
		_, err := u.popMark()
		return err
	case '2':
		v, err := u.top()
		if err != nil {
			return err
		}
		u.push(v)

	case 'c': // GLOBAL
		mod, err := u.readLine()
		if err != nil {
			return err
		}
		name, err := u.readLine()
		if err != nil {
			return err
		}
		u.push(Global{mod, name})
	case 0x93: // STACK_GLOBAL
		name, err := u.pop()
		if err != nil {
			return err
		}
		mod, err := u.pop()
		if err != nil {
			return err
		}
		ms, ok1 := mod.(string)
		ns, ok2 := name.(string)
		if !ok1 || !ok2 {
			return errors.New("pickle: STACK_GLOBAL needs two strings")
		}
		u.push(Global{ms, ns})

	case 'R': // REDUCE
		args, err := u.pop()
		if err != nil {
			return err
		}
		fn, err := u.pop()
		if err != nil {
			return err
		}
		v, err := u.call(fn, args)
		if err != nil {
			return err
		}
		u.push(v)
	case 0x81, 0x92: // NEWOBJ, NEWOBJ_EX
		if op == 0x92 {
			if _, err := u.pop(); err != nil { // kwargs
				return err
			}
		}
		args, err := u.pop()
		if err != nil {
			return err
		}
		cls, err := u.pop()
		if err != nil {
			return err
		}
		v, err := u.call(cls, args)
		if err != nil {
			return err
		}
		u.push(v)
	case 'b': // BUILD
		state, err := u.pop()
		if err != nil {
			return err
		}
		top, err := u.top()
		if err != nil {
			return err
		}
		if o, ok := top.(*Object); ok {
			o.State = state
		}
		// BUILD on a known value (e.g. an OrderedDict's _metadata) is ignored.

	case 'Q': // BINPERSID
		pid, err := u.pop()
		if err != nil {
			return err
		}
		if u.persistent == nil {
			return errors.New("pickle: persistent id without a loader")
		}
		v, err := u.persistent(pid)
		if err != nil {
			return err
		}
		u.push(v)

	case 'q', 'r': // BINPUT, LONG_BINPUT
		size := 1
		if op == 'r' {
			size = 4
		}
		i, err := u.readUint(size)
		if err != nil {
			return err
		}
		v, err := u.top()
		u.memo[int(i)] = v
		return err
	case 'p': // PUT
		s, err := u.readLine()
		if err != nil {
			return err
		}
		i, err := strconv.Atoi(s)
		if err != nil {
			return fmt.Errorf("pickle: bad PUT %q", s)
		}
		v, err := u.top()
		u.memo[i] = v
		return err
	case 0x94: // MEMOIZE
		v, err := u.top()
		u.memo[len(u.memo)] = v
		return err
	case 'h', 'j': // BINGET, LONG_BINGET
		size := 1
		if op == 'j' {
			size = 4
		}
		i, err := u.readUint(size)
		if err != nil {
			return err
		}
		return u.get(int(i))
	case 'g': // GET
		s, err := u.readLine()
		if err != nil {
			return err
		}
		i, err := strconv.Atoi(s)
		if err != nil {
			return fmt.Errorf("pickle: bad GET %q", s)
		}
		return u.get(i)

	default:
		return fmt.Errorf("pickle: unsupported opcode %#x", op)
	}
	return nil
}

func (u *unpickler) get(i int) error {
	v, ok := u.memo[i]
	if !ok {
		return fmt.Errorf("pickle: memo key %d not found", i)
	}
	u.push(v)
	return nil
}

func (u *unpickler) call(fn, args any) (any, error) {
	g, ok := fn.(Global)
	if !ok {
		return &Object{Args: asSlice(args)}, nil
	}
	t, _ := args.(Tuple)
	if u.reduce != nil {
		v, ok, err := u.reduce(g, t)
		if err != nil || ok {
			return v, err
		}
	}
	return &Object{Class: g, Args: t}, nil
}

func setItems(d *Dict, items []any) error {
	if len(items)%2 != 0 {
		return errors.New("pickle: odd number of items for SETITEMS")
	}
	for i := 0; i < len(items); i += 2 {
		d.set(items[i], items[i+1])
	}
	return nil
}

func asSlice(v any) []any {
	switch v := v.(type) {
	case Tuple:
		return v
	case []any:
		return v
	case *List:
		return v.Items
	}
	return nil
}

// decodeLong decodes a little-endian two's-complement integer. Values that do
// not fit in int64 are returned as *big.Int.
func decodeLong(b []byte) any {
	if len(b) == 0 {
		return int64(0)
	}
	if len(b) <= 8 {
		var v uint64
		for i := len(b) - 1; i >= 0; i-- {
			v = v<<8 | uint64(b[i])
		}
		shift := 64 - 8*uint(len(b))
		return int64(v<<shift) >> shift
	}
	be := make([]byte, len(b))
	for i := range b {
		be[len(b)-1-i] = b[i]
	}
	n := new(big.Int).SetBytes(be)
	if b[len(b)-1]&0x80 != 0 {
		n.Sub(n, new(big.Int).Lsh(big.NewInt(1), uint(8*len(b))))
	}
	return n
}
