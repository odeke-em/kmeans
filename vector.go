package kmeans

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

var errDimensionIndexOutOfBounds = errors.New("dimension index out of bounds")

type Coordinate struct {
	dimens            []interface{}
	transformer       func(pi interface{}) float64
	memoizedSignature interface{}
}

func (c *Coordinate) MarshalJSON() ([]byte, error) {
	if c == nil {
		return []byte("[]"), nil
	}
	return json.Marshal(c.dimens)
}

type Transformer interface {
	Transform(pi interface{}) float64
}

func quantifyAsFloat64(transformer func(interface{}) float64, pi interface{}) float64 {
	if transformer != nil {
		return transformer(pi)
	}

	if pi == nil {
		return 0.0
	}

	switch t := pi.(type) {
	case float64:
		return t
	case float32:
		return float64(t)
	case int64:
		return float64(t)
	case int:
		return float64(t)
	case uintptr:
		return float64(t)
	}

	return 0.0
}

func NewCoordinate(dimens ...interface{}) (*Coordinate, error) {
	c := &Coordinate{
		dimens: dimens[:],
	}
	return c, nil
}

func (c *Coordinate) Len() int {
	if c == nil {
		return 0
	}

	return len(c.dimens)
}

var _ Vector = (*Coordinate)(nil)

type Vector interface {
	Len() int
	Dimension(i int) (interface{}, error)
	Signature() interface{}
}

func (c *Coordinate) Signature() interface{} {
	if c.memoizedSignature != nil {
		return c.memoizedSignature
	}

	// Otherwise compute it
	var allStrings []string
	for _, dimen := range c.dimens {
		allStrings = append(allStrings, fmt.Sprintf("%#v", dimen))
	}

	c.memoizedSignature = strings.Join(allStrings, "-")
	return c.memoizedSignature
}

func (c *Coordinate) Dimension(i int) (interface{}, error) {
	if i >= c.Len() {
		return nil, errDimensionIndexOutOfBounds
	}
	return c.dimens[i], nil
}
