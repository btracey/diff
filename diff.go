package diff

import (
	"math"
	"sync"
)

// A Point is a stencil location in a difference method
type Point struct {
	Loc   float64
	Coeff float64
}

type Method struct {
	Stencil []Point
	Order   int // The order of the difference method (first derivative, second derivative, etc.)
}

// FDSettings is the settings structure for the FiniteDifference function
type FDSettings struct {
	OriginKnown bool    // Flag that the value at the origin x is known
	OriginValue float64 // Value at the origin (only used if OriginKnown is true)
	Step        float64 // step size
	Concurrent  bool    // Should the function calls be executed concurrently
	Method      Method  // Finite difference method to use
}

// DefaultFDSettings is a basic set of settings for the FiniteDifference
// function. Computes a central difference approximation for the first derivative
// of the function.
func DefaultFDSettings() *FDSettings {
	return &FDSettings{
		Step:   1e-6,
		Method: Central,
	}
}

// FiniteDifference estimates a derivative of the function f at the given location.
// The order of derivative, sample locations, and other options are specified
// by settings.
func FiniteDiffernce(f func(float64) float64, x float64, settings *FDSettings) float64 {
	var deriv float64
	method := settings.Method
	if !settings.Concurrent {
		for _, pt := range method.Stencil {
			if settings.OriginKnown && pt.Loc == 0 {
				deriv += pt.Coeff * settings.OriginValue
				continue
			}
			deriv += pt.Coeff * f(x+settings.Step*pt.Loc)
		}
		return deriv / math.Pow(settings.Step, float64(method.Order))
	}

	wg := &sync.WaitGroup{}
	mux := &sync.Mutex{}
	for _, pt := range method.Stencil {
		if settings.OriginKnown && pt.Loc == 0 {
			mux.Lock()
			deriv += pt.Coeff * settings.OriginValue
			mux.Unlock()
			continue
		}
		wg.Add(1)
		go func(pt Point) {
			defer wg.Done()
			fofx := f(x + settings.Step*pt.Loc)
			mux.Lock()
			defer mux.Unlock()
			deriv += pt.Coeff * fofx

		}(pt)
	}
	wg.Wait()
	return deriv / math.Pow(settings.Step, float64(method.Order))
}

var Forward = Method{
	Stencil: []Point{{Loc: 0, Coeff: -1}, {Loc: 1, Coeff: 1}},
	Order:   1,
}

var Backward = Method{
	Stencil: []Point{{Loc: -1, Coeff: -1}, {Loc: 0, Coeff: 1}},
	Order:   1,
}

// Central represents a first-order central difference for estimating the
// derivative of a function
var Central = Method{
	Stencil: []Point{{Loc: -1, Coeff: -0.5}, {Loc: 1, Coeff: 0.5}},
	Order:   1,
}

// Central represents a secord-order central difference for estimating the
// second derivative of a function
var Central2nd = Method{
	Stencil: []Point{{Loc: -1, Coeff: 1}, {Loc: 0, Coeff: -2}, {Loc: 1, Coeff: 1}},
	Order:   2,
}
