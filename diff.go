package diff

import (
	"math"
	"sync"
)

type Point struct {
	Loc   float64
	Coeff float64
}

type F interface {
	F(x float64) float64
}

type Func func(float64) float64

func (f Func) F(x float64) float64 {
	return f(x)
}

type Method struct {
	Stencil []Point
	Order   int // Decides the order of the method (will divide by step^order)
}

type Settings struct {
	FofX       float64 // Value of the function at the current location (set to NaN if unknown)
	Step       float64 // step size
	Concurrent bool    // Should the function calls be executed concurrently
}

var DefaultSettings = Settings{
	FofX: math.NaN(),
	Step: 1e-6,
}

// Estimate estimates the derivative of the function f at the given location using
// the specified method and settings
func Estimate(f F, x float64, method Method, settings Settings) float64 {
	var deriv float64
	if !settings.Concurrent {
		for _, pt := range method.Stencil {
			if !math.IsNaN(settings.FofX) && pt.Loc == 0 {
				deriv += pt.Coeff * settings.FofX
				continue
			}
			deriv += pt.Coeff * f.F(x+settings.Step*pt.Loc)
		}
		return deriv / math.Pow(settings.Step, float64(method.Order))
	}

	wg := &sync.WaitGroup{}
	mux := &sync.Mutex{}
	for _, pt := range method.Stencil {
		if !math.IsNaN(settings.FofX) && pt.Loc == 0 {
			mux.Lock()
			deriv += pt.Coeff * settings.FofX
			mux.Unlock()
			continue
		}
		wg.Add(1)
		go func(pt Point) {
			defer wg.Done()
			fofx := f.F(x + settings.Step*pt.Loc)
			mux.Lock()
			defer mux.Unlock()
			deriv += pt.Coeff * fofx

		}(pt)
	}
	wg.Wait()
	return deriv / math.Pow(settings.Step, float64(method.Order))
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
