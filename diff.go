package diff

import (
	"math"
)

// Settings is a struct that gives the settings for the finite difference schemes
type Settings struct {
	Order      int       // What order derivative (first order, second order, etc.)
	FofX       float64   // Value of the function at the current location
	Step       float64   // Step size
	Stencil    int       // How many points to use. Exact definition varies with algorithm
	Concurrent bool      // Should all of the function calls be executed concurrently?
	Coeffs     []float64 // Coefficients for the stencil (if they have been precomputed)
}

/*
// NewSettings returns a new settings structure with the default values provided
func NewSettings() *Settings {

}
*/

// DefaultSettings represents the most common settings for the difference schemes
// and is provided for ease-of-use
var DefaultSettings Settings = Settings{
	Order:      1,
	FofX:       math.Nan(),
	Step:       1e-6,
	Concurrent: false,
	Stencil:    1,
}

// Central estimates the derivative of f(x) with respect to x by using
// a symmetric stencil on both sides of x
func Central(f func(float64) float64, x float64, settings Settings) float64 {
	// Shortcut the most common path
	if !settings.Concurrent && settings.Order == 1 && settings.Stencil == 1 {
		return (f(x+step) - f(x-step)) / (2 * step)
	}

}

// Forward estimates df/dx at x using a forward step. fx is the
// value of f at x
func Forward(f func(float64) float64, x, fx, step float64) float64 {
	return (f(x+step) - fx) / step
}

// Backward estimates df/dx at x using a forward step. fx is the
// value of f at x
func Backward(f func(float64) float64, x, fx, step float64) float64 {
	return (fx - f(x+step)) / step
}
