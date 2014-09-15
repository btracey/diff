package scattered

import (
	"math"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

// WeightMethod is a function that computes the weight of point t when fitting
// to point s
type WeightNDer interface {
	WeightND(s, t []float64) float64
}

type Weighter interface {
	Weight(s, t float64) float64
}

type Uniform struct{}

func (Uniform) WeightND([]float64, []float64) float64 {
	return 1
}

func (Uniform) Weight(float64, float64) float64 {
	return 1
}

// InvSqDistance finds the weight using the inverse squared euclidean distance
// between the points
type InvSqDist struct{}

func (InvSqDist) WeightND(s []float64, t []float64) float64 {
	dist := floats.Distance(s, t, 2)
	return 1 / (dist * dist)
}

func (InvSqDist) Weight(s, t float64) float64 {
	return 1 / ((s - t) * (s - t))
}

type SqExponential struct {
	Scale float64
}

func (g SqExponential) WeightND(s, t []float64) float64 {
	dist := floats.Distance(s, t, 2)
	norm := dist / g.Scale
	return math.Exp(-norm * norm)
}

func (g SqExponential) Weight(s, t float64) float64 {
	dist := s - t
	norm := dist / g.Scale
	return math.Exp(-norm * norm)
}

type Point struct {
	Location float64
	Value    float64
	Weight   float64
}

type PointND struct {
	Location []float64
	Value    float64
	Weight   float64
}

// Intercept constrains the fit to the data.
type Intercept struct {
	Force bool // ForceIntercept forces the fit to go through settings.OriginValue
	Value float64
}

// Plane estimates the derivative at x by fitting a plane to the data points with
// the given weights and function values.
//
// If intercept.Force is true, the plane is forced to go through intercept.Value
// at x.
//
// deriv is stored in place

func Plane(x []float64, points []*PointND, intercept Intercept, deriv []float64) {
	nPoints := len(points)

	for _, pt := range points {
		if len(x) != len(pt.Location) {
			panic("scattered: slice length mismatch")
		}
	}
	if len(deriv) != len(x) {
		panic("scattered: slice length mismatch")
	}

	A := mat64.NewDense(nPoints, len(x), nil)

	if intercept.Force {
		for i, pt := range points {
			loc := A.RowView(i)
			copy(loc, pt.Location)
			floats.Sub(loc, x)
		}
	} else {
		for i, pt := range points {
			loc := A.RowView(i)
			copy(loc, pt.Location)
		}
	}

	b := mat64.NewDense(nPoints, 1, nil)
	if intercept.Force {
		for i, pt := range points {
			b.Set(i, 0, pt.Value-intercept.Value)
		}
	} else {
		for i, pt := range points {
			b.Set(i, 0, pt.Value)
		}
	}

	// Multiply by the weights to do a weighted solve
	for i, pt := range points {
		row := A.RowView(i)
		floats.Scale(pt.Weight, row)
		v := b.At(i, 0) * pt.Weight
		b.Set(i, 0, v)
	}
	ans := mat64.Solve(A, b)
	for i := range deriv {
		deriv[i] = ans.At(i, 0)
	}
}

// Line function similarly
