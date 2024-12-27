package utils

import "fmt"

type BoundingBox struct {
	Label          string
	Confidence     float32
	X1, Y1, X2, Y2 float32
}

func (b *BoundingBox) String() string {
	return fmt.Sprintf("Object %s (confidence %f): (%f, %f), (%f, %f)",
		b.Label, b.Confidence, b.X1, b.Y1, b.X2, b.Y2)
}
