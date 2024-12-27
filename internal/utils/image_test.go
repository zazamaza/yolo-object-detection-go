package utils

import (
	"image"
	"math"
	"testing"
)

func TestLetterbox(t *testing.T) {
	imgUtils := ImageUtils{}
	img := image.NewRGBA(image.Rect(0, 0, 200, 100))

	inputSize := 300
	expectedScale := float32(1.5)
	expectedDW := 0
	expectedDH := 75

	resultImg, scale, dw, dh := imgUtils.Letterbox(img, inputSize)

	if scale != expectedScale {
		t.Errorf("expected scale %f, got %f", expectedScale, scale)
	}
	if dw != expectedDW {
		t.Errorf("expected dw %d, got %d", expectedDW, dw)
	}
	if dh != expectedDH {
		t.Errorf("expected dh %d, got %d", expectedDH, dh)
	}

	bounds := resultImg.Bounds()
	if bounds.Dx() != inputSize || bounds.Dy() != inputSize {
		t.Errorf("expected image size %dx%d, got %dx%d", inputSize, inputSize, bounds.Dx(), bounds.Dy())
	}
}

func TestIou(t *testing.T) {
	imgUtils := ImageUtils{}

	box1 := image.Rect(0, 0, 100, 100)
	box2 := image.Rect(50, 50, 150, 150)
	expectedIou := 0.142857

	iou := imgUtils.Iou(box1, box2)

	if math.Abs(iou-expectedIou) > 1e-6 {
		t.Errorf("expected IOU %f, got %f", expectedIou, iou)
	}
}

func TestNMSBoxes(t *testing.T) {
	imgUtils := ImageUtils{}

	boxes := []image.Rectangle{
		image.Rect(0, 0, 100, 100),
		image.Rect(10, 10, 110, 110),
		image.Rect(200, 200, 300, 300),
	}
	scores := []float32{0.9, 0.8, 0.7}

	expectedIndices := []int{0, 2}

	resultIndices := imgUtils.NMSBoxes(&boxes, &scores, 0.5, 0.3)

	if len(*resultIndices) != len(expectedIndices) {
		t.Fatalf("expected %d indices, got %d", len(expectedIndices), len(*resultIndices))
	}
	for i, idx := range *resultIndices {
		if idx != expectedIndices[i] {
			t.Errorf("expected index %d, got %d", expectedIndices[i], idx)
		}
	}
}
