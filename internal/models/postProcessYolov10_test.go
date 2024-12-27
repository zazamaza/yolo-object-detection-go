package model

import (
	"testing"

	"github.com/Gass-AI/yolo-object-detection-go/internal/utils"
)

func TestYOLOv10PostProcess_PostProcess(t *testing.T) {
	yolo := &YOLOv10PostProcess{
		InputShape:  640,
		OutputShape: 3,
		Classes:     []string{"person", "bicycle", "car"},
	}

	output := []float32{
		10, 20, 50, 60, 0.9, 0, // Detection 1: person, high confidence
		15, 25, 55, 65, 0.7, 1, // Detection 2: bicycle, medium confidence
		100, 200, 150, 250, 0.4, 2, // Detection 3: car, low confidence (below threshold)
	}

	originalWidth := 1280
	originalHeight := 720
	scoreThreshold := float32(0.5)
	nmsThreshold := float32(0.4)
	scale := float32(0.5)
	dw := 5
	dh := 5

	expectedResults := []utils.BoundingBox{
		{
			Label:      "person",
			Confidence: 0.9,
			X1:         10,
			Y1:         30,
			X2:         90,
			Y2:         110,
		},
		{
			Label:      "bicycle",
			Confidence: 0.7,
			X1:         20,
			Y1:         40,
			X2:         100,
			Y2:         120,
		},
	}

	results := yolo.PostProcess(output, originalWidth, originalHeight, scoreThreshold, nmsThreshold, scale, dw, dh)

	if len(results) != len(expectedResults) {
		t.Errorf("Expected %d results, got %d", len(expectedResults), len(results))
	}

	for i, result := range results {
		if result.Label != expectedResults[i].Label ||
			result.Confidence != expectedResults[i].Confidence ||
			result.X1 != expectedResults[i].X1 ||
			result.Y1 != expectedResults[i].Y1 ||
			result.X2 != expectedResults[i].X2 ||
			result.Y2 != expectedResults[i].Y2 {
			t.Errorf("Result %d mismatch. Expected %+v, got %+v", i, expectedResults[i], result)
		}
	}
}
