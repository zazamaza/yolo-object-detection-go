package model

import (
	"reflect"
	"testing"

	"github.com/Gass-AI/yolo-object-detection-go/internal/utils"
)

func TestPostProcess(t *testing.T) {
	mockUtils := &utils.ImageUtils{}
	classes := []string{"person", "car"}

	yolo := YOLOPostProcess{
		OutputShape: 2,
		ImageUtils:  mockUtils,
		Classes:     classes,
	}

	output := []float32{
		100, 200, // x, y for box 1
		50, 60, // width, height for box 1
		0.9, 0.1, // class probabilities for box 1
		300, 400, // x, y for box 2
		70, 80, // width, height for box 2
		0.2, 0.8, // class probabilities for box 2
	}

	expectedBoxes := []utils.BoundingBox{
		{
			Label:      "person",
			Confidence: 80,
			X1:         199.95,
			Y1:         0,
			X2:         200.05,
			Y2:         260,
		},
		{
			Label:      "person",
			Confidence: 70,
			X1:         99.55,
			Y1:         0,
			X2:         100.45,
			Y2:         200,
		},
	}

	result := yolo.PostProcess(
		output,
		640, // origWidth
		480, // origHeight
		0.5, // scoreThresh
		0.5, // nmsThresh
		1.0, // scale
		0,   // dw
		0,   // dh
	)

	if len(result) != len(expectedBoxes) {
		t.Errorf("Expected %d boxes, got %d", len(expectedBoxes), len(result))
		return
	}

	for i, expected := range expectedBoxes {
		if !reflect.DeepEqual(expected, result[i]) {
			t.Errorf("Box %d mismatch:\nexpected: %+v\ngot: %+v", i, expected, result[i])
		}
	}
}
