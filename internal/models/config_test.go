package model

import (
	"reflect"
	"testing"
)

func TestNewYOLOConfiguration(t *testing.T) {
	expectedConfig := YOLOConfiguration{
		ModelPath:   "model.onnx",
		InputName:   "images",
		OutputName:  "output0",
		InputShape:  []int64{1, 3, 640, 640},
		OutputShape: []int64{1, 84, 8400},
		Classes: []string{
			"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
			"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
			"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
			"suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
			"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
			"bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
			"cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
			"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
			"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
		},
		Version: YOLOv11,
	}

	config := NewYOLOConfiguration()

	if config.ModelPath != expectedConfig.ModelPath {
		t.Errorf("ModelPath mismatch. Expected %s, got %s", expectedConfig.ModelPath, config.ModelPath)
	}

	if config.InputName != expectedConfig.InputName {
		t.Errorf("InputName mismatch. Expected %s, got %s", expectedConfig.InputName, config.InputName)
	}

	if config.OutputName != expectedConfig.OutputName {
		t.Errorf("OutputName mismatch. Expected %s, got %s", expectedConfig.OutputName, config.OutputName)
	}

	if !reflect.DeepEqual(config.InputShape, expectedConfig.InputShape) {
		t.Errorf("InputShape mismatch. Expected %v, got %v", expectedConfig.InputShape, config.InputShape)
	}

	if !reflect.DeepEqual(config.OutputShape, expectedConfig.OutputShape) {
		t.Errorf("OutputShape mismatch. Expected %v, got %v", expectedConfig.OutputShape, config.OutputShape)
	}

	if !reflect.DeepEqual(config.Classes, expectedConfig.Classes) {
		t.Errorf("Classes mismatch. Expected %v, got %v", expectedConfig.Classes, config.Classes)
	}

	if config.Version != expectedConfig.Version {
		t.Errorf("Version mismatch. Expected %v, got %v", expectedConfig.Version, config.Version)
	}
}
