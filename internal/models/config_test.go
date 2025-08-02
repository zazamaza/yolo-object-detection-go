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
		InputShape:  []int64{1, 3, 320, 320},
		OutputShape: []int64{1, 7, 2100},
		Classes: []string{
			"Head", "Enemy", "Flashed",
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
