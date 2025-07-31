package model

type YOLOVersion int

const (
	YOLOv5 YOLOVersion = iota
	YOLOv8
	YOLOv10
	YOLOv11
)

type YOLOConfiguration struct {
	ModelPath   string
	InputName   string
	OutputName  string
	InputShape  []int64
	OutputShape []int64
	Classes     []string
	Version     YOLOVersion
}

func NewYOLOConfiguration() YOLOConfiguration {
	configuration := YOLOConfiguration{}
	configuration.ModelPath = "model.onnx"
	configuration.InputName = "images"
	configuration.OutputName = "output0"
	configuration.InputShape = []int64{1, 3, 640, 640}
	configuration.OutputShape = []int64{1, 84, 8400}
	configuration.Classes = []string{
		"Head", "Enemy", "Flashed"
	}
	configuration.Version = YOLOv11

	return configuration
}
