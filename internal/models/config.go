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
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
		"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
		"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
		"suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
		"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
		"bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
		"cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
		"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
		"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
	}
	configuration.Version = YOLOv11

	return configuration
}
