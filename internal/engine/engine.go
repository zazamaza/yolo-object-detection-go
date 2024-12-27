package engine

type IEngine interface {
	SetInput(input *[]float32)
	GetOutput() []float32
	Run() error
	Destroy()
}

type ExecutionProvider int

const (
	CPU ExecutionProvider = iota
	CUDA
	OpenVINO
	TensorRT
	DirectML
	CoreML
)
