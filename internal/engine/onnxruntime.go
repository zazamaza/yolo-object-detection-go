package engine

import (
	"fmt"
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

type ONNXRuntime struct {
	Session     *ort.AdvancedSession
	Input       *ort.Tensor[float32]
	Output      *ort.Tensor[float32]
	InputShape  []int64
	OutputShape []int64
}

func getSharedLibPath() (string, error) {
	path := os.Getenv("ONNXRUNTIME_LIB_PATH")
	if path == "" {
		return "", fmt.Errorf("ONNXRUNTIME_LIB_PATH environment variable not set")
	}
	return path, nil
}

func (e *ONNXRuntime) setupExecutionProvider(options *ort.SessionOptions, provider ExecutionProvider) error {
	switch provider {
	case CUDA:
		cudaOptions, err := ort.NewCUDAProviderOptions()
		if err != nil {
			return fmt.Errorf("error creating CUDA options: %w", err)
		}
		defer cudaOptions.Destroy()

		if err := cudaOptions.Update(map[string]string{"device_id": "0"}); err != nil {
			return fmt.Errorf("error updating CUDA options: %w", err)
		}

		return options.AppendExecutionProviderCUDA(cudaOptions)

	case TensorRT:
		trtOptions, err := ort.NewTensorRTProviderOptions()
		if err != nil {
			return fmt.Errorf("error creating TensorRT options: %w", err)
		}
		defer trtOptions.Destroy()

		if err := trtOptions.Update(map[string]string{"device_id": "0"}); err != nil {
			return fmt.Errorf("error updating TensorRT options: %w", err)
		}

		return options.AppendExecutionProviderTensorRT(trtOptions)

	case OpenVINO:
		openVinoOptions := map[string]string{
			"device_type": "CPU",
		}
		return options.AppendExecutionProviderOpenVINO(openVinoOptions)

	case DirectML:
		return options.AppendExecutionProviderDirectML(0)

	case CoreML:
		return options.AppendExecutionProviderCoreML(0)
	case CPU:
		return options.AppendExecutionProviderDirectML(0)

	default:
		return fmt.Errorf("unsupported execution provider: %v", provider)
	}
}

func NewEngine(modelPath string,
	inputName, outputName string,
	inputShape, outputShape []int64,
	provider ExecutionProvider,
) (*ONNXRuntime, error) {

	libPath, err := getSharedLibPath()
	if err != nil {
		return nil, fmt.Errorf("error getting shared library path: %w", err)
	}

	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		return nil, fmt.Errorf("error initializing ORT environment: %w", err)
	}

	ortInputShape := ort.NewShape(inputShape...)
	inputTensor, err := ort.NewEmptyTensor[float32](ortInputShape)
	if err != nil {
		return nil, fmt.Errorf("error creating input tensor: %w", err)
	}

	ortOutputShape := ort.NewShape(outputShape...)
	outputTensor, err := ort.NewEmptyTensor[float32](ortOutputShape)
	if err != nil {
		inputTensor.Destroy()
		return nil, fmt.Errorf("error creating output tensor: %w", err)
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("error creating session options: %w", err)
	}
	defer options.Destroy()

	engine := &ONNXRuntime{
		Input:       inputTensor,
		Output:      outputTensor,
		InputShape:  inputShape,
		OutputShape: outputShape,
	}

	if err := engine.setupExecutionProvider(options, provider); err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("error setting up execution provider: %w", err)
	}

	session, err := ort.NewAdvancedSession(modelPath,
		[]string{inputName}, []string{outputName},
		[]ort.ArbitraryTensor{inputTensor},
		[]ort.ArbitraryTensor{outputTensor},
		options)
	if err != nil {
		inputTensor.Destroy()
		outputTensor.Destroy()
		return nil, fmt.Errorf("error creating session: %w", err)
	}

	engine.Session = session
	return engine, nil
}

func (e *ONNXRuntime) SetInput(input *[]float32) {
	data := e.Input.GetData()
	inputData := *input
	for i := 0; i < len(data); i++ {
		data[i] = inputData[i]
	}
}

func (e *ONNXRuntime) Run() error {
	return e.Session.Run()
}

func (e *ONNXRuntime) GetOutput() []float32 {
	return e.Output.GetData()
}

func (e *ONNXRuntime) Destroy() {
	e.Session.Destroy()
	e.Input.Destroy()
	e.Output.Destroy()
}
