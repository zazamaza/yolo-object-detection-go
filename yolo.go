package yolo

import (
	"fmt"
	"image"

	engine "github.com/zazamaza/yolo-object-detection-go/internal/engine"
	models "github.com/zazamaza/yolo-object-detection-go/internal/models"
	"github.com/zazamaza/yolo-object-detection-go/internal/utils"
)

type YOLO struct {
	preProcessor  models.IPreProcess
	engine        engine.IEngine
	postProcessor models.IPostProcess
	inputShape    int
	outputShape   int
	version       models.YOLOVersion
}

func NewYOLOv5(modelPath string) (*YOLO, error) {
	configuration := models.NewYOLOConfiguration()
	configuration.ModelPath = modelPath
	configuration.Version = models.YOLOv5
	return newYOLOHelper(&configuration)
}

func NewYOLOv8(modelPath string) (*YOLO, error) {
	configuration := models.NewYOLOConfiguration()
	configuration.ModelPath = modelPath
	configuration.Version = models.YOLOv8
	return newYOLOHelper(&configuration)
}

func NewYOLOv10(modelPath string) (*YOLO, error) {
	configuration := models.NewYOLOConfiguration()
	configuration.ModelPath = modelPath
	configuration.Version = models.YOLOv10
	configuration.InputShape = []int64{1, 3, 640, 640}
	configuration.OutputShape = []int64{1, 300, 6}
	return newYOLOHelper(&configuration)
}

func NewYOLOv11(modelPath string) (*YOLO, error) {
	configuration := models.NewYOLOConfiguration()
	configuration.ModelPath = modelPath
	configuration.Version = models.YOLOv11
	return newYOLOHelper(&configuration)
}

func NewYOLOWithConfiguration(configuration *models.YOLOConfiguration) (*YOLO, error) {
	fmt.Println(configuration.Version)
	return newYOLOHelper(configuration)
}

func newYOLOHelper(configuration *models.YOLOConfiguration) (*YOLO, error) {
	engine, err := engine.NewEngine(
		configuration.ModelPath,
		configuration.InputName,
		configuration.OutputName,
		configuration.InputShape,
		configuration.OutputShape,
		engine.CPU,
	)
	if err != nil {
		return nil, err
	}

	imageUtils := utils.ImageUtils{}
	inputShape := int(configuration.InputShape[2])
	outputShape := int(configuration.OutputShape[2])

	var postProcessor models.IPostProcess
	if configuration.Version == models.YOLOv10 {
		postProcessor = &models.YOLOv10PostProcess{
			InputShape:  inputShape,
			OutputShape: outputShape,
			Classes:     configuration.Classes,
		}
	} else {
		postProcessor = &models.YOLOPostProcess{
			OutputShape: outputShape,
			Classes:     configuration.Classes,
			ImageUtils:  &imageUtils,
		}
	}

	return &YOLO{
		preProcessor: &models.YOLOPreProcess{
			InputShape: inputShape,
			ImageUtils: &imageUtils,
		},
		engine:        engine,
		postProcessor: postProcessor,
		inputShape:    inputShape,
		outputShape:   outputShape,
		version:       configuration.Version,
	}, nil
}

func (yo *YOLO) Predict(img image.Image,
	scoreThreshold, nmsThreshold float32,
) ([]utils.BoundingBox, error) {

	originalWidth := img.Bounds().Canon().Dx()
	originalHeight := img.Bounds().Canon().Dy()

	channelSize := yo.inputShape * yo.inputShape
	totalSize := channelSize * 3

	inputData := make([]float32, totalSize)

	scale, dw, dh := yo.preProcessor.PreProcess(img, &inputData)
	yo.engine.SetInput(&inputData)

	err := yo.engine.Run()
	if err != nil {
		return nil, fmt.Errorf("error running ORT session: %s", err)
	}

	boxes := yo.postProcessor.PostProcess(yo.engine.GetOutput(),
		originalWidth,
		originalHeight,
		scoreThreshold,
		nmsThreshold,
		scale, dw, dh,
	)

	return boxes, nil
}

func (yo *YOLO) Destroy() {
	yo.engine.Destroy()
}
