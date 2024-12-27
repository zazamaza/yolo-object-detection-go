package model

import (
	"image"

	"github.com/Gass-AI/yolo-object-detection-go/internal/utils"
)

type IPreProcess interface {
	PreProcess(img image.Image, dst *[]float32) (float32, int, int)
}

type IModel interface {
	Predict(image.Image)
}

type IPostProcess interface {
	PostProcess(output []float32,
		originalWidth, originalHeight int,
		scoreThreshold, nmsThreshold, scale float32,
		dw, dh int,
	) []utils.BoundingBox
}
