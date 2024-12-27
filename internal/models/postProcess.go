package model

import (
	"image"
	"math"

	"github.com/Gass-AI/yolo-object-detection-go/internal/utils"
)

type YOLOPostProcess struct {
	OutputShape int
	ImageUtils  utils.IImageUtils
	Classes     []string
}

func (yo *YOLOPostProcess) PostProcess(output []float32,
	originalWidth, originalHeight int,
	scoreThreshold, nmsThreshold, scale float32,
	dw, dh int,
) []utils.BoundingBox {

	boundingBoxes := make([]utils.BoundingBox, 0, yo.OutputShape)

	var classID int
	var probability float32

	classesLength := len(yo.Classes)
	for index := 0; index < yo.OutputShape; index++ {
		probability = -1e9
		for col := 0; col < classesLength; col++ {
			currentProb := output[yo.OutputShape*(col+4)+index]
			if currentProb > probability {
				probability = currentProb
				classID = col
			}
		}

		if probability < scoreThreshold {
			continue
		}

		xc, yc := output[index], output[yo.OutputShape+index]
		w, h := output[2*yo.OutputShape+index], output[3*yo.OutputShape+index]

		x1 := (xc - w/2 - float32(dw)) / scale
		y1 := (yc - h/2 - float32(dh)) / scale
		x2 := (xc + w/2 - float32(dw)) / scale
		y2 := (yc + h/2 - float32(dh)) / scale

		x1 = float32(math.Max(0, math.Min(float64(x1), float64(originalWidth))))
		y1 = float32(math.Max(0, math.Min(float64(y1), float64(originalHeight))))
		x2 = float32(math.Max(0, math.Min(float64(x2), float64(originalWidth))))
		y2 = float32(math.Max(0, math.Min(float64(y2), float64(originalHeight))))

		boundingBoxes = append(boundingBoxes, utils.BoundingBox{
			Label:      yo.Classes[classID],
			Confidence: probability,
			X1:         x1,
			Y1:         y1,
			X2:         x2,
			Y2:         y2,
		})
	}

	boxes := make([]image.Rectangle, len(boundingBoxes))
	scores := make([]float32, len(boundingBoxes))
	for i, b := range boundingBoxes {
		boxes[i] = image.Rect(int(b.X1), int(b.Y1), int(b.X2), int(b.Y2))
		scores[i] = b.Confidence
	}

	indices := yo.ImageUtils.NMSBoxes(&boxes,
		&scores,
		scoreThreshold,
		nmsThreshold,
	)

	results := make([]utils.BoundingBox, len(*indices))
	for i, idx := range *indices {
		results[i] = boundingBoxes[idx]
	}

	return results
}
