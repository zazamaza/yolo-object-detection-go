package model

import (
	"math"

	"github.com/zazamaza/yolo-object-detection-go/internal/utils"
)

type YOLOv10PostProcess struct {
	InputShape  int
	OutputShape int
	Classes     []string
}

func (yo *YOLOv10PostProcess) PostProcess(output []float32,
	originalWidth, originalHeight int,
	scoreThreshold, nmsThreshold, scale float32,
	dw, dh int,
) []utils.BoundingBox {

	results := make([]utils.BoundingBox, 0, yo.OutputShape)
	for index := 0; index < len(output)/6; index++ {
		x1 := output[index*6]
		y1 := output[index*6+1]
		x2 := output[index*6+2]
		y2 := output[index*6+3]
		probability := output[index*6+4]
		classID := int(output[index*6+5])

		if probability < scoreThreshold {
			continue
		}

		xc, yc := (x2+x1)/2.0, (y2+y1)/2.0
		w, h := x2-x1, y2-y1

		x1 = (xc - w/2 - float32(dw)) / scale
		y1 = (yc - h/2 - float32(dh)) / scale
		x2 = (xc + w/2 - float32(dw)) / scale
		y2 = (yc + h/2 - float32(dh)) / scale

		x1 = float32(math.Max(0, math.Min(float64(x1), float64(originalWidth))))
		y1 = float32(math.Max(0, math.Min(float64(y1), float64(originalHeight))))
		x2 = float32(math.Max(0, math.Min(float64(x2), float64(originalWidth))))
		y2 = float32(math.Max(0, math.Min(float64(y2), float64(originalHeight))))

		results = append(results, utils.BoundingBox{
			Label:      yo.Classes[classID],
			Confidence: probability,
			X1:         x1,
			Y1:         y1,
			X2:         x2,
			Y2:         y2,
		})
	}

	return results
}
