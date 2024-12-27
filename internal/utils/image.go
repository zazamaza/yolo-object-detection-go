package utils

import (
	"image"
	"image/color"
	"math"

	"github.com/disintegration/imaging"
)

type IImageUtils interface {
	Letterbox(img image.Image, inputSize int) (image.Image, float32, int, int)
	NMSBoxes(boxes *[]image.Rectangle, scores *[]float32, scoreThreshold, nmsThreshold float32) *[]int
	Iou(box1, box2 image.Rectangle) float64
}

type ImageUtils struct{}

type Size struct {
	Width  int
	Height int
}

func (i *ImageUtils) Letterbox(img image.Image, inputSize int) (image.Image, float32, int, int) {
	origWidth := img.Bounds().Dx()
	origHeight := img.Bounds().Dy()

	scale := math.Min(float64(inputSize)/float64(origWidth), float64(inputSize)/float64(origHeight))
	newWidth := int(math.Round(float64(origWidth) * scale))
	newHeight := int(math.Round(float64(origHeight) * scale))

	dw := (inputSize - newWidth) / 2
	dh := (inputSize - newHeight) / 2

	resizedImg := imaging.Resize(img, newWidth, newHeight, imaging.NearestNeighbor)
	paddedImg := imaging.New(inputSize, inputSize, color.RGBA{114, 114, 114, 255})
	paddedImg = imaging.Paste(paddedImg, resizedImg, image.Pt(dw, dh))

	return paddedImg, float32(scale), dw, dh
}

func (iu *ImageUtils) NMSBoxes(boxes *[]image.Rectangle, scores *[]float32, scoreThreshold, nmsThreshold float32) *[]int {
	filteredBoxes := []image.Rectangle{}
	filteredScores := []float32{}
	indices := []int{}

	for idx, score := range *scores {
		if score > scoreThreshold {
			filteredBoxes = append(filteredBoxes, (*boxes)[idx])
			filteredScores = append(filteredScores, score)
			indices = append(indices, idx)
		}
	}

	selectedIndices := []int{}
	for len(indices) > 0 {
		maxIdx := 0
		for i, score := range filteredScores {
			if score > filteredScores[maxIdx] {
				maxIdx = i
			}
		}

		selectedIndices = append(selectedIndices, indices[maxIdx])

		currentBox := filteredBoxes[maxIdx]
		newIndices := []int{}
		newBoxes := []image.Rectangle{}
		newScores := []float32{}

		for i, idx := range indices {
			if i != maxIdx {
				if iu.Iou(currentBox, filteredBoxes[i]) < float64(nmsThreshold) {
					newIndices = append(newIndices, idx)
					newBoxes = append(newBoxes, filteredBoxes[i])
					newScores = append(newScores, filteredScores[i])
				}
			}
		}

		indices = newIndices
		filteredBoxes = newBoxes
		filteredScores = newScores
	}

	return &selectedIndices
}

func (i *ImageUtils) Iou(box1, box2 image.Rectangle) float64 {
	intersection := box1.Intersect(box2)
	interArea := float64(intersection.Dx() * intersection.Dy())
	if interArea <= 0 {
		return 0.0
	}

	box1Area := float64(box1.Dx() * box1.Dy())
	box2Area := float64(box2.Dx() * box2.Dy())
	unionArea := box1Area + box2Area - interArea

	return interArea / unionArea
}
