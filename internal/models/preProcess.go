package model

import (
	"image"

	"github.com/Gass-AI/yolo-object-detection-go/internal/utils"
)

type YOLOPreProcess struct {
	InputShape int
	ImageUtils utils.IImageUtils
}

func (yo *YOLOPreProcess) PreProcess(img image.Image, dst *[]float32) (float32, int, int) {
	img, scale, dw, dh := yo.ImageUtils.Letterbox(img, yo.InputShape)

	// Get the specific image type for better performance
	switch typedImg := img.(type) {
	case *image.RGBA:
		yo.processRGBA(typedImg, dst)
	case *image.NRGBA:
		yo.processNRGBA(typedImg, dst)
	default:
		// Fallback for other image types
		yo.processGeneric(img, dst)
	}
	return scale, dw, dh
}

func (yo *YOLOPreProcess) processRGBA(img *image.RGBA, dst *[]float32) {
	channelSize := yo.InputShape * yo.InputShape
	redChannel := (*dst)[0:channelSize]
	greenChannel := (*dst)[channelSize : channelSize*2]
	blueChannel := (*dst)[channelSize*2 : channelSize*3]

	pixels := img.Pix
	stride := img.Stride
	const div255 = 1.0 / 255.0

	for y := 0; y < yo.InputShape; y++ {
		offset := y * stride
		for x := 0; x < yo.InputShape; x++ {
			i := y*yo.InputShape + x
			pixelOffset := offset + x*4

			redChannel[i] = float32(pixels[pixelOffset]) * div255
			greenChannel[i] = float32(pixels[pixelOffset+1]) * div255
			blueChannel[i] = float32(pixels[pixelOffset+2]) * div255
		}
	}
}

func (yo *YOLOPreProcess) processNRGBA(img *image.NRGBA, dst *[]float32) {
	channelSize := yo.InputShape * yo.InputShape
	redChannel := (*dst)[0:channelSize]
	greenChannel := (*dst)[channelSize : channelSize*2]
	blueChannel := (*dst)[channelSize*2 : channelSize*3]

	pixels := img.Pix
	stride := img.Stride
	const div255 = 1.0 / 255.0

	for y := 0; y < yo.InputShape; y++ {
		offset := y * stride
		for x := 0; x < yo.InputShape; x++ {
			i := y*yo.InputShape + x
			pixelOffset := offset + x*4

			redChannel[i] = float32(pixels[pixelOffset]) * div255
			greenChannel[i] = float32(pixels[pixelOffset+1]) * div255
			blueChannel[i] = float32(pixels[pixelOffset+2]) * div255
		}
	}
}

func (yo *YOLOPreProcess) processGeneric(img image.Image, dst *[]float32) {
	channelSize := yo.InputShape * yo.InputShape
	redChannel := (*dst)[0:channelSize]
	greenChannel := (*dst)[channelSize : channelSize*2]
	blueChannel := (*dst)[channelSize*2 : channelSize*3]

	const div255 = 1.0 / 255.0

	for y := 0; y < yo.InputShape; y++ {
		for x := 0; x < yo.InputShape; x++ {
			i := y*yo.InputShape + x
			r, g, b, _ := img.At(x, y).RGBA()
			redChannel[i] = float32(r>>8) * div255
			greenChannel[i] = float32(g>>8) * div255
			blueChannel[i] = float32(b>>8) * div255
		}
	}
}
