package model

import (
	"image"
	"image/color"
	"reflect"
	"testing"
)

type MockImageUtils struct {
	LetterboxFunc func(img image.Image, inputSize int) (image.Image, float32, int, int)
	NMSBoxesFunc  func(boxes *[]image.Rectangle, scores *[]float32, scoreThreshold, nmsThreshold float32) *[]int
	IouFunc       func(box1, box2 image.Rectangle) float64
}

func (m *MockImageUtils) Letterbox(img image.Image, inputSize int) (image.Image, float32, int, int) {
	if m.LetterboxFunc != nil {
		return m.LetterboxFunc(img, inputSize)
	}
	return nil, 0, 0, 0
}

func (m *MockImageUtils) NMSBoxes(boxes *[]image.Rectangle, scores *[]float32, scoreThreshold, nmsThreshold float32) *[]int {
	if m.NMSBoxesFunc != nil {
		return m.NMSBoxesFunc(boxes, scores, scoreThreshold, nmsThreshold)
	}
	return &[]int{}
}

func (m *MockImageUtils) Iou(box1, box2 image.Rectangle) float64 {
	if m.IouFunc != nil {
		return m.IouFunc(box1, box2)
	}
	return 0.0
}

func TestPreProcess(t *testing.T) {
	mockUtils := &MockImageUtils{}

	// Define the input image
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			img.Set(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: 0, A: 255})
		}
	}

	// Mock the Letterbox method
	inputSize := 128
	mockedImage := image.NewRGBA(image.Rect(0, 0, inputSize, inputSize))
	mockUtils.LetterboxFunc = func(img image.Image, inputSize int) (image.Image, float32, int, int) {
		return mockedImage, 1.28, 10, 10
	}

	// Instantiate YOLOPreProcess
	preprocessor := YOLOPreProcess{
		InputShape: inputSize,
		ImageUtils: mockUtils,
	}

	dst := make([]float32, inputSize*inputSize*3) // R, G, B channels
	scale, dw, dh := preprocessor.PreProcess(img, &dst)

	// Assertions
	if scale != 1.28 {
		t.Errorf("Expected scale to be 1.28, got %f", scale)
	}
	if dw != 10 {
		t.Errorf("Expected dw to be 10, got %d", dw)
	}
	if dh != 10 {
		t.Errorf("Expected dh to be 10, got %d", dh)
	}
}

func TestProcessRGBA(t *testing.T) {
	preprocessor := YOLOPreProcess{
		InputShape: 128,
		ImageUtils: nil, // No dependency needed for this test
	}

	img := image.NewRGBA(image.Rect(0, 0, 128, 128))
	for y := 0; y < 128; y++ {
		for x := 0; x < 128; x++ {
			img.Set(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}

	dst := make([]float32, 128*128*3) // R, G, B channels
	preprocessor.processRGBA(img, &dst)

	expected := float32(0) / 255.0
	if dst[0] != expected {
		t.Errorf("Expected first red channel value to be %f, got %f", expected, dst[0])
	}
}

func TestProcessGeneric(t *testing.T) {
	preprocessor := YOLOPreProcess{
		InputShape: 64,
		ImageUtils: nil, // No dependency needed for this test
	}

	img := image.NewRGBA(image.Rect(0, 0, 64, 64))
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			img.Set(x, y, color.RGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}

	dst := make([]float32, 64*64*3) // R, G, B channels
	preprocessor.processGeneric(img, &dst)

	expected := []float32{
		float32(0) / 255.0,
		float32(1) / 255.0,
		float32(2) / 255.0,
	}

	if !reflect.DeepEqual(dst[:3], expected) {
		t.Errorf("First three red channel values mismatch. Expected %v, got %v", expected, dst[:3])
	}
}

func TestProcessNRGBA(t *testing.T) {
	preprocessor := YOLOPreProcess{
		InputShape: 64,
		ImageUtils: nil, // No dependency needed for this test
	}

	img := image.NewNRGBA(image.Rect(0, 0, 64, 64))
	for y := 0; y < 64; y++ {
		for x := 0; x < 64; x++ {
			img.Set(x, y, color.NRGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}

	dst := make([]float32, 64*64*3) // R, G, B channels
	preprocessor.processNRGBA(img, &dst)

	expected := []float32{
		float32(0) / 255.0,
		float32(1) / 255.0,
		float32(2) / 255.0,
	}

	if !reflect.DeepEqual(dst[:3], expected) {
		t.Errorf("First three red channel values mismatch. Expected %v, got %v", expected, dst[:3])
	}
}

func TestPreProcessWithNRGBA(t *testing.T) {
	mockUtils := &MockImageUtils{}

	// Define the input image
	img := image.NewNRGBA(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			img.Set(x, y, color.NRGBA{R: uint8(x), G: uint8(y), B: uint8(x + y), A: 255})
		}
	}

	// Mock the Letterbox method
	inputSize := 128
	mockedImage := image.NewNRGBA(image.Rect(0, 0, inputSize, inputSize))
	mockUtils.LetterboxFunc = func(img image.Image, inputSize int) (image.Image, float32, int, int) {
		return mockedImage, 1.28, 10, 10
	}

	// Instantiate YOLOPreProcess
	preprocessor := YOLOPreProcess{
		InputShape: inputSize,
		ImageUtils: mockUtils,
	}

	dst := make([]float32, inputSize*inputSize*3) // R, G, B channels
	scale, dw, dh := preprocessor.PreProcess(img, &dst)

	// Assertions
	if scale != 1.28 {
		t.Errorf("Expected scale to be 1.28, got %f", scale)
	}
	if dw != 10 {
		t.Errorf("Expected dw to be 10, got %d", dw)
	}
	if dh != 10 {
		t.Errorf("Expected dh to be 10, got %d", dh)
	}
}

func TestPreProcessWithFallback(t *testing.T) {
	mockUtils := &MockImageUtils{}

	// Define the input image
	img := image.NewGray(image.Rect(0, 0, 100, 100))
	for y := 0; y < 100; y++ {
		for x := 0; x < 100; x++ {
			img.SetGray(x, y, color.Gray{Y: uint8(x)})
		}
	}

	// Mock the Letterbox method
	inputSize := 128
	mockedImage := image.NewGray(image.Rect(0, 0, inputSize, inputSize))
	mockUtils.LetterboxFunc = func(img image.Image, inputSize int) (image.Image, float32, int, int) {
		return mockedImage, 1.28, 10, 10
	}

	// Instantiate YOLOPreProcess
	preprocessor := YOLOPreProcess{
		InputShape: inputSize,
		ImageUtils: mockUtils,
	}

	dst := make([]float32, inputSize*inputSize*3) // R, G, B channels
	scale, dw, dh := preprocessor.PreProcess(img, &dst)

	// Assertions
	if scale != 1.28 {
		t.Errorf("Expected scale to be 1.28, got %f", scale)
	}
	if dw != 10 {
		t.Errorf("Expected dw to be 10, got %d", dw)
	}
	if dh != 10 {
		t.Errorf("Expected dh to be 10, got %d", dh)
	}
}
