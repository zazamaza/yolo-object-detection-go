package main

import (
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"

	yolo "github.com/Gass-AI/yolo-object-detection-go"
)

func loadImageFile(filePath string) (image.Image, error) {
	f, e := os.Open(filePath)
	if e != nil {
		return nil, fmt.Errorf("error opening %s: %w", filePath, e)
	}
	defer f.Close()
	pic, _, e := image.Decode(f)
	if e != nil {
		return nil, fmt.Errorf("error decoding %s: %w", filePath, e)
	}
	return pic, nil
}

func main() {
	model, error := yolo.NewYOLOv11("./models/yolo11n.onnx")
	if error != nil {
		fmt.Println(error)
	}

	defer model.Destroy()

	img, e := loadImageFile("./assets/image.jpg")
	if e != nil {
		fmt.Printf("Error loading input image: %s\n", e)
	}

	boxes, err := model.Predict(img, 0.2, 0.5) //scoreThreshold: 0.2, nmsThreshold: 0.5
	if err != nil {
		fmt.Printf("Error %s \n", err)
	}
	for i, box := range boxes {
		fmt.Printf("Object %d: %s x1:%.2f y1:%.2f x2:%.2f y2:%.2f Confidence:%f\n", i, box.Label, box.X1, box.Y1, box.X2, box.X2, box.Confidence)
	}
}
