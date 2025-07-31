# YOLO Object Detection in Golang 🌟

## 🌐 About

This is a Go package library for object detection using the YOLO (You Only Look Once) algorithm. It provides an easy-to-use interface for detecting objects in images using pre-trained YOLO models.

## ⚙️ Features

- Load and utilize YOLO models.
- Perform object detection on images.
- Customizable confidence and NMS thresholds.

## 📋 Supported YOLO Versions

| Function           | Description                        | Supported YOLO Versions |
|--------------------|------------------------------------|-------------------------|
| `NewYOLOv5`        | Creates a YOLOv5 model.             | ✅                     |
| `NewYOLOv6`        | Creates a YOLOv6 model.             | ❌                     |
| `NewYOLOv7`        | Creates a YOLOv7 model.             | ❌                     |
| `NewYOLOv8`        | Creates a YOLOv8 model.             | ✅                     |
| `NewYOLOv9`        | Creates a YOLOv9 model.             | ❌                     |
| `NewYOLOv10`       | Creates a YOLOv10 model.            | ✅                     |
| `NewYOLOv11`       | Creates a YOLOv11 model.            | ✅                     |

## 🛠️ Installation

You can install the package using:

```bash
go get github.com/zazamaza/yolo-object-detection-go
```

## 🌟 Example Usage
Here’s an example demonstrating how to use the package:

```go
package main

import (
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"

	yolo "github.com/zazamaza/yolo-object-detection-go"
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

	boxes, err := model.Predict(img, 0.2, 0.5)
	if err != nil {
		fmt.Printf("Error %s \n", err)
	}
	for i, box := range boxes {
		fmt.Printf("Object %d: %s x1:%.2f y1:%.2f x2:%.2f y2:%.2f Confidence:%f\n", i, box.Label, box.X1, box.Y1, box.X2, box.X2, box.Confidence)
	}
}
```
How to run
```bash
ONNXRUNTIME_LIB_PATH=ONNX_LIBRARY_PATH go run main.go
# example 
# ONNXRUNTIME_LIB_PATH=/usr/local/lib/libonnxruntime.so.1.20.1 go run main.go
```

Output:
```bash
Object 0: bus x1:23.23 y1:229.70 x2:802.13 y2:802.13 Confidence:0.915536
Object 1: person x1:47.94 y1:397.81 x2:242.10 y2:242.10 Confidence:0.893309
Object 2: person x1:670.38 y1:394.92 x2:810.00 y2:810.00 Confidence:0.845367
Object 3: person x1:222.90 y1:406.55 x2:344.93 y2:344.93 Confidence:0.830382
Object 4: person x1:0.26 y1:557.49 x2:60.09 y2:60.09 Confidence:0.458066
```

## 🤝 Contributing
Contributions are welcome! Feel free to fork the repository and submit pull requests or report issues.

## 📝 License
This project is licensed under the MIT License. See the LICENSE file for more details.

