package utils

import (
	"testing"
)

func TestBoundingBoxString(t *testing.T) {
	tests := []struct {
		name     string
		bbox     BoundingBox
		expected string
	}{
		{
			name: "Basic case",
			bbox: BoundingBox{
				Label:      "Person",
				Confidence: 0.95,
				X1:         0.0,
				Y1:         0.0,
				X2:         100.0,
				Y2:         200.0,
			},
			expected: "Object Person (confidence 0.950000): (0.000000, 0.000000), (100.000000, 200.000000)",
		},
		{
			name: "Empty label",
			bbox: BoundingBox{
				Label:      "",
				Confidence: 0.80,
				X1:         -10.5,
				Y1:         5.5,
				X2:         50.0,
				Y2:         75.0,
			},
			expected: "Object  (confidence 0.800000): (-10.500000, 5.500000), (50.000000, 75.000000)",
		},
		{
			name: "Zero confidence",
			bbox: BoundingBox{
				Label:      "Car",
				Confidence: 0.0,
				X1:         1.1,
				Y1:         2.2,
				X2:         3.3,
				Y2:         4.4,
			},
			expected: "Object Car (confidence 0.000000): (1.100000, 2.200000), (3.300000, 4.400000)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := tt.bbox.String()
			if result != tt.expected {
				t.Errorf("expected %q, got %q", tt.expected, result)
			}
		})
	}
}
