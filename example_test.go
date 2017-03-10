package kmeans_test

import (
	"fmt"
	"log"

	"github.com/odeke-em/kmeans"
)

func ExampleEuclideanDistance() {
	ptA, _ := kmeans.NewCoordinate(10.2, 15.6, 25)
	ptB, _ := kmeans.NewCoordinate(23.7, -8.9, 99.4)

	dist, err := kmeans.EuclideanDistance(ptA, ptB)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Euclidean distance: %.3f\n", dist)

	// Output:
	// Euclidean distance: 79.485
}

type person struct {
	Age float32 `json:"age"`

	NumLanguages int `json:"nlang"`

	WorkExperience float32 `json:"wexp"`
}

var _ kmeans.Vector = (*person)(nil)

func (p *person) Signature() interface{} {
	return fmt.Sprintf("%f-%d-%f", p.Age, p.NumLanguages, p.WorkExperience)
}

func (p *person) Len() int { return 3 }
func (p *person) Dimension(i int) (interface{}, error) {
	switch i {
	case 0:
		return p.Age, nil
	case 1:
		return p.NumLanguages, nil
	case 2:
		return p.WorkExperience, nil
	default:
		return 0, fmt.Errorf("unhandled dimension")
	}
}

func ExampleCustomTransformer() {
	lebron := &person{
		Age:            32,
		NumLanguages:   1,
		WorkExperience: 14,
	}

	kobe := &person{
		Age:            38,
		WorkExperience: 18,
		NumLanguages:   3,
	}

	dist, err := kmeans.EuclideanDistance(kobe, lebron)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Euclidean distance: %.3f\n", dist)

	// Output:
	// Euclidean distance: 7.483
}
