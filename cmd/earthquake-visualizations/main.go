package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/ioutil"
	"log"

	"github.com/odeke-em/kmeans"
	"github.com/odeke-em/usgs"
)

var errUnimplemented = errors.New("unimplemented")

type feature struct {
	usgs.Feature
	memoizedSignature string
}

var _ kmeans.Vector = (*feature)(nil)

func (f *feature) Signature() interface{} {
	if f.memoizedSignature != "" {
		return f.memoizedSignature
	}
	f.memoizedSignature = fmt.Sprintf("%f-%f-%f",
		f.Geometry.Coordinates.Latitude,
		f.Geometry.Coordinates.Longitude,
		f.Geometry.Coordinates.Depth,
	)
	return f.memoizedSignature
}

func (f *feature) Len() int { return 3 }
func (f *feature) Dimension(i int) (interface{}, error) {
	switch i {
	case 0:
		return f.Geometry.Coordinates.Latitude, nil
	case 1:
		return f.Geometry.Coordinates.Longitude, nil
	case 2:
		return f.Geometry.Coordinates.Depth, nil
	default:
		return nil, errUnimplemented
	}
}

// Transformer for a feature --> Geometry.Coordinates

func earthquakeFeaturesToKmeans(features []*usgs.Feature) []kmeans.Vector {
	var vectors []kmeans.Vector
	for _, feat := range features {
		f := feature{*feat, ""}
		vectors = append(vectors, &f)
	}

	return vectors
}

func main() {
	var srcPath string
	flag.StringVar(&srcPath, "src", "", "the path to load the earthquate data from")
	flag.Parse()

	blob, err := ioutil.ReadFile(srcPath)
	if err != nil {
		log.Fatalf("failed to read source path: %v", err)
	}

	usgsResp := new(usgs.Response)
	if err := json.Unmarshal(blob, usgsResp); err != nil {
		log.Fatalf("failed to unmarshal usgs response: %v", err)
	}

	// Goal is to group Feature items by magnitude?
	// Transform magnitues to vectors
	earthquakeVectors := earthquakeFeaturesToKmeans(usgsResp.Features)

	cluster, err := kmeans.KMeans(10, earthquakeVectors...)
	if err != nil {
		log.Fatalf("clustering err: %v", err)
	}

	for key, points := range cluster {
		keyFeature := key.(*feature)
		fmt.Printf("\nStart of cluster:key: %v\n", keyFeature.Geometry.Coordinates)
		for i, pt := range points {
			feat := pt.(*feature)
			fmt.Printf("\t#%d: %#v\n", i, feat.Geometry.Coordinates)
		}
		fmt.Printf("\n\n")
	}
}
