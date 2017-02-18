package kmeans_test

import (
	"encoding/json"
	"math/rand"
	"testing"

	"github.com/odeke-em/kmeans"
)

func coord(args ...interface{}) *kmeans.Coordinate {
	c, _ := kmeans.NewCoordinate(args...)
	return c
}

func coordsAsVectors(coords ...*kmeans.Coordinate) []kmeans.Vector {
	var points []kmeans.Vector
	for _, c := range coords {
		points = append(points, c)
	}

	return points
}

func personsAsVectors(players ...*person) []kmeans.Vector {
	var points []kmeans.Vector
	for _, p := range players {
		points = append(points, p)
	}

	return points
}

func TestKMeans(t *testing.T) {
	rand.Seed(10)

	tests := []struct {
		k      int
		seed   int64
		points []kmeans.Vector
		want   kmeans.Cluster
	}{

		0: {
			seed: 10,
			k:    4,
			points: coordsAsVectors(
				coord(23, 10, 15),
				coord(255, 169, 200),
				coord(230, 150, 215),
				coord(156, 255, 215),
				coord(123, 10, 15),
				coord(77, 0, 47),
				coord(95, 0, 15),
				coord(89, 120, 15),
			),
			want: kmeans.Cluster{
				coord(255, 169, 200): nil,
				coord(230, 150, 215): []kmeans.Vector{
					coord(156, 255, 215),
				},
				coord(123, 10, 15): []kmeans.Vector{
					coord(77, 0, 47),
					coord(95, 0, 15),
					coord(89, 120, 15),
				},
				coord(23, 10, 15): nil,
			},
		},
		1: {
			k:    3,
			seed: 648,
			points: personsAsVectors(
				&person{Age: 32, NumLanguages: 1, WorkExperience: 14},
				&person{Age: 38, WorkExperience: 18, NumLanguages: 3},
				&person{Age: 10, WorkExperience: 0, NumLanguages: 5},
				&person{Age: 16, WorkExperience: 2, NumLanguages: 1},
				&person{Age: 65, WorkExperience: 45, NumLanguages: 2},
				&person{Age: 25, WorkExperience: 2, NumLanguages: 1},
				&person{Age: 63, WorkExperience: 50, NumLanguages: 6},
				&person{Age: 23, WorkExperience: 0, NumLanguages: 1},
			),
			want: kmeans.Cluster{
				&person{Age: 38, NumLanguages: 3, WorkExperience: 18}: []kmeans.Vector{
					&person{Age: 32, NumLanguages: 1, WorkExperience: 14},
					&person{Age: 65, NumLanguages: 2, WorkExperience: 45},
					&person{Age: 63, NumLanguages: 6, WorkExperience: 50},
				},
				&person{Age: 10, NumLanguages: 5, WorkExperience: 0}: nil,
				&person{Age: 25, NumLanguages: 1, WorkExperience: 2}: []kmeans.Vector{
					&person{Age: 23, NumLanguages: 1, WorkExperience: 0},
					&person{Age: 16, NumLanguages: 1, WorkExperience: 2},
				},
			},
		},
	}

	for i, tt := range tests {
		got, err := kmeans.KMeanify(&kmeans.KMean{K: tt.k, Vectors: tt.points, Seed: tt.seed})
		if err != nil {
			t.Fatalf("#%d: kmeans err: %v", i, err)
		}

		if kmeans.ClustersEqual(got, tt.want) {
			// Pass
			continue
		}

		gotBlob, _ := json.Marshal(&got)
		wantBlob, _ := json.Marshal(&tt.want)
		t.Errorf("#%d:\n\tgot:  %s\n\twant: %s\n", i, gotBlob, wantBlob)
	}
}
