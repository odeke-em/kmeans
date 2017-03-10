package kmeans

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"
)

// EuclideanDistance:
// sqrt((p1 - q1)^2 + (p2 - q2)^2 + (p3 - q3)^2 + ...(pn - qn)^2)
func TransformedEuclideanDistance(t Transformer, p, q Vector) (float64, error) {
	pLen, qLen := p.Len(), q.Len()
	if pLen != qLen {
		return 0, fmt.Errorf("len(p)=%d != len(q)=%d", pLen, qLen)
	}

	var fn func(interface{}) float64
	if t != nil {
		fn = t.Transform
	}

	var sqrdD float64
	for i := 0; i < pLen; i++ {
		pDim, _ := p.Dimension(i)
		qDim, _ := q.Dimension(i)
		pi := quantifyAsFloat64(fn, pDim)
		qi := quantifyAsFloat64(fn, qDim)

		pq := pi - qi
		sqrdD += (pq * pq)
	}

	return math.Sqrt(sqrdD), nil
}

func EuclideanDistance(p, q Vector) (float64, error) {
	return TransformedEuclideanDistance(nil, p, q)
}

type Cluster map[Vector][]Vector

type KMean struct {
	K       int
	Vectors []Vector
	Seed    int64
}

func KMeans(k int, points ...Vector) (Cluster, error) {
	return KMeanify(&KMean{K: k, Vectors: points})
}

func KMeanify(km *KMean) (Cluster, error) {
	k := km.K
	points := km.Vectors
	seed := km.Seed

	if k < 2 {
		return nil, fmt.Errorf("at least 2 centroids are to be picked")
	}

	// 1. Pick k centroids at random as the initial centroids
	if k >= len(points) {
		return nil, fmt.Errorf("k=%d >= len(points)=%d", k, len(points))
	}

	if seed <= 0 {
		seed = time.Now().Unix()
	}
	randSource := rand.New(rand.NewSource(seed))

	centroidIndices := randSource.Perm(len(points))[:k]
	centroidIndicesMap := make(map[int]bool)

	var centroids []Vector
	for _, i := range centroidIndices {
		centroids = append(centroids, points[i])
		centroidIndicesMap[i] = true
	}

	var lastCluster Cluster

	passes := uint64(0)
	for {
		curCluster := make(Cluster)
		// Step 2: Assign each object to the centroid closest to it.
		for i, p := range points {
			if _, isCentroid := centroidIndicesMap[i]; isCentroid {
				curCluster[p] = curCluster[p]
				continue
			}

			indicesToDistances := distances(nil, p, centroids...)
			minDistanceCentroidIndex := minDistanceIndex(indicesToDistances)

			closestCentroid := centroids[minDistanceCentroidIndex]
			curCluster[closestCentroid] = append(curCluster[closestCentroid], p)
		}

		if ClustersEqual(lastCluster, curCluster) { // Centroids are no longer moving
			break
		}

		passes += 1
		lastCluster = curCluster
	}

	return lastCluster, nil
}

func signatureMap(c Cluster) map[interface{}][]Vector {
	signatures := make(map[interface{}][]Vector)
	for k, vec := range c {
		signatures[k.Signature()] = vec
	}
	return signatures
}

func ClustersEqual(cA, cB Cluster) bool {
	if len(cA) != len(cB) {
		return false
	}

	cASignatures := signatureMap(cA)
	cBSignatures := signatureMap(cB)

	for kA, vecA := range cASignatures {
		vecB, inB := cBSignatures[kA]
		if !inB {
			return false
		}

		if !vectorSlicesEqual(vecA, vecB) {
			return false
		}
	}

	return true
}

func vectorSlicesEqual(va, vb []Vector) bool {
	if va == nil || vb == nil {
		return va == nil && vb == nil
	}
	if len(va) != len(vb) {
		return false
	}

	// Even if they aren't sorted, they'll be equal
	hA := make(map[interface{}]struct{})
	for _, vai := range va {
		hA[vai.Signature()] = struct{}{}
	}

	for _, vbi := range vb {
		sig := vbi.Signature()
		_, ok := hA[sig]
		if !ok {
			return false
		}
	}

	return true
}

func minDistanceIndex(indicesToDistances map[int]float64) int {
	minDIndex := 0
	minD := indicesToDistances[minDIndex]
	for index, dist := range indicesToDistances {
		if dist < minD {
			minD = dist
			minDIndex = index
		}
	}

	return minDIndex
}

func distances(t Transformer, subject Vector, others ...Vector) map[int]float64 {
	if len(others) < 1 {
		return nil
	}

	ds := make(map[int]float64)
	for i, other := range others {
		d, _ := TransformedEuclideanDistance(t, subject, other)
		ds[i] = d
	}

	return ds
}

func (c *Cluster) MarshalJSON() ([]byte, error) {
	if c == nil {
		return []byte("{}"), nil
	}

	var strs []string
	for centroid, elements := range *c {
		cBlob, err := json.Marshal(centroid)
		if err != nil {
			return nil, err
		}
		elemsBlob, err := json.Marshal(elements)
		if err != nil {
			return nil, err
		}
		strs = append(strs, fmt.Sprintf("%q: %s", cBlob, elemsBlob))
	}

	buf := new(bytes.Buffer)
	buf.Write([]byte("{"))
	buf.Write([]byte(strings.Join(strs, ",")))
	buf.Write([]byte("}"))

	return buf.Bytes(), nil
}
