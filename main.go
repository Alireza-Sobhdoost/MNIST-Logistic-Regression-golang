package main

import (
    "io/ioutil"
    "path/filepath"
    "time"
	"strconv"
	"github.com/go-gota/gota/dataframe"
    "fmt"
    "math"
    "math/rand"
    "gonum.org/v1/gonum/mat"
    "os"
    "image"
    "image/color"
    "github.com/nfnt/resize"
    "github.com/gonum/matrix/mat64"
	"encoding/json"
	"encoding/csv"
	_ "image/jpeg" // Register JPEG support
)

type LogisticRegression struct {
    learningRate float64
    epochs       int
    batchSize    int
    beta1        float64
    beta2        float64
    epsilon      float64
    weights      *mat.VecDense
    bias         float64
    m            *mat.VecDense
    v            *mat.VecDense
    mb           float64
    vb           float64
}

func sigmoid(z float64) float64 {
    return 1.0 / (1.0 + math.Exp(-z))
}

func computeLoss(predictions, y *mat.VecDense) float64 {
    m := float64(predictions.Len())
    loss := 0.0
    for i := 0; i < predictions.Len(); i++ {
        p := predictions.AtVec(i)
        label := y.AtVec(i)
        loss += label*math.Log(p) + (1-label)*math.Log(1-p)
    }
    return -loss / m
}


func calculateMetrics(yTrue, yPred *mat.VecDense) (precision, recall, f1 float64) {
    tp, fp, fn := 0.0, 0.0, 0.0
    for i := 0; i < yTrue.Len(); i++ {
        if yPred.AtVec(i) == 1 {
            if yTrue.AtVec(i) == 1 {
                tp++
            } else {
                fp++
            }
        } else if yTrue.AtVec(i) == 1 {
            fn++
        }
    }
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    return
}


// Predict makes predictions using the trained logistic regression model
func (lr *LogisticRegression) predict(X *mat.Dense) *mat.VecDense {
    numSamples, _ := X.Dims()
    predictions := mat.NewVecDense(numSamples, nil)

    for i := 0; i < numSamples; i++ {
        sum := lr.bias
        for j := 0; j < X.RawMatrix().Cols; j++ {
            sum += X.At(i, j) * lr.weights.AtVec(j)
        }
        predictions.SetVec(i, sigmoid(sum))
    }

    // Apply a threshold of 0.5 to classify predictions into binary labels (0 or 1)
    for i := 0; i < predictions.Len(); i++ {
        if predictions.AtVec(i) >= 0.5 {
            predictions.SetVec(i, 1)
        } else {
            predictions.SetVec(i, 0)
        }
    }

    return predictions
}

func (lr *LogisticRegression) fit(X *mat.Dense, y *mat.VecDense) {
    numSamples, numFeatures := X.Dims()
    lr.weights = mat.NewVecDense(numFeatures, nil)
    lr.m = mat.NewVecDense(numFeatures, nil)
    lr.v = mat.NewVecDense(numFeatures, nil)
    lr.mb = 0.0
    lr.vb = 0.0
    lr.bias = 0.0
    t := 0
	counter := 1500

    for epoch := 0; epoch < lr.epochs; epoch++ {
        perm := rand.Perm(numSamples)
        for batchStart := 0; batchStart < numSamples; batchStart += lr.batchSize {
            t++
            batchEnd := batchStart + lr.batchSize
            if batchEnd > numSamples {
                batchEnd = numSamples
            }
            batchSize := batchEnd - batchStart

            db := 0.0
            dw := mat.NewVecDense(numFeatures, nil)
            predictions := mat.NewVecDense(batchSize, nil)

            for i := batchStart; i < batchEnd; i++ {
                idx := perm[i]
                sum := lr.bias
                for j := 0; j < numFeatures; j++ {
                    sum += X.At(idx, j) * lr.weights.AtVec(j)
                }
                pred := sigmoid(sum)
                predictions.SetVec(i-batchStart, pred)
                error := pred - y.AtVec(idx)
                db += error
                for j := 0; j < numFeatures; j++ {
                    dw.SetVec(j, dw.AtVec(j)+error*X.At(idx, j))
                }
            }

            loss := computeLoss(predictions, y)

			if counter % 1500 == 0 {
            	fmt.Printf("Epoch %d, Batch %d - Loss: %f\n", epoch+1, t, loss)
			}
			counter++

            for j := 0; j < numFeatures; j++ {
                g := dw.AtVec(j) / float64(batchSize)
                lr.m.SetVec(j, lr.beta1*lr.m.AtVec(j)+(1-lr.beta1)*g)
                lr.v.SetVec(j, lr.beta2*lr.v.AtVec(j)+(1-lr.beta2)*g*g)
                mHat := lr.m.AtVec(j) / (1 - math.Pow(lr.beta1, float64(t)))
                vHat := lr.v.AtVec(j) / (1 - math.Pow(lr.beta2, float64(t)))
                lr.weights.SetVec(j, lr.weights.AtVec(j)-lr.learningRate*mHat/(math.Sqrt(vHat)+lr.epsilon))
            }
            g := db / float64(batchSize)
            lr.mb = lr.beta1*lr.mb + (1-lr.beta1)*g
            lr.vb = lr.beta2*lr.vb + (1-lr.beta2)*g*g
            mbHat := lr.mb / (1 - math.Pow(lr.beta1, float64(t)))
            vbHat := lr.vb / (1 - math.Pow(lr.beta2, float64(t)))
            lr.bias -= lr.learningRate * mbHat / (math.Sqrt(vbHat) + lr.epsilon)
        }
    }
}

func loadDataFromDir(directory string, label float64) ([]string, []float64) {
	files, err := ioutil.ReadDir(directory)
	if err != nil {
		fmt.Println("Error reading directory:", err)
		return nil, nil
	}

	var filePaths []string
	var labels []float64
	for _, file := range files {
		if !file.IsDir() {
			filePaths = append(filePaths, filepath.Join(directory, file.Name()))
			labels = append(labels, label)
		}
	}
	return filePaths, labels
}

func splitData(filePaths []string, labels []float64, trainRatio float64) (trainPaths []string, testPaths []string, trainLabels []float64, testLabels []float64) {
	// Shuffle data
	rand.Seed(time.Now().UnixNano())
	shuffledIndices := rand.Perm(len(filePaths))

	// Split data into training and test sets
	trainSize := int(float64(len(filePaths)) * trainRatio)

	for i, idx := range shuffledIndices {
		if i < trainSize {
			trainPaths = append(trainPaths, filePaths[idx])
			trainLabels = append(trainLabels, labels[idx])
		} else {
			testPaths = append(testPaths, filePaths[idx])
			testLabels = append(testLabels, labels[idx])
		}
	}
	return
}

// Function to load and resize an image
func loadImage(filepath string, width, height uint) (image.Image, error) {
	// Open image file
	file, err := os.Open(filepath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Decode the image
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}

	// Resize image if needed
	img = resize.Resize(width, height, img, resize.Lanczos3)

	return img, nil
}

// Function to convert a grayscale image to a Gonum matrix
func imageToMatrix(img image.Image) *mat64.Dense {
	bounds := img.Bounds()
	width, height := bounds.Max.X, bounds.Max.Y

	// Create a slice to hold the pixel values
	data := make([]float64, width*height) // Single channel for grayscale
	index := 0

	// Loop through all pixels and extract the grayscale values
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			gray := img.At(x, y).(color.Gray) // Extract grayscale value
			// Normalize the grayscale values to be in the range 0-1
			data[index] = float64(gray.Y) / 255.0
			index++
		}
	}

	// Convert slice to a Gonum matrix (Dense format)
	matrix := mat64.NewDense(height*width, 1, data) // height*width pixels, 1 channel (grayscale)

	return matrix
}

// Function to process a list of image paths
func processImagePaths(imagePaths []string, width, height uint) ([]*mat64.Dense, error) {
	var matrices []*mat64.Dense

	for _, path := range imagePaths {
		// Load and process each image
		img, err := loadImage(path, width, height)
		if err != nil {
			return nil, fmt.Errorf("failed to load image %s: %v", path, err)
		}

		// Convert the image to a Gonum matrix
		matrix := imageToMatrix(img)

		// Append the matrix to the result slice
		matrices = append(matrices, matrix)
	}

	return matrices, nil
}

func (lr *LogisticRegression) saveModel(filename string) error {
	// Create a struct to store the model parameters
	modelParams := struct {
		Weights   []float64 `json:"weights"`
		Bias      float64   `json:"bias"`
		M         []float64 `json:"m"`
		V         []float64 `json:"v"`
		Mb        float64   `json:"mb"`
		Vb        float64   `json:"vb"`
		LearningRate float64 `json:"learningRate"`
		Epochs      int      `json:"epochs"`
		BatchSize   int      `json:"batchSize"`
		Beta1       float64  `json:"beta1"`
		Beta2       float64  `json:"beta2"`
		Epsilon     float64  `json:"epsilon"`
	}{
		Weights:   lr.weights.RawVector().Data,
		Bias:      lr.bias,
		M:         lr.m.RawVector().Data,
		V:         lr.v.RawVector().Data,
		Mb:        lr.mb,
		Vb:        lr.vb,
		LearningRate: lr.learningRate,
		Epochs:      lr.epochs,
		BatchSize:   lr.batchSize,
		Beta1:       lr.beta1,
		Beta2:       lr.beta2,
		Epsilon:     lr.epsilon,
	}

	// Open file for writing
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("could not create file: %v", err)
	}
	defer file.Close()

	// Encode the model parameters into JSON and write to file
	encoder := json.NewEncoder(file)
	err = encoder.Encode(modelParams)
	if err != nil {
		return fmt.Errorf("could not write model to file: %v", err)
	}

	return nil
}


func (lr *LogisticRegression) loadModel(filename string) error {
	// Open the file for reading
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("could not open file: %v", err)
	}
	defer file.Close()

	// Create a struct to store the model parameters
	modelParams := struct {
		Weights   []float64 `json:"weights"`
		Bias      float64   `json:"bias"`
		M         []float64 `json:"m"`
		V         []float64 `json:"v"`
		Mb        float64   `json:"mb"`
		Vb        float64   `json:"vb"`
		LearningRate float64 `json:"learningRate"`
		Epochs      int      `json:"epochs"`
		BatchSize   int      `json:"batchSize"`
		Beta1       float64  `json:"beta1"`
		Beta2       float64  `json:"beta2"`
		Epsilon     float64  `json:"epsilon"`
	}{}

	// Decode the JSON from the file into the struct
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&modelParams)
	if err != nil {
		return fmt.Errorf("could not read model from file: %v", err)
	}

	// Set the parameters in the logistic regression model
	lr.weights = mat.NewVecDense(len(modelParams.Weights), modelParams.Weights)
	lr.bias = modelParams.Bias
	lr.m = mat.NewVecDense(len(modelParams.M), modelParams.M)
	lr.v = mat.NewVecDense(len(modelParams.V), modelParams.V)
	lr.mb = modelParams.Mb
	lr.vb = modelParams.Vb
	lr.learningRate = modelParams.LearningRate
	lr.epochs = modelParams.Epochs
	lr.batchSize = modelParams.BatchSize
	lr.beta1 = modelParams.Beta1
	lr.beta2 = modelParams.Beta2
	lr.epsilon = modelParams.Epsilon

	return nil
}

func save_to_csv(path string, data []string, label []float64){

	file, err := os.Create(path)
	if err != nil {
		fmt.Println("Error creating CSV:", err)
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	writer.Write([]string{"path", "label"})

	// Write data rows
	for i := range data {
		writer.Write([]string{data[i], fmt.Sprintf("%.0f", label[i])})
	}

}

func readCSV(filename string) ([]string, []float64, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	df := dataframe.ReadCSV(file)
	if df.Err != nil {
		return nil, nil, df.Err
	}

	paths := df.Col("path").Records()
	labels := make([]float64, len(paths))
	labelStrings := df.Col("label").Records()
	
	for i, label := range labelStrings {
		val, err := strconv.ParseFloat(label, 64)
		if err != nil {
			return nil, nil, err
		}
		labels[i] = val
	}

	return paths, labels, nil
}

func merge_slices(matrices []*mat64.Dense) *mat.Dense{
    rows := len(matrices)
    cols := matrices[0].RawMatrix().Rows * matrices[0].RawMatrix().Cols
    X := mat.NewDense(rows, cols, nil)
    for i, matrix := range matrices {
        data := matrix.RawMatrix().Data
        for j, val := range data {
            X.Set(i, j, val)
        }
    }

	return X
}

func main() {

	// Set random seed
    rand.Seed(time.Now().UnixNano())

	// Prepare data directories
	
	class0Dir := "./data/0"
	class1Dir := "./data/1"

	// Load data
	class0Paths, class0Labels := loadDataFromDir(class0Dir, 0.0)
	class1Paths, class1Labels := loadDataFromDir(class1Dir, 1.0)

	// Combine data
	allPaths := append(class0Paths, class1Paths...)
	allLabels := append(class0Labels, class1Labels...)

	// Split data into train and test
	trainPaths, testPaths, Y_train, Y_test := splitData(allPaths, allLabels, 0.7)

	// Write data to CSV

	save_to_csv("train_path.csv", trainPaths, Y_train)
	save_to_csv("test_path.csv", testPaths, Y_test)


	// Load images

	width, height := uint(28), uint(28) // Standard MNIST dimensions (28x28)
    train_matrices, _ := processImagePaths(trainPaths, width, height)
    testMatrices, _ := processImagePaths(testPaths, width, height)

    // Convert slice of matrices to a single matrix
	X_train := merge_slices(train_matrices)
    X_test := merge_slices(testMatrices)


	// init the model

    model := LogisticRegression{
        learningRate: 0.001,
        epochs:       1000,
        batchSize:    2,
        beta1:        0.9,
        beta2:        0.999,
        epsilon:      1e-8,
    }

    yTrainVec := mat.NewVecDense(len(Y_train), Y_train)

	// train the model

    model.fit(X_train, yTrainVec)

	// Get predictions on test data
	yPred := model.predict(X_test)



	// Calculate metrics (precision, recall, f1)
	precision, recall, f1 := calculateMetrics(mat.NewVecDense(len(Y_test), Y_test), yPred)
		
	fmt.Printf("Precision: %.4f, Recall: %.4f, F1 Score: %.4f\n", precision, recall, f1)
	
	// Save the model
	err := model.saveModel("logistic_model.json")
    if err != nil {
        fmt.Println("Error saving model:", err)
        return
    }
    fmt.Println("Model saved successfully!")

	// Check the results manually
	fmt.Println("Prediction\tActual")
	for i := 0; i < 20; i++ {
		fmt.Printf("%.2f\t\t%.2f\n", yPred.AtVec(i), Y_test[i])
	}

	// ##Test the loaded model###

	// testPaths, testLabels, err := readCSV("test_path.csv")
	// if err != nil {
	// 	fmt.Println("Error reading CSV:", err)
	// 	return
	// }

	// // Load the images

	// width, height := uint(28), uint(28) // Standard MNIST dimensions (28x28)
	// test_matrices, _ := processImagePaths(testPaths, width, height)
	// // Convert matrices to the format needed for prediction
	// X_test_loaded := merge_slices(test_matrices)

	// // Load the model

	// var loadedModel LogisticRegression
	// err = loadedModel.loadModel("./logistic_model.json")
	// if err != nil {
	// 	fmt.Println("Error loading model:", err)
	// 	return
	// }
	// fmt.Println("Model loaded successfully!")

	// // Make predictions using the loaded model
	// yPred_loaded := loadedModel.predict(X_test_loaded)
	// // Calculate metrics (precision, recall, f1)
	// precision_loaded, recall_loaded, f1_loaded := calculateMetrics(mat.NewVecDense(len(testLabels), testLabels), yPred_loaded)
	// fmt.Printf("Loaded Model - Precision: %.4f, Recall: %.4f, F1 Score: %.4f\n", precision_loaded, recall_loaded, f1_loaded)


}



