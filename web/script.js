
const fileInput = document.querySelector('#fileInput');
const dropzone = document.querySelector('#dropzone');
const previewBox = document.querySelector('#previewBox');
const previewImg = document.querySelector('#previewImg');
const filename = document.querySelector('#filename');
const checkBtn = document.querySelector('#checkBtn');
const emptyState = document.querySelector('#emptyState');
const resultState = document.querySelector('#resultState');

const MODEL_PATH = './mobilenet_model/model.json';
const TEST_ACCURACY = '84.11%';
const IMAGE_SIZE = 128;
const GAUGE_LENGTH = 235.62;

let chosenFile = null;
let objectUrl = null;
let model = null;
let isAnalyzing = false;

async function tryLoadModel() {
    const status = document.querySelector('#modelStatus');

    try {
        if (typeof tf === 'undefined') {
            throw new Error(
                'TensorFlow.js did not load. Check your internet connection.'
            );
        }

        status.textContent = 'Loading trained model...';

        model = await tf.loadLayersModel(MODEL_PATH);

        status.textContent = 'Trained model loaded';
        checkBtn.disabled = !chosenFile;

        console.log('MobileNetV2 loaded successfully.');
        console.log('Input shape:', model.inputs[0].shape);
        console.log('Output shape:', model.outputs[0].shape);
    } catch (error) {
        console.error('Model loading failed:', error);
        status.textContent = 'Model unavailable';
        checkBtn.disabled = true;
    }
}

function chooseFile(file) {
    if (!file) return;

    const allowedTypes = [
        'image/jpeg',
        'image/png',
        'image/webp'
    ];

    if (!allowedTypes.includes(file.type)) {
        alert('Please choose a JPG, PNG, or WEBP image.');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        alert('Please choose an image smaller than 10 MB.');
        return;
    }

    chosenFile = file;

    if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
    }

    objectUrl = URL.createObjectURL(file);
    previewImg.src = objectUrl;
    filename.textContent = file.name;

    previewBox.classList.remove('hidden');
    dropzone.classList.add('hidden');

    checkBtn.disabled = !model || isAnalyzing;

    emptyState.classList.remove('hidden');
    resultState.classList.add('hidden');

    resetGauge();

    console.log('Selected image:', file.name);
}

function resetGauge() {
    const gaugeProgress = document.querySelector('#gaugeProgress');
    const gaugeNumber = document.querySelector('#gaugeNumber');
    const gaugeLabel = document.querySelector('#gaugeLabel');
    const gaugePercentage = document.querySelector('#gaugePercentage');

    gaugeProgress.style.strokeDasharray = GAUGE_LENGTH;
    gaugeProgress.style.strokeDashoffset = GAUGE_LENGTH;
    gaugeProgress.style.stroke = '#67e8f9';

    gaugeNumber.textContent = '0%';
    gaugeLabel.textContent = 'PENDING';
    gaugeLabel.className = 'gauge-label';
    gaugePercentage.textContent = '0.00%';
}

function updateGauge(predictedClass, score) {
    const gaugeProgress = document.querySelector('#gaugeProgress');
    const gaugeNumber = document.querySelector('#gaugeNumber');
    const gaugeLabel = document.querySelector('#gaugeLabel');
    const gaugePercentage = document.querySelector('#gaugePercentage');

    const percentage = Math.max(
        0,
        Math.min(100, score * 100)
    );

    gaugeProgress.style.strokeDasharray = GAUGE_LENGTH;

    // Animate the semicircular gauge from empty to the model score.
    gaugeProgress.style.strokeDashoffset =
        GAUGE_LENGTH * (1 - percentage / 100);

    gaugeNumber.textContent = Math.round(percentage) + '%';
    gaugePercentage.textContent = percentage.toFixed(2) + '%';

    gaugeLabel.textContent = predictedClass;
    gaugeLabel.className =
        'gauge-label ' + predictedClass.toLowerCase();

    gaugeProgress.style.stroke =
        predictedClass === 'REAL' ? '#7ce7b1' : '#ff858d';
}

function showError(message) {
    emptyState.classList.add('hidden');
    resultState.classList.remove('hidden');

    document.querySelector('#resultTitle').textContent =
        'Prediction unavailable';

    document.querySelector('#resultText').textContent = message;

    document.querySelector('#predictionModelStatus').textContent =
        'Error';

    resetGauge();
}

fileInput.addEventListener('change', event => {
    chooseFile(event.target.files[0]);
});

document.querySelector('#removeBtn').addEventListener('click', () => {
    chosenFile = null;
    fileInput.value = '';

    if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
        objectUrl = null;
    }

    previewImg.removeAttribute('src');
    previewBox.classList.add('hidden');
    dropzone.classList.remove('hidden');

    checkBtn.disabled = true;

    resultState.classList.add('hidden');
    emptyState.classList.remove('hidden');

    resetGauge();
});

['dragenter', 'dragover'].forEach(type => {
    dropzone.addEventListener(type, event => {
        event.preventDefault();
        dropzone.classList.add('dragging');
    });
});

['dragleave', 'drop'].forEach(type => {
    dropzone.addEventListener(type, event => {
        event.preventDefault();
        dropzone.classList.remove('dragging');
    });
});

dropzone.addEventListener('drop', event => {
    const file = event.dataTransfer.files[0];
    chooseFile(file);
});

// Load demo samples from the web/demo_samples folder.
document.querySelectorAll('[data-sample]').forEach(button => {
    button.addEventListener('click', async () => {
        try {
            const response = await fetch(
                './demo_samples/' + button.dataset.sample
            );

            if (!response.ok) {
                throw new Error(
                    'Sample image not found in web/demo_samples.'
                );
            }

            const blob = await response.blob();
            const sampleName = button.dataset.sample;

            const extension = sampleName.split('.').pop().toLowerCase();

            const mimeTypes = {
                jpg: 'image/jpeg',
                jpeg: 'image/jpeg',
                png: 'image/png',
                webp: 'image/webp'
            };

            const sampleFile = new File(
                [blob],
                sampleName,
                {
                    type: mimeTypes[extension] || blob.type
                }
            );

            chooseFile(sampleFile);
        } catch (error) {
            console.error('Sample loading failed:', error);

            alert(
                'Could not load this sample. Check its filename in ' +
                'web/demo_samples, or upload your own image.'
            );
        }
    });
});

checkBtn.addEventListener('click', async () => {
    if (!chosenFile || !model || isAnalyzing) return;

    isAnalyzing = true;
    checkBtn.disabled = true;

    const title = document.querySelector('#resultTitle');
    const text = document.querySelector('#resultText');
    const modelStatus = document.querySelector('#predictionModelStatus');

    emptyState.classList.add('hidden');
    resultState.classList.remove('hidden');

    title.textContent = 'Analyzing image...';
    text.textContent = 'Running the trained MobileNetV2 model.';
    modelStatus.textContent = 'Analyzing';

    let inputTensor = null;
    let predictionTensor = null;

    try {
        await previewImg.decode();

        // Match the model's training preprocessing:
        // resize to 128 × 128 and scale RGB values to [-1, 1].
        inputTensor = tf.tidy(() => {
            return tf.browser
                .fromPixels(previewImg)
                .resizeBilinear([IMAGE_SIZE, IMAGE_SIZE])
                .toFloat()
                .div(127.5)
                .sub(1)
                .expandDims(0);
        });

        predictionTensor = model.predict(inputTensor);

        const values = await predictionTensor.data();
        const score = values[0];

        if (!Number.isFinite(score) || score < 0 || score > 1) {
            throw new Error(
                'The model returned an invalid score. Please check the model.'
            );
        }

        // The trained model's class order is:
        // fake = 0, real = 1.
        const predictedClass = score >= 0.5 ? 'REAL' : 'FAKE';

        // Show the model score associated with the predicted class.
        const classScore =
            predictedClass === 'REAL' ? score : 1 - score;

        updateGauge(predictedClass, classScore);

        title.textContent = 'Model prediction: ' + predictedClass;

        text.textContent =
            'The model classified this image as ' + predictedClass +
            ', with a model score of ' + (classScore * 100).toFixed(2) +
            '%. This score is not a verified probability of authenticity. ' +
            'The model may misclassify currency notes or unrelated images.';

        modelStatus.textContent = 'Loaded';
        console.log('Raw model output:', score);
        console.log('Predicted class:', predictedClass);
        console.log('Class score:', classScore);
        console.log('Test dataset accuracy:', TEST_ACCURACY);
    } catch (error) {
        console.error('Prediction failed:', error);
        showError(error.message || 'An unexpected error occurred.');
    } finally {
        if (predictionTensor) {
            predictionTensor.dispose();
        }

        if (inputTensor) {
            inputTensor.dispose();
        }

        isAnalyzing = false;
        checkBtn.disabled = !chosenFile || !model;
    }
});

tryLoadModel();
