
# Diabetic Retinopathy Classification using CNN

This project implements a Convolutional Neural Network (CNN) model for the classification of diabetic retinopathy using the InceptionV3 architecture. The model is trained on a dataset of Gaussian filtered images, achieving a validation accuracy of approximately 84%.

## Dataset

The dataset used for this project is the **Diabetic Retinopathy - 224x224 Gaussian Filtered** dataset, available on [Kaggle](https://www.kaggle.com/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered). The dataset contains preprocessed images of retinas, categorized into five classes:

1. Mild
2. Moderate
3. No_DR (No Diabetic Retinopathy)
4. Proliferative_DR
5. Severe

## Project Structure

- `diabetic_retinopathy.ipynb`: Jupyter notebook containing the code for data preprocessing, model training, and evaluation.
- `README.md`: This file, providing an overview of the project.
- `requirements.txt`: List of dependencies required to run the project.

## Model Architecture

The model uses the **InceptionV3** architecture, pre-trained on ImageNet, with custom top layers added for classification. The architecture is as follows:

1. **InceptionV3 Base Model**: Pre-trained on ImageNet, excluding the top layers.
2. **GlobalAveragePooling2D Layer**: Reduces the dimensions of the feature maps.
3. **Dense Layer**: Fully connected layer with 1024 units and ReLU activation.
4. **Output Layer**: Fully connected layer with 5 units and softmax activation for multi-class classification.

## Training

1. **Data Augmentation**: The training data is augmented using various transformations like rotation, width/height shifts, shearing, zooming, and horizontal flipping.
2. **Initial Training**: The base model layers are frozen, and only the custom layers are trained for 30 epochs.
3. **Fine-Tuning**: After initial training, all layers are unfrozen and the entire model is fine-tuned with a lower learning rate for an additional 30 epochs.

## Results

The model achieved the following results on the validation set:

- **Overall Validation Accuracy**: ~84%
- **Classification Report**:

| Class            | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| Mild             | 0.13      | 0.12   | 0.12     | 74      |
| Moderate         | 0.29      | 0.36   | 0.32     | 199     |
| No_DR            | 0.50      | 0.50   | 0.50     | 361     |
| Proliferative_DR | 0.08      | 0.05   | 0.06     | 59      |
| Severe           | 0.00      | 0.00   | 0.00     | 38      |

- **Weighted Average**: Precision: 0.34, Recall: 0.36, F1-Score: 0.35

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/diabetic-retinopathy-cnn.git
    cd diabetic-retinopathy-cnn
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Jupyter notebook to train and evaluate the model:
    ```bash
    jupyter notebook diabetic_retinopathy.ipynb
    ```

## Acknowledgments

- The dataset was provided by [Sovit Rath](https://www.kaggle.com/sovitrath) on Kaggle.
- The InceptionV3 model used in this project was pre-trained on ImageNet.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
