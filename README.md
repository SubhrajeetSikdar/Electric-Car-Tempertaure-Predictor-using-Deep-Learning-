# Electric Motor Temperature Prediction

This repository contains code for predicting the temperature of an electric motor using deep learning techniques. The code utilizes PyTorch for building and training the neural network model.

## Dataset

The dataset used for training the model is the "Electric Motor Temperature Data" available on Kaggle. The dataset contains various features related to the temperature of an electric motor, such as ambient temperature, coolant temperature, motor speed, torque, etc.

## Model Architecture

The neural network model used for prediction consists of a combination of convolutional and linear layers. The input to the model is a sequence of features, and the output is the predicted rotor temperature.

## Training

The model is trained using the Adam optimizer and Mean Squared Error (MSE) loss function. The training process involves iterating over the dataset for multiple epochs and updating the model parameters to minimize the loss.

## Evaluation

The trained model is evaluated using the R^2 score, which measures the goodness of fit of the predicted values to the actual values. Additionally, the predicted versus actual temperature plot is also visualized for further analysis.

## Usage

1. Clone this repository to your local machine.
2. Install the required dependencies using the following command:
    ```
    pip install pandas numpy matplotlib seaborn torch scikit-learn
    ```
3. Run the main script to train and evaluate the model:
    ```
    python main.py
    ```
4. Check the output for the R^2 score and the predicted versus actual temperature plot.

## Results

The trained model achieves an R^2 score of approximately 0.986, indicating a high level of accuracy in predicting the electric motor temperature.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: [Electric Motor Temperature Data](https://www.kaggle.com/wkirgsn/electric-motor-temperature)
- PyTorch: [PyTorch](https://pytorch.org/)
- Scikit-learn: [Scikit-learn](https://scikit-learn.org/)
