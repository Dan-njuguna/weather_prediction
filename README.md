# 🌦️ HaliYaAnga - Weather Predictions

HaliYaAnga is a project that demonstrates the integration of machine learning pipelines with CI/CD processes using Continuous Machine Learning (CML) in GitHub Actions workflows. This repository provides a practical example of how to automate the training, evaluation, and deployment of weather prediction models in a collaborative development environment.

## ✨ Key Features

- 🚀 **Automated ML Workflows**: Leverage CML to streamline the entire machine learning lifecycle, from data preprocessing to model deployment.
- 🌤️ **Weather Prediction Models**: Build and evaluate models for accurate weather forecasting.
- 🔄 **CI/CD Integration**: Seamlessly integrate machine learning pipelines into GitHub Actions for continuous delivery and deployment.
- 🤝 **Collaborative Development**: Enable teams to work together efficiently on machine learning projects.

## 🛠️ Getting Started

To get started with HaliYaAnga, follow these steps:

### Prerequisites

1. **Initialize DVC**: Run the following command to initialize a DVC repository:
    ```bash
    dvc init
    ```
    This sets up DVC in your project directory.

2. **Add Stages for Data Versioning**:
    - Add a preprocessing stage:
      ```bash
      dvc stage add -n preprocess -d preprocess_dataset.py -d raw_dataset/weather.csv -d utils_and_constants.py -o processed_dataset/weather.csv python3 preprocess_dataset.py
      ```
    - Add a training stage:
      ```bash
      dvc stage add -n train -d metrics_and_plots.py -d model.py -d processed_dataset/weather.csv -d train.py -d utils_and_constants.py -o metrics.json -o confusion_matrix.png python3 train.py
      ```

3. **Reproduce Pipelines**: Use the following command to execute the pipeline and ensure all stages are run:
    ```bash
    dvc repro
    ```

4. **Track Changes with Git**: After adding stages, track the changes by running:
    ```bash
    git add dvc.yaml
    ```

### Explore the Project

Refer to the repository documentation and examples provided to set up your environment, run the workflows, and explore the capabilities of the project.

---

🌍 *HaliYaAnga* means "weather forecast" in Swahili, reflecting the project's focus on weather prediction.