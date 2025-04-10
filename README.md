# Fruit Freshness Detection Backend

This repository contains the backend implementation for detecting the freshness of fruits using machine learning techniques. The system analyzes fruit images to determine their freshness level.

## Features

- **Machine Learning Models**: Utilizes trained models to assess fruit freshness.
- **Scripted Analysis**: Contains scripts to process images and predict freshness.
- **Utility Functions**: Provides utility functions to support image processing and model inference.

## Repository Structure

- `.idea/`: Contains project-specific settings and configurations.
- `models/`: Directory where trained machine learning models are stored.
- `scripts/`: Includes scripts for image processing and freshness prediction.
- `utils/`: Contains utility functions to support the main scripts.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `freshness.py`: Main script to execute the freshness detection.
- `main.py`: Entry point for the backend application.
- `download.jpeg`, `fruit_info_qr.png`, `location_qr.png`: Sample images used for testing and demonstration purposes.

## Getting Started

To set up and run the backend:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Kshitiz-002/Fruit-Freshness-Backend.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd Fruit-Freshness-Backend
   ```

3. **Install Required Dependencies**:

   Ensure that you have Python installed. Then, install the necessary packages:

   ```bash
   pip install -r requirements.txt
   ```

   *(Note: If `requirements.txt` is missing, install packages manually by checking the imports used in `main.py` and other scripts.)*

4. **Run the Application**:

   Execute the main script to start the backend service:

   ```bash
   python main.py
   ```

## Usage

Once the backend is running, it can process fruit images to predict their freshness. You can integrate this backend with a frontend application or use API calls to send images and receive predictions.

## Contributing

Contributions are welcome! If you'd like to contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your fork.
5. Submit a pull request to the main repository.
