# Driver_Drowsiness_streamlit
A real-time driver drowsiness detection system using computer vision and machine learning. The system monitors driver's eye state and provides progressive alerts to prevent accidents due to drowsiness.


## Features

- Real-time eye state monitoring
- Three-level alert system:
  - Level 1: Plays favorite song when initial drowsiness detected
  - Level 2: Plays alarm when continued drowsiness detected
  - Level 3: Emergency mode (triggered by 'E' key)
- Customizable alert thresholds
- Intelligent blink detection
- Upload and play custom alert songs
- Real-time statistics and monitoring

## Prerequisites

- Python 3.8 or higher
- Webcam
- Audio output device
- Required Python packages (listed in requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd Driver-Drowsiness-Detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure you have the following files in your project directory:
- app.py (main application)
- eye_state_model1.h5 (pre-trained model)
- mixkit-facility-alarm-sound-999.wav (alarm sound)
- saved_songs/ (directory for uploaded songs)

## Running the Application

1. Start the application:
```bash
streamlit run app.py
```

2. Access the application in your web browser at http://localhost:8501

## Usage Instructions

1. **Initial Setup**:
   - Adjust alert thresholds in the sidebar
   - Upload your favorite song (MP3/WAV format)
   - Set alarm and song volumes

2. **Starting Detection**:
   - Click "Start Detection" to begin monitoring
   - Position yourself in front of the camera
   - Ensure proper lighting for accurate detection

3. **Alert Levels**:
   - Level 1 (Song Alert): Plays when initial drowsiness is detected
   - Level 2 (Alarm): Activates with continued drowsiness
   - Level 3 (Emergency): Press 'E' to simulate emergency

4. **Customization**:
   - Adjust drowsiness threshold
   - Modify song alert threshold
   - Fine-tune blink detection sensitivity
   - Control audio volumes

## Deployment Options

### 1. Local Deployment
- Follow the installation instructions above
- Suitable for personal use or testing

### 2. Cloud Deployment (Streamlit Cloud)
1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy the application

### 3. Docker Deployment
```dockerfile
FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

To build and run:
```bash
docker build -t drowsiness-detection .
docker run -p 8501:8501 drowsiness-detection
```

## Security Considerations

1. Camera Access:
   - Ensure proper permissions for webcam access
   - Handle camera access requests appropriately

2. Audio Files:
   - Implement file type validation for uploaded songs
   - Limit file sizes for uploads

3. Data Privacy:
   - No video/image data is stored
   - Process all detection locally

## Troubleshooting

1. Camera Issues:
   - Ensure webcam is properly connected
   - Check camera permissions
   - Try restarting the application

2. Audio Issues:
   - Verify audio output device is working
   - Check volume settings
   - Ensure audio files are in correct format

3. Performance Issues:
   - Close other resource-intensive applications
   - Check system requirements
   - Adjust detection thresholds

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

[Your chosen license]

## Acknowledgments

- OpenCV for computer vision capabilities
- MediaPipe for face mesh detection
- TensorFlow for ML model implementation
- Streamlit for the web interface 
