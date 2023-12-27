Motor Signal Analysis App
Overview
This Streamlit web application is designed for analyzing motor signals, applying various filters, and classifying the motor's health based on the signal characteristics. It includes functionalities such as Fast Fourier Transform (FFT) analysis, IIR filtering, Wiener filtering, and power spectrum analysis.

Requirements
pip install streamlit pandas numpy scipy plotly matplotlib
Usage

Clone the Repository:
git clone https://github.com/wadewayne001/motor-signal-analysis-app.git
cd motor-signal-analysis-app

Run the App:
streamlit run app.py

Open the App in Browser:

Visit http://localhost:8501 in your web browser to access the app.

App Features
Uploading Data
Upload a CSV file containing motor signal data.
Choose from predefined scenarios (No Load - Healthy, No Load - Faulty, Full Load - Healthy, Full Load - Faulty).
Original Signal Analysis
Display the original motor signal graph.
FFT Analysis
Visualize the FFT analysis of the motor signal.
Apply Filters
Checkbox to apply filters (Butterworth, Chebyshev, Wiener).
Adjust cutoff frequency for IIR filters.
Filtered Signal Analysis
Display the signal after applying the selected filter.
Generate and display the power spectrum after filtering.
Identify and display the peak frequency and amplitude in the power spectrum.
Motor Classification
Classify the motor's health based on the power spectrum characteristics.
Contributing
Feel free to contribute to the development of this app by opening issues or submitting pull requests.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Feel free to customize the README to fit your project structure and add more details as needed.




