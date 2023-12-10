import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, cheby1
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Function to apply FFT
def apply_fft(data, sampling_rate):
    n = len(data)
    fft_result = np.fft.fft(data)
    freq = np.fft.fftfreq(n, 1 / sampling_rate)
    return freq, np.abs(fft_result)

# Function to apply Wiener filter
def apply_wiener_filter(noisy_signal, noise_level):
    signal_power = np.mean(noisy_signal ** 2)
    noise_power = noise_level ** 2
    transfer_function = 1 - noise_power / signal_power
    filtered_signal = noisy_signal * transfer_function
    return filtered_signal

# Function to apply IIR filter
def apply_iir_filter(data, cutoff_freq, sampling_rate, filter_type='butter'):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist

    if filter_type == 'butter':
        b, a = butter(4, normal_cutoff, btype='low', analog=False)
    elif filter_type == 'chebyshev':
        b, a = cheby1(4, 0.5, normal_cutoff, btype='low', analog=False)
    else:
        raise ValueError("Invalid filter type")

    filtered_data = lfilter(b, a, data)
    return filtered_data

# Function to classify motor signal
def classify_motor_signal(data, sampling_rate, noise_level):
    _, fft_result = apply_fft(data, sampling_rate)
    num_components = 5
    threshold_multiplier = 1.5
    
    # Calculate mean amplitude of the first few frequency components
    first_few_amplitudes = np.abs(fft_result[1:num_components + 1])
    mean_first_few_amplitude = np.mean(first_few_amplitudes)
    
    # Calculate mean amplitude of the entire FFT spectrum
    mean_total_amplitude = np.mean(np.abs(fft_result))
    
    # Calculate the ratio of mean_first_few_amplitude to mean_total_amplitude
    ratio = mean_first_few_amplitude / mean_total_amplitude
    
    # Set the threshold based on the ratio
    threshold = threshold_multiplier
    
    if ratio > threshold:
        return "Faulty Motor: High-frequency components detected."
    else:
        return "Healthy Motor: No significant fault detected."

# Function to generate power spectrum
def generate_power_spectrum(signal, sampling_rate):
    n = len(signal)
    frequencies = np.fft.fftfreq(n, d=1/sampling_rate)
    spectrum = np.abs(np.fft.fft(signal))
    return frequencies, spectrum

# Function to identify peak in the power spectrum
def identify_peak(frequencies, spectrum):
    peak_frequency = frequencies[np.argmax(spectrum)]
    peak_amplitude = np.max(spectrum)
    return peak_frequency, peak_amplitude

# Function to classify motor based on power spectrum
def classify_motor_spectrum(frequencies, spectrum, threshold=0.1):
    peak_frequency, peak_amplitude = identify_peak(frequencies, spectrum)
    if peak_amplitude > threshold:
        return f"Faulty Motor: High amplitude at peak frequency {peak_frequency} Hz."
    else:
        return "Healthy Motor: No significant fault detected."

# Streamlit app
def main():
    st.title("Motor Signal Analysis App")

    # Define tabs for different scenarios
    tab_names = ["No Load - Healthy", "No Load - Faulty", "Full Load - Healthy", "Full Load - Faulty"]
    selected_tab = st.radio("Select Scenario", tab_names)

    # Set default values
    default_noise_level = 0.1
    default_cutoff_freq = 50

    # Upload CSV file
    uploaded_file = st.file_uploader(f"Upload CSV for {selected_tab}", type=["csv"])

    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        data_column = df.columns[0]  # Assume data is in the first column
        data = df[data_column].values

        # Display original signal graph
        st.subheader("Original Signal")
        original_trace = go.Scatter(x=df.index, y=df[data_column], mode='lines', name='Original Signal')
        st.plotly_chart([original_trace])

        # Sampling rate
        sampling_rate = st.number_input("Enter the sampling rate", min_value=1, value=1000)

        # Apply FFT
        freq, fft_result = apply_fft(data, sampling_rate)

        # Plot FFT
        st.subheader("FFT Analysis")
        fft_trace = go.Scatter(x=np.abs(freq), y=np.abs(fft_result), mode='lines', name='FFT Result')

        # Set layout parameters
        layout_fft = go.Layout(
            title='FFT Result Analysis',
            xaxis=dict(title='Frequency (Hz)'),
            yaxis=dict(title='Amplitude'),  # Remove or comment out 'autorange' setting
        )

        st.plotly_chart(go.Figure(data=[fft_trace], layout=layout_fft))

        # Apply selected filter
        apply_filter_option = st.checkbox("Apply Filter")
        if apply_filter_option:
            # Cutoff frequency for IIR filter
            cutoff_freq = st.number_input("Enter the cutoff frequency", min_value=1, value=default_cutoff_freq)

            # Filter type for IIR filter
            filter_type = st.selectbox("Select Filter Type", ['butter', 'chebyshev', 'wiener'], index=0)

            # Apply selected filter
            if filter_type == 'butter':
                filtered_data = apply_iir_filter(fft_result, cutoff_freq, sampling_rate, filter_type='butter')
            elif filter_type == 'chebyshev':
                filtered_data = apply_iir_filter(fft_result, cutoff_freq, sampling_rate, filter_type='chebyshev')
            elif filter_type == 'wiener':
                filtered_data = apply_wiener_filter(fft_result, default_noise_level)

            # Generate power spectrum after applying filter
            frequencies_spectrum, spectrum = generate_power_spectrum(filtered_data, sampling_rate)

            # Identify peak in power spectrum
            peak_frequency, peak_amplitude = identify_peak(frequencies_spectrum, spectrum)

            # Plot the results using Plotly
            st.subheader(f"{filter_type.capitalize()} Filtered Signal")
            filtered_trace = go.Scatter(x=np.abs(freq), y=np.abs(filtered_data), mode='lines',
                                    name=f'{filter_type.capitalize()} Filtered Signal')
            layout_filtered = go.Layout(
            title='Filtered signal',
            xaxis=dict(title='Frequency (Hz)'),
            yaxis=dict(title='Amplitude'),  # Remove or comment out 'autorange' setting
        )

            st.plotly_chart(go.Figure(data=[filtered_trace], layout=layout_filtered))
        

            st.subheader(f"Power Spectrum after {filter_type.capitalize()} Filter")
            power_spectrum_trace = go.Scatter(x=frequencies_spectrum, y=np.abs(spectrum), mode='lines',
                                            name=f'Power Spectrum ({filter_type.capitalize()} Filter)')

            # Set layout parameters for power spectrum plot
            layout_power_spectrum = go.Layout(
                title=f'Power Spectrum after {filter_type.capitalize()} Filter',
                xaxis=dict(title='Frequency (Hz)'),
                yaxis=dict(title='Amplitude'),
            )

            st.plotly_chart(go.Figure(data=[power_spectrum_trace], layout=layout_power_spectrum))

            # Display peak frequency and amplitude
            st.subheader("Peak in Power Spectrum")
            st.write(f"Frequency: {peak_frequency} Hz, Amplitude: {peak_amplitude}")

            # Classify motor signal based on power spectrum
            motor_status_spectrum = classify_motor_spectrum(frequencies_spectrum, spectrum)
            st.info(f"Motor Status (Power Spectrum after {filter_type.capitalize()} Filter): {motor_status_spectrum}")

# Run the app
if __name__ == "__main__":
    main()
