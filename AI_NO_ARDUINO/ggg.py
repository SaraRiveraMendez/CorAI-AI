# Modifica las columnas del DataFrame
df["heart_rate_bpm"] = np.nan  # BPM (ritmo cardíaco)
df["dominant_freq_hz"] = np.nan  # Hz (frecuencia dominante de señal)

# Calcular frecuencia dominante para cada ventana (ej. cada segundo)
window_size_sec = 1.0
window_samples = int(window_size_sec * fs)

for i in range(0, len(df) - window_samples, window_samples):
    ecg_window = df["ECG2"].iloc[i : i + window_samples].values

    # Calcular frecuencia dominante
    fft_result = np.fft.fft(ecg_window)
    freqs = np.fft.fftfreq(len(ecg_window), d=1 / fs)

    # Solo frecuencias positivas
    positive_mask = freqs > 0
    freqs_pos = freqs[positive_mask]
    magnitude = np.abs(fft_result[positive_mask])

    # Frecuencia dominante (excluyendo DC)
    if len(magnitude) > 0:
        dominant_idx = np.argmax(magnitude[1:]) + 1  # Saltar DC
        dominant_freq = freqs_pos[dominant_idx]

        # Asignar al DataFrame
        df.iloc[i : i + window_samples, df.columns.get_loc("dominant_freq_hz")] = (
            dominant_freq
        )

# Guardar
print(f"[INFO] Columnas guardadas: {list(df.columns)}")
# Ahora tendrás: ECG2, time_sec, heart_rate_bpm, dominant_freq_hz
