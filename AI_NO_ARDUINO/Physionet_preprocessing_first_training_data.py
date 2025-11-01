"""
afdb_feature_extraction.py (versión comentada)

Script ligero y eficiente para extraer datos de la base de datos MIT-BIH Atrial Fibrillation Database (AFDB)
directamente desde PhysioNet, con detección automática de la derivación MLII (Lead II).

Características principales:
- Carga los registros por bloques, evitando consumir mucha memoria.
- Detecta automáticamente qué canal corresponde a la derivación MLII.
- Permite extraer características básicas (media, desviación, HR estimado, etc.) por bloques.
- Escribre resultados de manera incremental en un CSV.

Requisitos:
  pip install wfdb numpy pandas scipy

Uso:
  # Procesar un solo registro (detección automática de MLII)
  python afdb_feature_extraction.py --records 04015 --channels auto

  # Procesar varios registros manualmente
  python afdb_feature_extraction.py --records 04015,04043 --channels 0

  # Procesar todos los registros disponibles en PhysioNet
  python afdb_feature_extraction.py --all-records
"""

from __future__ import annotations
import argparse
import logging
import math
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, find_peaks

# ------------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ------------------------------------------------------------------
DEFAULT_PN_DIR = "afdb/1.0.0"  # Ruta base de PhysioNet

# ------------------------------------------------------------------
# CONFIGURACIÓN DE LOGGING
# ------------------------------------------------------------------


def setup_logging(level: str = "INFO") -> None:
    """Configura el formato y nivel del registro en consola."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


# ------------------------------------------------------------------
# PARSEO DE ARGUMENTOS
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Define y lee los argumentos de línea de comandos."""
    p = argparse.ArgumentParser(
        description="AFDB extraction with automatic MLII channel detection."
    )
    p.add_argument(
        "--records",
        type=str,
        default="",
        help="Registros a procesar, separados por coma (e.g. 04015,04043).",
    )
    p.add_argument(
        "--all-records",
        action="store_true",
        help="Procesar todos los registros disponibles en PhysioNet.",
    )
    p.add_argument(
        "--pn-dir", default=DEFAULT_PN_DIR, help="Ruta del dataset en PhysioNet."
    )
    p.add_argument(
        "--channels",
        default="auto",
        help="Índices de canales (0,1,...) o 'auto' para detectar MLII automáticamente.",
    )
    p.add_argument(
        "--block-sec",
        type=float,
        default=60.0,
        help="Duración de cada bloque de lectura en segundos.",
    )
    p.add_argument("--out", default="afdb_features.csv", help="Archivo CSV de salida.")
    p.add_argument(
        "--force-fs",
        type=float,
        default=0.0,
        help="Frecuencia de muestreo forzada (si no se encuentra en el header).",
    )
    p.add_argument(
        "--log", default="INFO", help="Nivel de logging (DEBUG/INFO/WARNING)."
    )
    return p.parse_args()


# ------------------------------------------------------------------
# FUNCIONES AUXILIARES DE SEÑAL
# ------------------------------------------------------------------


def to_float32(x: np.ndarray) -> np.ndarray:
    """Convierte el array a float32 para ahorrar memoria."""
    return x.astype(np.float32, copy=False)


def bandpass_signal(
    sig: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0, order: int = 3
) -> np.ndarray:
    """Aplica un filtro pasabanda Butterworth a la señal."""
    nyq = 0.5 * fs
    lown, highn = low / nyq, high / nyq
    b, a = butter(order, [lown, highn], btype="band")
    return to_float32(filtfilt(b, a, sig))


# ------------------------------------------------------------------
# EXTRACCIÓN DE FEATURES
# ------------------------------------------------------------------


def extract_basic_time_features(block: np.ndarray) -> dict:
    """Calcula características estadísticas básicas del bloque de señal."""
    x = block
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "rms": float(np.sqrt(np.mean(x * x))),
        "zcr": float(((x[:-1] * x[1:]) < 0).sum()) / max(1, x.size - 1),
    }


def estimate_hr_from_peaks(block: np.ndarray, fs: float) -> tuple[float, int]:
    """Calcula una estimación simple de la frecuencia cardíaca (bpm) a partir de los picos R."""
    if block.size < int(0.5 * fs):
        return float("nan"), 0
    try:
        sig_f = bandpass_signal(block, fs, 5.0, 25.0, 2)
    except Exception:
        sig_f = block
    peaks, _ = find_peaks(sig_f, distance=int(0.2 * fs))
    if len(peaks) <= 1:
        return float("nan"), len(peaks)
    rr = np.diff(peaks) / fs
    hr = 60.0 / np.mean(rr)
    return float(hr), len(peaks)


def extract_features_block(block: np.ndarray, fs: float) -> dict:
    """Combina las características básicas y la HR estimada de un bloque."""
    feats = extract_basic_time_features(block)
    hr, n = estimate_hr_from_peaks(block, fs)
    feats.update({"hr_bpm": hr, "n_peaks": n})
    return feats


# ------------------------------------------------------------------
# DETECCIÓN AUTOMÁTICA DE DERIVACIÓN MLII
# ------------------------------------------------------------------


def detect_mlii_channel(record_name: str, pn_dir: str) -> Optional[int]:
    """Detecta automáticamente el índice del canal MLII (Lead II) en el registro."""
    try:
        header = wfdb.rdheader(record_name, pn_dir=pn_dir)
        sig_names = [s.lower() for s in getattr(header, "sig_name", [])]
        for i, name in enumerate(sig_names):
            if "mlii" in name or "ii" in name:
                logging.info(
                    f"Canal MLII detectado para {record_name}: índice {i} ({header.sig_name[i]})"
                )
                return i
    except Exception as e:
        logging.warning(f"No se pudo detectar el canal MLII para {record_name}: {e}")
    return None


# ------------------------------------------------------------------
# PROCESAMIENTO PRINCIPAL POR BLOQUES
# ------------------------------------------------------------------


def process_record_streaming(
    record_name: str,
    pn_dir: str,
    channels: List[int],
    block_sec: float,
    out_csv: str,
    force_fs: Optional[float] = None,
) -> None:
    """Procesa un registro en bloques de tiempo y guarda las características en CSV."""
    logging.info(f"Procesando registro {record_name} (pn_dir={pn_dir})")

    # Leer encabezado
    header = wfdb.rdheader(record_name, pn_dir=pn_dir)
    fs = float(force_fs) if force_fs else float(getattr(header, "fs", 250.0))
    sig_len = int(getattr(header, "sig_len", 0))
    block_samples = int(block_sec * fs)

    write_header = not os.path.exists(out_csv)
    samp_start, block_idx = 0, 0

    # Procesamiento por bloques
    while samp_start < sig_len:
        samp_end = min(samp_start + block_samples, sig_len)
        rec = wfdb.rdrecord(
            record_name,
            pn_dir=pn_dir,
            sampfrom=samp_start,
            sampto=samp_end,
            channels=channels,
        )
        p_signal = to_float32(rec.p_signal)
        rows = []

        # Procesar cada canal solicitado
        for ch_idx, ch in enumerate(channels):
            sig = p_signal[:, ch_idx] if p_signal.ndim == 2 else p_signal
            feats = extract_features_block(sig, fs)
            rows.append(
                {
                    "record": record_name,
                    "channel": ch,
                    "block_idx": block_idx,
                    "t_start": samp_start / fs,
                    "t_end": samp_end / fs,
                    "n_samples": len(sig),
                    "fs": fs,
                    **feats,
                }
            )

        # Guardar resultados incrementales
        pd.DataFrame(rows).to_csv(out_csv, mode="a", index=False, header=write_header)
        write_header = False

        samp_start = samp_end
        block_idx += 1
        logging.info(f"Bloque {block_idx} procesado para {record_name}")


# ------------------------------------------------------------------
# UTILIDAD PARA OBTENER LISTA DE REGISTROS REMOTOS
# ------------------------------------------------------------------


def get_remote_record_list(pn_dir: str) -> List[str]:
    """Obtiene la lista de registros disponibles desde PhysioNet."""
    try:
        return list(wfdb.get_record_list(pn_dir))
    except Exception as e:
        logging.warning(f"No se pudo obtener la lista de registros: {e}")
        return []


# ------------------------------------------------------------------
# FUNCIÓN PRINCIPAL
# ------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    setup_logging(args.log)

    # Determinar qué registros procesar
    if args.all_records:
        recs = get_remote_record_list(args.pn_dir)
    else:
        recs = [r.strip() for r in args.records.split(",") if r.strip()]

    if not recs:
        logging.error("No se especificaron registros para procesar.")
        return

    # Procesar cada registro individualmente
    for rec in recs:
        # Detección automática del canal MLII si está habilitada
        if args.channels == "auto":
            ch = detect_mlii_channel(rec, args.pn_dir)
            if ch is None:
                logging.warning(f"No se encontró canal MLII para {rec}, se omite.")
                continue
            channels = [ch]
        else:
            channels = [int(c) for c in args.channels.split(",") if c.strip()]

        process_record_streaming(
            rec,
            pn_dir=args.pn_dir,
            channels=channels,
            block_sec=args.block_sec,
            out_csv=args.out,
            force_fs=args.force_fs,
        )


if __name__ == "__main__":
    main()
