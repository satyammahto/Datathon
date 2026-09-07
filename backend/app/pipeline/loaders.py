"""
Safe and versatile dataset loaders for CSV, XLSX, XLS, and JSON files.
Includes encoding auto-detection, delimiter inference, and malformed structure resilience.
"""
import os
import csv
from typing import Tuple, Dict, Any, Optional
import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json"}


def detect_csv_dialect(file_path: str, encoding: str = "utf-8") -> Tuple[str, Optional[str]]:
    """
    Detect the delimiter of a CSV file using sample lines.
    Defaults to ',' if detection fails.
    """
    delimiters = [",", ";", "\t", "|"]
    try:
        with open(file_path, "r", encoding=encoding, errors="replace") as f:
            sample = ""
            for _ in range(20):
                line = f.readline()
                if not line:
                    break
                sample += line
            
            if not sample.strip():
                return ",", None
            
            sniffer = csv.Sniffer()
            try:
                dialect = sniffer.sniff(sample, delimiters=delimiters)
                return dialect.delimiter, None
            except Exception:
                # Count delimiter frequencies on non-empty lines
                lines = [l for l in sample.splitlines() if l.strip()]
                if lines:
                    best_delim = ","
                    best_count = -1
                    for d in delimiters:
                        counts = [l.count(d) for l in lines]
                        if counts and counts[0] > 0 and len(set(counts)) == 1:
                            return d, None
                        total_d = sum(counts)
                        if total_d > best_count:
                            best_count = total_d
                            best_delim = d
                    return best_delim, None
    except Exception as e:
        return ",", str(e)
    return ",", None


def load_dataset(file_path: str, sheet_name: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load a dataset securely from file path.
    Supports CSV, TSV, XLSX, XLS, and JSON.
    Returns:
        (pd.DataFrame, file_info_dict)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at path: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file format '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise ValueError("Cannot load an empty file (0 bytes).")

    df = None
    encoding_used = "utf-8"
    delimiter_used = ","

    if ext in (".csv", ".tsv", ".txt"):
        encodings_to_try = ["utf-8", "utf-8-sig", "latin1", "cp1252", "iso-8859-1"]
        last_error = None
        for enc in encodings_to_try:
            try:
                delim, _ = detect_csv_dialect(file_path, encoding=enc)
                delimiter_used = delim
                # Try reading with C engine first for speed, fallback to python engine on malformed rows
                try:
                    df = pd.read_csv(file_path, encoding=enc, sep=delimiter_used, on_bad_lines="skip")
                except Exception:
                    df = pd.read_csv(file_path, encoding=enc, sep=delimiter_used, engine="python", on_bad_lines="skip")
                
                encoding_used = enc
                break
            except (UnicodeDecodeError, UnicodeError) as e:
                last_error = e
                continue
            except Exception as e:
                last_error = e
                continue

        if df is None:
            raise ValueError(f"Failed to decode and parse CSV file with any standard encoding: {last_error}")

    elif ext in (".xlsx", ".xls"):
        try:
            excel_file = pd.ExcelFile(file_path, engine="openpyxl" if ext == ".xlsx" else None)
            target_sheet = sheet_name or excel_file.sheet_names[0]
            df = pd.read_excel(excel_file, sheet_name=target_sheet)
            sheet_name = target_sheet
            encoding_used = "binary"
            delimiter_used = None
        except Exception as e:
            raise ValueError(f"Failed to read Excel workbook: {str(e)}")

    elif ext == ".json":
        try:
            df = pd.read_json(file_path)
            encoding_used = "utf-8"
            delimiter_used = None
        except Exception:
            try:
                # Try lines format
                df = pd.read_json(file_path, lines=True)
                encoding_used = "utf-8"
                delimiter_used = None
            except Exception as e:
                raise ValueError(f"Failed to parse JSON file: {str(e)}")

    if df is None or df.empty or df.shape[0] == 0:
        raise ValueError(f"Dataset in '{file_path}' contains zero data rows (header only or empty).")

    # Clean whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]

    file_info = {
        "file_path": file_path,
        "file_type": ext.lstrip("."),
        "file_size_bytes": file_size,
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "encoding": encoding_used,
        "delimiter": delimiter_used,
        "sheet_name": sheet_name,
    }

    return df, file_info
