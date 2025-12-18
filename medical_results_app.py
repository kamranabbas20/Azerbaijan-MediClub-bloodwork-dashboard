import os
import re
from typing import Dict, Tuple, Optional, Callable, Any, List

import pandas as pd
import pdfplumber
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def extract_pdf_results(pdf_path: str) -> Tuple[Optional[str], pd.DataFrame]:
	"""Extract the test results table and date from a single PDF.

	Returns (date_string, results_df).
	- date_string is in the format "dd.mm.yyyy" or None if not found.
	- results_df has columns ["Test", "Result", "Unit", "Reference Range"].
	"""

	tables = []
	date: Optional[str] = None

	with pdfplumber.open(pdf_path) as pdf:
		extract_tables_from_page = False

		for page in pdf.pages:
			text = page.extract_text()

			if text:
				if "LABORATORY RESULTS" in text:
					extract_tables_from_page = True

					for line in text.split("\n"):
						if "Date" in line:
							match = re.search(r"(\d{2}\.\d{2}\.\d{4}) (\d{2}:\d{2})", line)
							if match:
								date = match.group(1)

				if extract_tables_from_page:
					extracted_tables = page.extract_tables() or []
					for table in extracted_tables:
						if table:
							tables.append(table)

	df_list = []
	for table in tables:
		df = pd.DataFrame(table)
		df_list.append(df)

	if df_list:
		results = pd.concat(df_list, ignore_index=True)
		if results.shape[1] >= 4:
			results = results.iloc[:, :4]
			results.columns = ["Test", "Result", "Unit", "Reference Range"]
		else:
			results = pd.DataFrame(columns=["Test", "Result", "Unit", "Reference Range"])
	else:
		results = pd.DataFrame(columns=["Test", "Result", "Unit", "Reference Range"])

	return date, results


def parse_lab_results_folder(
	folder_path: str,
	progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
	"""Parse all PDF lab results in a folder into a single table.

	Returns (df, test_metadata)
	- df:
	  - Rows: test names
	  - Columns: dates (dd.mm.yyyy)
	  - Values: test result values as strings
	- test_metadata:
	  - test name -> {"unit": str, "ref_raw": str}
	"""

	if not os.path.isdir(folder_path):
		raise ValueError(f"Folder does not exist: {folder_path}")

	pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
	total_files = len(pdf_files)

	merged_data: Dict[str, Dict[str, str]] = {}
	test_metadata: Dict[str, Dict[str, Any]] = {}

	for idx, pdf_file in enumerate(pdf_files):
		pdf_path = os.path.join(folder_path, pdf_file)

		if progress_callback is not None:
			progress_callback(idx, total_files, pdf_file)

		date, results_df = extract_pdf_results(pdf_path)

		if not date or results_df.empty:
			continue

		for _, row in results_df.iterrows():
			test_name = str(row.get("Test", "")).strip()
			test_value = str(row.get("Result", "")).strip()
			unit = str(row.get("Unit", "")).strip()
			ref_range = str(row.get("Reference Range", "")).strip()

			if not test_name:
				continue

			if test_name not in merged_data:
				merged_data[test_name] = {}

			merged_data[test_name][date] = test_value

			# Collect unit and raw reference-range string for this test
			if test_name not in test_metadata:
				test_metadata[test_name] = {"unit": "", "ref_raw": ""}
			info = test_metadata[test_name]
			if unit and not info.get("unit"):
				info["unit"] = unit
			if ref_range and not info.get("ref_raw"):
				info["ref_raw"] = ref_range

	if not merged_data:
		return pd.DataFrame(), {}

	df = pd.DataFrame.from_dict(merged_data, orient="index")

	# Remove rows where all values are non-numeric (likely text/header rows)
	numeric_df = df.apply(pd.to_numeric, errors="coerce")
	mask_keep = numeric_df.notna().any(axis=1)
	df = df[mask_keep]

	try:
		df = df.reindex(
			sorted(df.columns, key=lambda x: pd.to_datetime(x, format="%d.%m.%Y")),
			axis=1,
		)
	except Exception:
		df = df.reindex(sorted(df.columns), axis=1)

	df.index.name = "Test"

	return df, test_metadata


def parse_lab_results_uploaded(
	files: List[Any],
	progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
	"""Parse uploaded PDF lab results into a single table.

	`files` is a list of file-like objects (e.g. Streamlit UploadedFile).
	Returns (df, test_metadata) in the same format as parse_lab_results_folder.
	"""

	if not files:
		return pd.DataFrame(), {}

	merged_data: Dict[str, Dict[str, str]] = {}
	test_metadata: Dict[str, Dict[str, Any]] = {}
	total_files = len(files)

	for idx, file_obj in enumerate(files):
		filename = getattr(file_obj, "name", f"file_{idx + 1}.pdf")
		if hasattr(file_obj, "seek"):
			file_obj.seek(0)

		if progress_callback is not None:
			progress_callback(idx, total_files, filename)

		# Reuse existing extractor; pdfplumber.open accepts file-like objects
		date, results_df = extract_pdf_results(file_obj)

		if not date or results_df.empty:
			continue

		for _, row in results_df.iterrows():
			test_name = str(row.get("Test", "")).strip()
			test_value = str(row.get("Result", "")).strip()
			unit = str(row.get("Unit", "")).strip()
			ref_range = str(row.get("Reference Range", "")).strip()

			if not test_name:
				continue

			if test_name not in merged_data:
				merged_data[test_name] = {}

			merged_data[test_name][date] = test_value

			if test_name not in test_metadata:
				test_metadata[test_name] = {"unit": "", "ref_raw": ""}
			info = test_metadata[test_name]
			if unit and not info.get("unit"):
				info["unit"] = unit
			if ref_range and not info.get("ref_raw"):
				info["ref_raw"] = ref_range

	if not merged_data:
		return pd.DataFrame(), {}

	df = pd.DataFrame.from_dict(merged_data, orient="index")

	numeric_df = df.apply(pd.to_numeric, errors="coerce")
	mask_keep = numeric_df.notna().any(axis=1)
	df = df[mask_keep]

	try:
		df = df.reindex(
			sorted(df.columns, key=lambda x: pd.to_datetime(x, format="%d.%m.%Y")),
			axis=1,
		)
	except Exception:
		df = df.reindex(sorted(df.columns), axis=1)

	df.index.name = "Test"

	return df, test_metadata


def prepare_numeric_series(df: pd.DataFrame, test_name: str) -> Optional[Tuple[pd.Series, pd.Series]]:
	"""Prepare cleaned date and numeric value series for a given test.

	Returns (dates, values) or None if not usable.
	"""

	if df.empty or test_name not in df.index:
		return None

	series = df.loc[test_name].dropna()
	if series.empty:
		return None

	values = pd.to_numeric(series.values, errors="coerce")
	mask = ~pd.isna(values)
	if mask.sum() == 0:
		return None

	dates = pd.to_datetime(series.index[mask], format="%d.%m.%Y", errors="coerce")
	values = values[mask]

	mask_valid = ~pd.isna(dates)
	if mask_valid.sum() == 0:
		return None

	dates = dates[mask_valid]
	values = values[mask_valid]

	return dates, values


def plot_test_timeseries(df: pd.DataFrame, test_name: str) -> None:
	"""Plot a single test over time with optional reference-range band."""

	prepared = prepare_numeric_series(df, test_name)
	if prepared is None:
		st.warning("Not enough numeric data for this test to draw a time series.")
		return

	dates, values = prepared

	fig, ax = plt.subplots(figsize=(5, 3))
	ax.plot(dates, values, marker="o", color="tab:blue")

	# Reference-range band, if known (prefers PDF metadata)
	ref_rng = get_reference_range(test_name)
	if ref_rng is not None:
		low, high = ref_rng
		ax.axhspan(low, high, color="tab:green", alpha=0.12, label="Reference range")
		ax.legend(fontsize=8, loc="best")

	ax.set_title(test_name)
	ax.set_xlabel("Date")
	# Use unit in y-axis label if available
	meta: Dict[str, Dict[str, Any]] = st.session_state.get("test_metadata") or {}
	unit = ""
	info = meta.get(test_name)
	if info and isinstance(info, dict):
		unit = str(info.get("unit", ""))
	y_label = f"Result ({unit})" if unit else "Result"
	ax.set_ylabel(y_label)
	plt.xticks(rotation=45)
	plt.tight_layout()

	st.pyplot(fig)


def plot_test_histogram(df: pd.DataFrame, test_name: str) -> None:
	"""Plot a histogram of values for a given test."""
	prepared = prepare_numeric_series(df, test_name)
	if prepared is None:
		st.warning("Not enough numeric data for this test to draw a histogram.")
		return

	_, values = prepared

	fig, ax = plt.subplots(figsize=(4, 3))
	ax.hist(values, bins="auto", alpha=0.7, color="tab:orange", edgecolor="black")
	ax.set_title("Value distribution")
	ax.set_xlabel("Result")
	ax.set_ylabel("Count")
	plt.tight_layout()
	st.pyplot(fig)


TEST_ABBREVIATIONS: Dict[str, str] = {
	"WBC": "White blood cell count",
	"RBC": "Red blood cell count",
	"HGB": "Hemoglobin",
	"HCT": "Hematocrit",
	"PLT": "Platelet count",
	"CRP": "C-reactive protein",
	"ESR": "Erythrocyte sedimentation rate",
	"ALT": "Alanine aminotransferase",
	"AST": "Aspartate aminotransferase",
	"ALP": "Alkaline phosphatase",
	"GGT": "Gamma-glutamyl transferase",
	"LDL": "Low-density lipoprotein cholesterol",
	"HDL": "High-density lipoprotein cholesterol",
	"TSH": "Thyroid-stimulating hormone",
	"FT4": "Free thyroxine",
	"FT3": "Free triiodothyronine",
	"HBA1C": "Glycated hemoglobin (average blood glucose)",
	"GLU": "Glucose",
	"CREAT": "Creatinine",
	"UREA": "Urea",
	"VITAMIN D 25OH": "25-hydroxy vitamin D (Vitamin D status)",
	"MCV": "Mean corpuscular volume",
	"MCH": "Mean corpuscular hemoglobin",
	"MCHC": "Mean corpuscular hemoglobin concentration",
	"RDW-SD": "Red cell distribution width (standard deviation)",
	"RDW-CV": "Red cell distribution width (coefficient of variation)",
	"PDW": "Platelet distribution width",
	"MPV": "Mean platelet volume",
	"P-LCR": "Platelet large cell ratio",
	"ASO": "Antistreptolysin O titer",
}


# Simple reference ranges (illustrative; not individualized medical advice)
REFERENCE_RANGES: Dict[str, Tuple[float, float]] = {
	"WBC": (4.0, 11.0),
	"RBC": (4.0, 6.0),
	"HGB": (120.0, 170.0),  # g/L
	"HCT": (0.36, 0.50),    # fraction
	"PLT": (150.0, 450.0),
	"GLU": (3.9, 6.1),      # mmol/L fasting
	"HBA1C": (4.0, 6.0),    # %
	"CREAT": (60.0, 115.0),
	"UREA": (2.5, 7.8),
	"ALT": (0.0, 40.0),
	"AST": (0.0, 40.0),
	"LDL": (0.0, 3.0),
	"HDL": (1.0, 2.5),
	"TSH": (0.4, 4.0),
}


def parse_reference_range(ref_raw: str) -> Optional[Tuple[float, float]]:
	"""Parse a reference-range string like "3.9 - 6.1" into (low, high)."""
	if not ref_raw:
		return None
	# Look for two numbers separated by '-' or '–'
	match = re.search(r"(-?\d+(?:[.,]\d+)?)\s*[-–]\s*(-?\d+(?:[.,]\d+)?)", ref_raw)
	if not match:
		return None
	try:
		low = float(match.group(1).replace(",", "."))
		high = float(match.group(2).replace(",", "."))
	except ValueError:
		return None
	return (low, high) if low <= high else (high, low)


def get_reference_range(test_name: str) -> Optional[Tuple[float, float]]:
	"""Return reference range for a test.

	Priority:
	1) Use range parsed from PDF "Reference Range" column if available.
	2) Fall back to simple hard-coded REFERENCE_RANGES.
	"""
	# 1) Try metadata from parsed PDFs (stored in session_state)
	meta: Dict[str, Dict[str, Any]] = st.session_state.get("test_metadata") or {}
	name_upper = test_name.upper()

	# Exact match first
	info = meta.get(test_name)
	if not info:
		# Try case-insensitive contains search
		for key, val in meta.items():
			if key.upper() in name_upper or name_upper in key.upper():
				info = val
				break
	if info and isinstance(info, dict):
		ref_raw = str(info.get("ref_raw", "")).strip()
		parsed = parse_reference_range(ref_raw)
		if parsed is not None:
			return parsed

	# 2) Fallback to simple built-in ranges
	for key, rng in REFERENCE_RANGES.items():
		if key in name_upper:
			return rng
	return None


def plot_scatter_relationship(df: pd.DataFrame, test_x: str, test_y: str) -> None:
	"""Scatter plot with regression line between two tests."""
	if df.empty or test_x not in df.index or test_y not in df.index:
		st.warning("Selected tests are not available in the table.")
		return

	sx = pd.to_numeric(df.loc[test_x], errors="coerce")
	sy = pd.to_numeric(df.loc[test_y], errors="coerce")
	both = pd.concat([sx, sy], axis=1, keys=["x", "y"]).dropna()

	if len(both) < 2:
		st.warning("Not enough overlapping numeric data to draw a scatter plot.")
		return

	fig, ax = plt.subplots(figsize=(4, 4))
	sns.regplot(x="x", y="y", data=both, ax=ax, scatter_kws={"alpha": 0.7, "s": 30})
	ax.set_xlabel(test_x)
	ax.set_ylabel(test_y)
	ax.set_title(f"{test_y} vs {test_x}")
	plt.tight_layout()
	st.pyplot(fig)


def get_test_description(test_name: str) -> Optional[str]:
	"""Return a human-readable description for a test abbreviation, if known."""
	name_upper = test_name.upper()
	for key, desc in TEST_ABBREVIATIONS.items():
		if key in name_upper:
			return desc
	return None


def main() -> None:
	st.set_page_config(page_title="Lab Results Dashboard", layout="wide")
	st.title("Lab Results Dashboard")
	st.write(
		"Select the folder that contains your lab result PDFs. "
		"The app will parse them and build a table of test results by date."
	)

	with st.sidebar:
		st.header("Settings")
		base_dir = os.getcwd()
		data_source = st.radio(
			"Data source",
			("Folder on disk", "Upload PDFs"),
			key="data_source",
		)
		folder_path = ""
		uploaded_files: Optional[List[Any]] = None

		if data_source == "Folder on disk":
			folder_mode = st.radio(
				"How to choose folder?",
				("Browse project folders", "Enter custom path"),
				key="folder_mode",
			)
			if folder_mode == "Browse project folders":
				subdirs = []
				for root, dirs, files in os.walk(base_dir):
					rel_root = os.path.relpath(root, base_dir)
					if rel_root == ".":
						continue
					subdirs.append(rel_root)
				subdirs = sorted(set(subdirs))
				if subdirs:
					selected_rel = st.selectbox(
						"Folder with PDF files",
						options=subdirs,
						key="folder_select",
						help="Choose a folder inside this project that contains your .pdf lab results.",
					)
					folder_path = os.path.join(base_dir, selected_rel)
				else:
					st.info("No subfolders found in this project; please enter a custom path instead.")
					folder_path = st.text_input(
						"Folder with PDF files",
						value=base_dir,
						key="folder_input",
						help="Enter a folder path that contains your .pdf lab results.",
					)
			else:
				folder_path = st.text_input(
					"Folder with PDF files",
					value=base_dir,
					key="folder_input",
					help="Enter a folder path that contains your .pdf lab results.",
				)
		else:
			uploaded_files = st.file_uploader(
				"Upload PDF lab reports",
				accept_multiple_files=True,
				type=["pdf"],
				key="pdf_uploader",
			)

		parse_button = st.button("Parse PDFs", key="parse_button")

	if "parsed_df" not in st.session_state:
		st.session_state["parsed_df"] = None
	if "parsed_folder" not in st.session_state:
		st.session_state["parsed_folder"] = None
	if "test_metadata" not in st.session_state:
		st.session_state["test_metadata"] = {}

	if parse_button:
		with st.spinner("Parsing PDFs..."):
			progress_bar = st.progress(0)
			status_text = st.empty()

			def progress_cb(current: int, total: int, filename: str) -> None:
				if total > 0:
					percent = int(((current + 1) / total) * 100)
					progress_bar.progress(percent)
				status_text.text(f"Processing {current + 1}/{total} – {filename}")

			try:
				if data_source == "Folder on disk":
					if not folder_path or not os.path.isdir(folder_path):
						st.error("Please provide a valid existing folder path.")
						return
					parsed_df, test_metadata = parse_lab_results_folder(
						folder_path, progress_callback=progress_cb
					)
					st.session_state["parsed_folder"] = folder_path
				else:
					if not uploaded_files:
						st.error("Please upload at least one PDF file.")
						return
					parsed_df, test_metadata = parse_lab_results_uploaded(
						uploaded_files, progress_callback=progress_cb
					)
					st.session_state["parsed_folder"] = None

				st.session_state["parsed_df"] = parsed_df
				st.session_state["test_metadata"] = test_metadata
			except Exception as e:
				st.session_state["parsed_df"] = None
				st.error(f"Error while parsing PDFs: {e}")

	parsed_df: Optional[pd.DataFrame] = st.session_state.get("parsed_df")
	parsed_folder: Optional[str] = st.session_state.get("parsed_folder")
	test_metadata: Dict[str, Dict[str, Any]] = st.session_state.get("test_metadata") or {}

	if parsed_df is None:
		st.info("Enter a folder path and click 'Parse PDFs' to get started.")
		return

	if parsed_df.empty:
		st.warning("No lab results could be extracted from the PDFs in this folder.")
		return

	st.subheader("Merged Lab Results Table")

	# Add units and acronym explanations as columns next to the test name
	parsed_df_with_meaning = parsed_df.copy()
	meanings = []
	units = []
	for test_name in parsed_df_with_meaning.index:
		meaning = get_test_description(str(test_name)) or ""
		meanings.append(meaning)
		info = test_metadata.get(test_name, {})
		units.append(str(info.get("unit", "")))
	parsed_df_with_meaning.insert(0, "Unit", units)
	parsed_df_with_meaning.insert(1, "Meaning", meanings)

	st.dataframe(parsed_df_with_meaning)

	if parsed_folder:
		csv_path = os.path.join(parsed_folder, "merged_lab_results.csv")
		try:
			parsed_df.to_csv(csv_path, index=True)
			st.caption(f"Table also saved to: {csv_path}")
		except Exception:
			pass

	with st.expander("Test abbreviations (what each means)"):
		abbr_df = pd.DataFrame(
			[(k, v) for k, v in sorted(TEST_ABBREVIATIONS.items())],
			columns=["Abbreviation", "Meaning"],
		)
		st.table(abbr_df)

	st.subheader("Test Time Series")
	all_tests = list(parsed_df.index)
	selected_test = st.selectbox("Choose a test", options=all_tests)

	if selected_test:
		col_line, col_hist = st.columns([2, 1])
		with col_line:
			plot_test_timeseries(parsed_df, selected_test)
		with col_hist:
			plot_test_histogram(parsed_df, selected_test)

		desc = get_test_description(selected_test)
		if desc:
			st.caption(f"{selected_test}: {desc}")
		else:
			st.caption("No description available for this test abbreviation yet.")

	# Grouped panels for related tests
	st.subheader("Grouped panels for related tests")
	test_groups: Dict[str, list] = {
		"Complete blood count (CBC)": [
			"WBC",
			"RBC",
			"HGB",
			"HCT",
			"PLT",
			"MCV",
			"MCH",
			"MCHC",
			"RDW-SD",
			"RDW-CV",
			"MPV",
			"PDW",
			"P-LCR",
		],
		"Liver function": ["ALT", "AST", "ALP", "GGT"],
		"Lipid profile": ["LDL", "HDL"],
		"Thyroid function": ["TSH", "FT4", "FT3"],
		"Renal function": ["CREAT", "UREA"],
		"Glucose control": ["GLU", "HBA1C"],
	}

	selected_group = st.selectbox(
		"Choose a test group",
		options=list(test_groups.keys()),
		key="group_select",
	)

	if selected_group:
		group_tests = [t for t in test_groups[selected_group] if t in parsed_df.index]
		if not group_tests:
			st.info("No tests from this group are available in your data.")
		else:
			cols = st.columns(3)
			for i, tname in enumerate(group_tests):
				with cols[i % 3]:
					st.caption(tname)
					plot_test_timeseries(parsed_df, tname)

	# Scatter plots for key relationships
	st.subheader("Scatter plots for key relationships")
	col_x, col_y = st.columns(2)
	with col_x:
		x_test = st.selectbox("X-axis test", options=all_tests, key="scatter_x")
	with col_y:
		y_test = st.selectbox("Y-axis test", options=all_tests, key="scatter_y")

	if x_test and y_test:
		if x_test == y_test:
			st.info("Choose two different tests to see their relationship.")
		else:
			plot_scatter_relationship(parsed_df, x_test, y_test)

	st.subheader("Cross-correlation between tests")

	# Exclude tests that should not be part of correlation (e.g. Specific Gravity, Amount)
	exclude_patterns = ["specific grav", "specific gravity", "amount"]
	index_series = parsed_df.index.to_series().astype(str).str.lower()
	mask_rows = ~index_series.str.contains("|".join(exclude_patterns))
	filtered_df = parsed_df[mask_rows]

	numeric_df = filtered_df.apply(pd.to_numeric, errors="coerce")
	corr_df = numeric_df.T.corr(min_periods=2)
	corr_df = corr_df.dropna(how="all").dropna(axis=1, how="all")

	if corr_df.empty:
		st.info("Not enough numeric data to compute correlations between tests.")
	else:
		st.write("Correlation matrix (Pearson):")
		st.dataframe(corr_df)

		mask = np.triu(np.ones_like(corr_df, dtype=bool))
		fig, ax = plt.subplots(figsize=(7, 6))
		sns.heatmap(
			corr_df,
			mask=mask,
			ax=ax,
			cmap="vlag",
			center=0,
			square=True,
			annot=False,
			linewidths=0.3,
			linecolor="white",
			cbar_kws={"shrink": 0.8, "label": "Correlation"},
		)
		ax.set_title("Test cross-correlation heatmap")
		plt.xticks(rotation=45, ha="right", fontsize=8)
		plt.yticks(rotation=0, fontsize=8)
		plt.tight_layout()
		st.pyplot(fig)


if __name__ == "__main__":
	main()

