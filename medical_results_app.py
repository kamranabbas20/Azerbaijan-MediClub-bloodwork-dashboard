import os
import re
from typing import Dict, Tuple, Optional, Callable

import pandas as pd
import pdfplumber
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


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
) -> pd.DataFrame:
	"""Parse all PDF lab results in a folder into a single table.

	The returned DataFrame:
	- Rows: test names
	- Columns: dates (dd.mm.yyyy)
	- Values: test result values as strings
	"""

	if not os.path.isdir(folder_path):
		raise ValueError(f"Folder does not exist: {folder_path}")

	pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
	total_files = len(pdf_files)

	merged_data: Dict[str, Dict[str, str]] = {}

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

			if not test_name:
				continue

			if test_name not in merged_data:
				merged_data[test_name] = {}

			merged_data[test_name][date] = test_value

	if not merged_data:
		return pd.DataFrame()

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

	return df


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
	"""Plot a single test over time using Matplotlib inside Streamlit."""

	prepared = prepare_numeric_series(df, test_name)
	if prepared is None:
		st.warning("Not enough numeric data for this test to draw a time series.")
		return

	dates, values = prepared

	fig, ax = plt.subplots(figsize=(5, 3))
	ax.plot(dates, values, marker="o")
	ax.set_title(test_name)
	ax.set_xlabel("Date")
	ax.set_ylabel("Result")
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
}


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
		default_folder = os.getcwd()
		folder_path = st.text_input(
			"Folder with PDF files",
			value=default_folder,
			help="Enter a folder path on this computer that contains your .pdf lab results.",
		)
		parse_button = st.button("Parse PDFs")

	if "parsed_df" not in st.session_state:
		st.session_state["parsed_df"] = None
	if "parsed_folder" not in st.session_state:
		st.session_state["parsed_folder"] = None

	if parse_button:
		if not folder_path or not os.path.isdir(folder_path):
			st.error("Please provide a valid existing folder path.")
		else:
			with st.spinner("Parsing PDFs..."):
				progress_bar = st.progress(0)
				status_text = st.empty()

				def progress_cb(current: int, total: int, filename: str) -> None:
					if total > 0:
						percent = int(((current + 1) / total) * 100)
						progress_bar.progress(percent)
					status_text.text(f"Processing {current + 1}/{total} – {filename}")

				try:
					parsed_df = parse_lab_results_folder(
						folder_path, progress_callback=progress_cb
					)
					st.session_state["parsed_df"] = parsed_df
					st.session_state["parsed_folder"] = folder_path
				except Exception as e:
					st.session_state["parsed_df"] = None
					st.error(f"Error while parsing PDFs: {e}")

	parsed_df: Optional[pd.DataFrame] = st.session_state.get("parsed_df")
	parsed_folder: Optional[str] = st.session_state.get("parsed_folder")

	if parsed_df is None:
		st.info("Enter a folder path and click 'Parse PDFs' to get started.")
		return

	if parsed_df.empty:
		st.warning("No lab results could be extracted from the PDFs in this folder.")
		return

	st.subheader("Merged Lab Results Table")

	# Add acronym explanations as a column next to the test name
	parsed_df_with_meaning = parsed_df.copy()
	meanings = []
	for test_name in parsed_df_with_meaning.index:
		meaning = get_test_description(str(test_name)) or ""
		meanings.append(meaning)
	parsed_df_with_meaning.insert(0, "Meaning", meanings)

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

	st.subheader("Cross-correlation between tests")
	numeric_df = parsed_df.apply(pd.to_numeric, errors="coerce")
	corr_df = numeric_df.T.corr(min_periods=2)
	corr_df = corr_df.dropna(how="all").dropna(axis=1, how="all")

	if corr_df.empty:
		st.info("Not enough numeric data to compute correlations between tests.")
	else:
		st.write("Correlation matrix (Pearson):")
		st.dataframe(corr_df)

		fig, ax = plt.subplots(figsize=(6, 5))
		sns.heatmap(corr_df, ax=ax, cmap="coolwarm", center=0, square=True)
		ax.set_title("Test cross-correlation heatmap")
		plt.tight_layout()
		st.pyplot(fig)


if __name__ == "__main__":
	main()
import os
import re
from typing import Dict, Tuple, Optional, Callable

import pandas as pd
import pdfplumber
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


def extract_pdf_results(pdf_path: str) -> Tuple[Optional[str], pd.DataFrame]:
	"""Extracts the test results table and date from a single PDF.

	Returns (date_string, results_df).
	- date_string is in the format "dd.mm.yyyy" or None if not found.
	- results_df has columns ["Test", "Result", "Unit", "Reference Range"].
	"""

	tables = []
	date: Optional[str] = None

	with pdfplumber.open(pdf_path) as pdf:
		import os
		import re
		from typing import Dict, Tuple, Optional, Callable

		import pandas as pd
		import pdfplumber
		import streamlit as st
		import matplotlib.pyplot as plt
		import seaborn as sns


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
		) -> pd.DataFrame:
			"""Parse all PDF lab results in a folder into a single table.

			The returned DataFrame:
			- Rows: test names
			- Columns: dates (dd.mm.yyyy)
			- Values: test result values as strings
			"""

			if not os.path.isdir(folder_path):
				raise ValueError(f"Folder does not exist: {folder_path}")

			pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
			total_files = len(pdf_files)

			merged_data: Dict[str, Dict[str, str]] = {}

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

					if not test_name:
						continue

					if test_name not in merged_data:
						merged_data[test_name] = {}

					merged_data[test_name][date] = test_value

			if not merged_data:
				return pd.DataFrame()

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

			return df
	folder_path = input("Enter full path to folder with lab results PDFs: ").strip().strip('"')

	if not folder_path:
		print("No folder path provided. Exiting.")
		return

	if not os.path.isdir(folder_path):
		print(f"Folder does not exist: {folder_path}")
		return

	print(f"\nParsing PDFs in: {folder_path}\n")

	try:
		df = parse_lab_results_folder(folder_path)
	except Exception as e:
		print(f"Error while parsing PDFs: {e}")
		return

	if df.empty:
		print("No lab results could be extracted from the PDFs in this folder.")
		return

	print("Parsed table (first 10 rows):")
	print(df.head(10))

	default_csv = os.path.join(folder_path, "merged_lab_results.csv")
	try:
		df.to_csv(default_csv, index=True)
		print(f"\nTable saved to: {default_csv}")
	except Exception as e:
		print(f"Could not save CSV to {default_csv}: {e}")


if __name__ == "__main__":
	def plot_test_timeseries(df: pd.DataFrame, test_name: str) -> None:
		"""Plot a single test over time using Matplotlib inside Streamlit."""

		if df.empty or test_name not in df.index:
			st.warning("No data available for this test.")
			return

		series = df.loc[test_name]
		series = series.dropna()

		if series.empty:
			st.warning("No numeric values available for this test.")
			return

		try:
			values = pd.to_numeric(series.values, errors="coerce")
		except Exception:
			values = pd.Series([None] * len(series), index=series.index)

		mask = ~pd.isna(values)
		if mask.sum() == 0:
			st.warning("Could not convert results to numbers for plotting.")
			return

		dates = pd.to_datetime(series.index[mask], format="%d.%m.%Y", errors="coerce")
		values = values[mask]

		mask_valid = ~pd.isna(dates)
		if mask_valid.sum() == 0:
			st.warning("Could not parse dates for plotting.")
			return

		dates = dates[mask_valid]
		values = values[mask_valid]

		fig, ax = plt.subplots(figsize=(8, 4))
		ax.plot(dates, values, marker="o")
		ax.set_title(test_name)
		ax.set_xlabel("Date")
		ax.set_ylabel("Result")
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
	}


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
			default_folder = os.getcwd()
			folder_path = st.text_input(
				"Folder with PDF files",
				value=default_folder,
				help="Enter a folder path on this computer that contains your .pdf lab results.",
			)
			parse_button = st.button("Parse PDFs")

		if "parsed_df" not in st.session_state:
			st.session_state["parsed_df"] = None
		if "parsed_folder" not in st.session_state:
			st.session_state["parsed_folder"] = None

		if parse_button:
			if not folder_path or not os.path.isdir(folder_path):
				st.error("Please provide a valid existing folder path.")
			else:
				with st.spinner("Parsing PDFs..."):
					progress_bar = st.progress(0)
					status_text = st.empty()

					def progress_cb(current: int, total: int, filename: str) -> None:
						if total > 0:
							percent = int(((current + 1) / total) * 100)
							progress_bar.progress(percent)
						status_text.text(f"Processing {current + 1}/{total} – {filename}")

					try:
						parsed_df = parse_lab_results_folder(folder_path, progress_callback=progress_cb)
						st.session_state["parsed_df"] = parsed_df
						st.session_state["parsed_folder"] = folder_path
					except Exception as e:
						st.session_state["parsed_df"] = None
						st.error(f"Error while parsing PDFs: {e}")

		parsed_df: Optional[pd.DataFrame] = st.session_state.get("parsed_df")
		parsed_folder: Optional[str] = st.session_state.get("parsed_folder")

		if parsed_df is None:
			st.info("Enter a folder path and click 'Parse PDFs' to get started.")
			return

		if parsed_df.empty:
			st.warning("No lab results could be extracted from the PDFs in this folder.")
			return

		st.subheader("Merged Lab Results Table")

		# Add acronym explanations as a column next to the test name
		parsed_df_with_meaning = parsed_df.copy()
		meanings = []
		for test_name in parsed_df_with_meaning.index:
			meaning = get_test_description(str(test_name)) or ""
			meanings.append(meaning)
		parsed_df_with_meaning.insert(0, "Meaning", meanings)

		st.dataframe(parsed_df_with_meaning)

		with st.expander("Test abbreviations (what each means)"):
			abbr_df = pd.DataFrame(
				[(k, v) for k, v in sorted(TEST_ABBREVIATIONS.items())],
				columns=["Abbreviation", "Meaning"],
			)
			st.table(abbr_df)

		if parsed_folder:
			csv_path = os.path.join(parsed_folder, "merged_lab_results.csv")
			try:
				parsed_df.to_csv(csv_path, index=True)
				st.caption(f"Table also saved to: {csv_path}")
			except Exception:
				pass

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

		st.subheader("Cross-correlation between tests")
		numeric_df = parsed_df.apply(pd.to_numeric, errors="coerce")
		corr_df = numeric_df.T.corr(min_periods=2)
		corr_df = corr_df.dropna(how="all").dropna(axis=1, how="all")

		if corr_df.empty:
			st.info("Not enough numeric data to compute correlations between tests.")
		else:
			st.write("Correlation matrix (Pearson):")
			st.dataframe(corr_df)

			fig, ax = plt.subplots(figsize=(6, 5))
			sns.heatmap(corr_df, ax=ax, cmap="coolwarm", center=0, square=True)
			ax.set_title("Test cross-correlation heatmap")
			plt.tight_layout()
			st.pyplot(fig)


	if __name__ == "__main__":
		main()

