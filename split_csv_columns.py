import csv
import os


def split_csv_columns(input_file, output_dir="column_files"):
    """
    Split a CSV file into separate files, one for each column.
    Each output file contains the header on the first line,
    and all column values on the second line, separated by ', '.

    Parameters:
    input_file (str): Path to the input CSV file
    output_dir (str): Directory to store the output column files (default: "column_files")
    """
    os.makedirs(output_dir, exist_ok=True)

    # First pass: get headers
    with open(input_file, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        num_columns = len(headers)

    # Prepare clean filenames
    clean_headers = []
    for header in headers:
        clean = "".join(
            c for c in header if c.isalnum() or c in (" ", "-", "_")
        ).strip()
        if not clean:
            clean = f"column_{len(clean_headers)}"
        clean_headers.append(clean)

    # Initialize lists to collect column data (only if file is not too huge!)
    # WARNING: This loads all data into memory.
    columns = [[] for _ in range(num_columns)]

    # Second pass: read all data and split into columns
    with open(input_file, "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            for i in range(num_columns):
                if i < len(row):
                    columns[i].append(row[i])
                else:
                    columns[i].append("")  # handle missing values

    # Write each column to its own file with ', ' delimiter
    for i, header in enumerate(headers):
        filename = os.path.join(output_dir, f"{clean_headers[i]}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            # f.write(f"{header}\n")
            f.write(", ".join(columns[i]) + "\n")

    print(
        f"Successfully split {input_file} into {num_columns} column files in '{output_dir}' directory."
    )


if __name__ == "__main__":
    split_csv_columns((input("name of the file: ")))
