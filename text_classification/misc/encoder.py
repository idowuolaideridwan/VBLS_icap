import pandas as pd

def read_csv_file(file_path, encoding='utf-8'):
    try:
        return pd.read_csv(file_path, encoding=encoding)
    except UnicodeDecodeError as e:
        print(f"Unicode decode error: {e}")
        print(f"Trying with a different encoding...")
        return pd.read_csv(file_path, encoding='ISO-8859-1')  # Trying with ISO-8859-1 if UTF-8 fails

def save_csv_utf8(df, file_path):
    try:
        df.to_csv(file_path, encoding='utf-8', index=False)
        print("File saved successfully in UTF-8 format.")
    except Exception as e:
        print(f"Failed to save the CSV file due to: {e}")

# Main script
if __name__ == "__main__":
    file_path = 'data/dataset.csv'
    new_file_path = 'data/dataset_v2.csv'

    df = read_csv_file(file_path, encoding='utf-8')  # First try UTF-8
    if df is not None:
        print(df.head())
        save_csv_utf8(df, new_file_path)
