import openpyxl
import pandas as pd

FILENAME = r"C:\Users\ethangaogao_i\Downloads\专车实验_补充列_未来30天_含星期几.xlsx"

# Quick row count
wb = openpyxl.load_workbook(FILENAME, read_only=True)
ws = wb['Sheet1']
count = 0
for _ in ws.iter_rows(min_row=2, values_only=True):
    count += 1
wb.close()
print(f"Total rows (excl header): {count}")

# Full analysis
df = pd.read_excel(FILENAME)
time_col = df.columns[0]
car_col = df.columns[1]
city_col = df.columns[2]
df['_seq'] = df[car_col].astype(str) + ' / ' + df[city_col].astype(str)
seq_counts = df.groupby('_seq')[time_col].nunique()
print(f"Unique sequences: {len(seq_counts)}")
print(f"Min seq length: {seq_counts.min()}")
print(f"Max seq length: {seq_counts.max()}")
print(f"Mean seq length: {seq_counts.mean():.0f}")
print(f"Date range: {df[time_col].min()} ~ {df[time_col].max()}")
print(f"Unique dates: {df[time_col].nunique()}")
print(f"Unique cities: {df[city_col].nunique()}")
print(f"Unique car types: {df[car_col].nunique()}")
print()
print("Sample sequences:")
for seq, length in seq_counts.head(15).items():
    print(f"  {seq}: {length} periods")
print("  ...")
for seq, length in seq_counts.tail(5).items():
    print(f"  {seq}: {length} periods")
