import pandas as pd

file_path = "dataset/news.csv"

print("=" * 60)
print("        FAKE NEWS DETECTOR - DATASET CHECK")
print("=" * 60)

try:
    df = pd.read_csv(file_path)

    print("\n✅ Dataset loaded successfully!")

    print("\nNumber of rows:", len(df))
    print("Number of columns:", len(df.columns))

    print("\n📌 COLUMN NAMES:")
    for column in df.columns:
        print(" -", column)

    print("\n📌 FIRST 5 ROWS:")
    print(df.head())

    print("\n📌 MISSING VALUES:")
    print(df.isnull().sum())

    print("\n📌 DATA TYPES:")
    print(df.dtypes)

    print("\n" + "=" * 60)
    print("             DATASET CHECK COMPLETED")
    print("=" * 60)

except Exception as e:
    print("\n❌ ERROR:")
    print(e)