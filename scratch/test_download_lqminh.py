import os
import s3fs
import pyarrow.dataset as ds
import pyarrow as pa
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
ENDPOINT_URL = os.getenv('ENDPOINT_URL')

fs = s3fs.S3FileSystem(
    key=ACCESS_KEY,
    secret=SECRET_KEY,
    client_kwargs={"endpoint_url": ENDPOINT_URL}
)

tables = [
    'churn_customers',
    'churn_orders',
    'churn_payments',
    'churn_subscriptions',
    'churn_support_tickets',
    'churn_marketing_interactions',
    'churn_product_usage'
]

os.makedirs('data', exist_ok=True)

for folder in tables:
    print(f"\n{'-'*50}")
    print(f"⏳ Đang tải từ MinIO: lqminh/silver/devdb/{folder}")
    drive_save_path = f'data/{folder}.csv'
    
    try:
        dataset_path = f"lqminh/silver/devdb/{folder}/"
        dataset = ds.dataset(dataset_path, filesystem=fs, format="parquet")
        table = dataset.to_table()
        
        # Cast timestamp/date to string if needed to avoid pandas/isoformat errors
        new_schema_fields = []
        for field in table.schema:
            if pa.types.is_timestamp(field.type) or pa.types.is_date(field.type):
                new_schema_fields.append(pa.field(field.name, pa.string()))
            else:
                new_schema_fields.append(field)
        
        table_safe = table.cast(pa.schema(new_schema_fields))
        df = table_safe.to_pandas()
        
        # Drop partition columns if added by pyarrow dataset (year, month) or keep them clean
        for pcol in ['year', 'month']:
            if pcol in df.columns and pcol not in ['signup_year', 'signup_month']:
                df = df.drop(columns=[pcol])
                
        df.to_csv(drive_save_path, index=False)
        print(f"✅ Thành công! Đã tải {len(df):,} dòng, {len(df.columns)} cột -> {drive_save_path}")
        print(f"   Khách hàng unique: {df['customer_id'].nunique() if 'customer_id' in df.columns else 'N/A'}")
        
    except Exception as e:
        print(f"❌ Lỗi khi tải {folder}: {e}")

print(f"\n{'-'*50}")
print("🎉 HOÀN TẤT TẢI DỮ LIỆU TỪ lqminh/silver/devdb/")
