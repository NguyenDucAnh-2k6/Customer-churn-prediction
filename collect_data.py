import pandas as pd
import s3fs
import pyarrow.dataset as ds
import pyarrow as pa
import os
from deltalake import DeltaTable
from dotenv import load_dotenv

load_dotenv()

ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
ENDPOINT_URL = os.getenv('ENDPOINT_URL')

# Cấu hình cho DeltaTable
storage_options = {
    "AWS_ACCESS_KEY_ID": ACCESS_KEY,
    "AWS_SECRET_ACCESS_KEY": SECRET_KEY,
    "AWS_ENDPOINT_URL": ENDPOINT_URL,
    "AWS_REGION": "us-east-1",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
    "AWS_ALLOW_HTTP": "true" 
}

folders_to_download = [
    
    'customer_360'
]

os.makedirs('data', exist_ok=True)

# Khởi tạo kết nối cho PyArrow fallback
fs = s3fs.S3FileSystem(
    key=ACCESS_KEY,
    secret=SECRET_KEY,
    client_kwargs={"endpoint_url": ENDPOINT_URL}
)

for folder in folders_to_download:
    print(f"\n{'-'*50}")
    print(f"⏳ Đang xử lý thư mục: {folder}")
    drive_save_path = f'data/{folder}.csv'
    
    try:
        # LUỒNG 1: Đọc bằng DeltaTable (Ưu tiên)
        dt = DeltaTable(f"s3://test-doanh/silver/{folder}", storage_options=storage_options)
        df = dt.to_pandas()
        df.to_csv(drive_save_path, index=False)
        print(f"✅ Thành công (DeltaTable)! Đã tải {len(df)} dòng.")
        
    except Exception as e:
        error_msg = str(e).lower()
        
        # Kiểm tra xem lỗi có phải do thời gian dị thường không (như năm 20662)
        if "isoformat" in error_msg or "out of range" in error_msg or "year" in error_msg:
            print("⚠️ Phát hiện lỗi dị thường dữ liệu thời gian, đang tự động chuyển sang chế độ PyArrow thô...")
            try:
                # LUỒNG 2: Fallback sang PyArrow để ép kiểu Date thành String
                dataset_path = f"test-doanh/silver/{folder}/"
                dataset = ds.dataset(dataset_path, filesystem=fs, format="parquet")
                table = dataset.to_table()
                
                new_schema_fields = []
                for field in table.schema:
                    if pa.types.is_timestamp(field.type) or pa.types.is_date(field.type):
                        new_schema_fields.append(pa.field(field.name, pa.string()))
                    else:
                        new_schema_fields.append(field)
                
                table_safe = table.cast(pa.schema(new_schema_fields))
                df = table_safe.to_pandas()
                df.to_csv(drive_save_path, index=False)
                print(f"✅ Thành công (PyArrow Fallback)! Đã tải {len(df)} dòng.")
                
            except Exception as inner_e:
                print(f"❌ LỖI NGHIÊM TRỌNG khi tải {folder} bằng PyArrow: {inner_e}")
        else:
            # Nếu là một lỗi lạ khác (không phải thời gian), in ra để debug
            print(f"❌ LỖI khi tải {folder} bằng DeltaTable: {e}")

print(f"\n{'-'*50}")
print("🎉 HOÀN TẤT! Toàn bộ dữ liệu đã được xử lý xong.")