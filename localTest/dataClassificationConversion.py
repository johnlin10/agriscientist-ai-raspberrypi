# 此程式用於將現有的單一的Document數據結構，轉換為以週為單位的多個Documents
# 避免單一Document數據滿載的情況
# 目前狀態：數據已完成遷移

import datetime

# Firebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("../serviceAccountKey.json")
firebase_admin.initialize_app(
    cred,
    {"storageBucket": "agriscientist-ai.appspot.com"},
)
db = firestore.client()


# 獲取當前週的第一天（星期一）的日期
def get_week_start_date_from_timestamp(timestamp):
    week_start = timestamp - datetime.timedelta(days=timestamp.weekday())
    return week_start


# 根據給定的日期獲取或創建一個新的document印用
def get_or_create_weekly_document(db, date):
    # 格式化日期作為document名
    doc_name = date.strftime("%Y%m%d")
    # 獲取或創建對應的document引用
    weekly_ref = db.collection("sensors_data").document(doc_name)
    return weekly_ref


# 遷移數據到以週為單位的新documents中
def migrate_data_to_weekly_documents(db):
    documents_to_migrate = ["sensors", "sensors_2"]
    # 每週的數據
    weekly_data = {}
    # 循環目前已有的sensors與sensors_2數據合集
    for doc_name in documents_to_migrate:
        old_ref = db.collection("sensors_data").document(doc_name)
        doc = old_ref.get()
        if doc.exists:
            # 獲取 old document 中的數據
            sensors_data_list = doc.to_dict().get("data", [])
            # 循環每筆數據，並根據每筆數據的 timestramp 進行週分類
            for data in sensors_data_list:
                timestamp = data["timestamp"].date()
                week_start_date = get_week_start_date_from_timestamp(timestamp)
                doc_name = week_start_date.strftime("%Y%m%d")
                if doc_name not in weekly_data:
                    weekly_data[doc_name] = []
                weekly_data[doc_name].append(data)

    # 將分類好的每週數據，批量寫入Firestore，document以每週第一天命名
    batch = db.batch()
    for week, data_list in weekly_data.items():
        week_ref = db.collection("sensors_data").document(week)
        batch.set(week_ref, {"data": data_list})

    # 提交批量寫入
    batch.commit()


# 運行程式
migrate_data_to_weekly_documents(db)
