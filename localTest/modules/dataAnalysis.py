from datetime import datetime, timedelta, timezone

import statistics


# Firebase
import firebase_admin
from firebase_admin import credentials, initialize_app
from firebase_admin import firestore
from firebase_admin import storage

cred = credentials.Certificate("../serviceAccountKey.json")
firebase_admin.initialize_app(
    cred,
    {"storageBucket": "agriscientist-ai.appspot.com"},
)
db = firestore.client()


# 將字符串轉換為浮點數並將時間戳記轉換為datetime對象


def get_data():
    global sensors2_data_ref, temperature_data_ref, sensor_data, sensor_data_list, db
    sensors2_data_ref = db.collection("sensors_data").document("sensors_2")  # 綜合數據
    sensor_data = sensors2_data_ref.get().to_dict()
    sensor_data_list = sensor_data["data"]

    for data in sensor_data_list:
        data["humidity"] = float(data["humidity"])
        data["soilHumidity"] = float(data["soilHumidity"])
        data["temperature"] = float(data["temperature"])


# 定義一個函數來計算給定時間範圍的統計資訊
def calculate_stats(time_range):
    # 過濾出在time_range範圍內的數據
    filtered_data = [
        data
        for data in sensor_data_list
        if data["timestamp"] > datetime.now(timezone.utc) - time_range
    ]

    # 計算統計信息
    stats = {}
    for key in ["humidity", "light", "soilHumidity", "temperature", "water"]:
        values = [data[key] for data in filtered_data if key in data]
        if values:  # 檢查列表是否不為空
            stats[key] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
            }
        else:
            stats[key] = {
                "mean": None,  # 或者你可以設置為0或其他合適的預設值
                "median": None,  # 或者你可以設置為0或其他合適的預設值
            }
    return stats


def dataAnalysistoText(db):
    get_data(db)

    stats_hour = calculate_stats(timedelta(hours=1))
    stats_day = calculate_stats(timedelta(days=1))
    stats_three_days = calculate_stats(timedelta(days=3))
    stats_week = calculate_stats(timedelta(weeks=1))
    stats_three_week = calculate_stats(timedelta(weeks=3))
    stats_month = calculate_stats(timedelta(days=30))
    stats_three_month = calculate_stats(timedelta(days=90))
    stats_year = calculate_stats(timedelta(days=365))

    analysis_text = (
        # 數據 PrePrompt
        "當使用者詢問農場情況，需要數據時，請將數據盡可能簡單的表達。如果使用者未明確特定時段，請以最近的數據分析時段作為資料。",
        # 時間
        f"目前時間為：utc+8 {datetime.now()}。",
        # 空氣溫度
        f'【空氣溫度 一時】平均: {str(stats_hour["temperature"]["mean"])},中位:{str(stats_hour["temperature"]["median"])};',
        f'【空氣溫度 一日】平均: {str(stats_day["temperature"]["mean"])},中位:{str(stats_day["temperature"]["median"])};',
        f'【空氣溫度 三日】平均: {str(stats_three_days["temperature"]["mean"])},中位: {str(stats_three_days["temperature"]["median"])};',
        f'【空氣溫度 一週】平均: {str(stats_week["temperature"]["mean"])},中位: {str(stats_week["temperature"]["median"])};',
        f'【空氣溫度 三週】平均: {str(stats_three_week["temperature"]["mean"])},中位: {str(stats_three_week["temperature"]["median"])};',
        f'【空氣溫度 一個月】平均: {str(stats_month["temperature"]["mean"])},中位: {str(stats_month["temperature"]["median"])};',
        f'【空氣溫度 三個月】平均: {str(stats_three_month["temperature"]["mean"])},中位: {str(stats_three_month["temperature"]["median"])};',
        f'【空氣溫度 一年】平均: {str(stats_year["temperature"]["mean"])},中位: {str(stats_year["temperature"]["median"])};;',
        # 空氣濕度
        f'【空氣濕度 一時】平均: {str(stats_hour["humidity"]["mean"])},中位:{str(stats_hour["humidity"]["median"])};',
        f'【空氣濕度 一日】平均: {str(stats_day["humidity"]["mean"])},中位:{str(stats_day["humidity"]["median"])};',
        f'【空氣濕度 三日】平均: {str(stats_three_days["humidity"]["mean"])},中位: {str(stats_three_days["humidity"]["median"])};',
        f'【空氣濕度 一週】平均: {str(stats_week["humidity"]["mean"])},中位: {str(stats_week["humidity"]["median"])};',
        f'【空氣濕度 三週】平均: {str(stats_three_week["humidity"]["mean"])},中位: {str(stats_three_week["humidity"]["median"])};',
        f'【空氣濕度 一個月】平均: {str(stats_month["humidity"]["mean"])},中位: {str(stats_month["humidity"]["median"])};',
        f'【空氣濕度 三個月】平均: {str(stats_three_month["humidity"]["mean"])},中位: {str(stats_three_month["humidity"]["median"])};',
        f'【空氣濕度 一年】平均: {str(stats_year["humidity"]["mean"])},中位: {str(stats_year["humidity"]["median"])};;',
        # 土壤濕度
        f'【土壤濕度 一時】平均: {str(stats_hour["soilHumidity"]["mean"])},中位:{str(stats_hour["soilHumidity"]["median"])};',
        f'【土壤濕度 一日】平均: {str(stats_day["soilHumidity"]["mean"])},中位:{str(stats_day["soilHumidity"]["median"])};',
        f'【土壤濕度 三日】平均: {str(stats_three_days["soilHumidity"]["mean"])},中位: {str(stats_three_days["soilHumidity"]["median"])};',
        f'【土壤濕度 一週】平均: {str(stats_week["soilHumidity"]["mean"])},中位: {str(stats_week["soilHumidity"]["median"])};',
        f'【土壤濕度 三週】平均: {str(stats_three_week["soilHumidity"]["mean"])},中位: {str(stats_three_week["soilHumidity"]["median"])};',
        f'【土壤濕度 一個月】平均: {str(stats_month["soilHumidity"]["mean"])},中位: {str(stats_month["soilHumidity"]["median"])};',
        f'【土壤濕度 三個月】平均: {str(stats_three_month["soilHumidity"]["mean"])},中位: {str(stats_three_month["soilHumidity"]["median"])};',
        f'【土壤濕度 一年】平均: {str(stats_year["soilHumidity"]["mean"])},中位: {str(stats_year["soilHumidity"]["median"])};;',
        # 光照度
        f'【光照度 一時】平均: {str(stats_hour["light"]["mean"])},中位:{str(stats_hour["light"]["median"])};',
        f'【光照度 一日】平均: {str(stats_day["light"]["mean"])},中位:{str(stats_day["light"]["median"])};',
        f'【光照度 三日】平均: {str(stats_three_days["light"]["mean"])},中位: {str(stats_three_days["light"]["median"])};',
        f'【光照度 一週】平均: {str(stats_week["light"]["mean"])},中位: {str(stats_week["light"]["median"])};',
        f'【光照度 三週】平均: {str(stats_three_week["light"]["mean"])},中位: {str(stats_three_week["light"]["median"])};',
        f'【光照度 一個月】平均: {str(stats_month["light"]["mean"])},中位: {str(stats_month["light"]["median"])};',
        f'【光照度 三個月】平均: {str(stats_three_month["light"]["mean"])},中位: {str(stats_three_month["light"]["median"])};',
        f'【光照度 一年】平均: {str(stats_year["light"]["mean"])},中位: {str(stats_year["light"]["median"])};;',
        # 儲水量
        f'【儲水量 一時】平均: {str(stats_hour["water"]["mean"])},中位:{str(stats_hour["water"]["median"])};',
        f'【儲水量 一日】平均: {str(stats_day["water"]["mean"])},中位:{str(stats_day["water"]["median"])};',
        f'【儲水量 三日】平均: {str(stats_three_days["water"]["mean"])},中位: {str(stats_three_days["water"]["median"])};',
        f'【儲水量 一週】平均: {str(stats_week["water"]["mean"])},中位: {str(stats_week["water"]["median"])};',
        f'【儲水量 三週】平均: {str(stats_three_week["water"]["mean"])},中位: {str(stats_three_week["water"]["median"])};',
        f'【儲水量 一個月】平均: {str(stats_month["water"]["mean"])},中位: {str(stats_month["water"]["median"])};',
        f'【儲水量 三個月】平均: {str(stats_three_month["water"]["mean"])},中位: {str(stats_three_month["water"]["median"])};',
        f'【儲水量 一年】平均: {str(stats_year["water"]["mean"])},中位: {str(stats_year["water"]["median"])};',
    )

    return analysis_text
