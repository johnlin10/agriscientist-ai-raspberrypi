from datetime import datetime, timedelta, timezone
from scipy.stats import linregress

import statistics

# Firebase
# import firebase_admin
# from firebase_admin import credentials, initialize_app
# from firebase_admin import firestore
# from firebase_admin import storage

# cred = credentials.Certificate("./serviceAccountKey.json")
# firebase_admin.initialize_app(
#     cred,
#     {"storageBucket": "agriscientist-ai.appspot.com"},
# )
# db = firestore.client()

sensor_data_list = []  # 感測器數據列表


# 獲取感測器數據
def get_data(db):
    global sensor_data_list

    sensors_data_ref = db.collection("sensors_data")
    documents = sensors_data_ref.list_documents()

    all_sensor_data = []
    for doc_ref in documents:
        doc_data = doc_ref.get().to_dict()
        for data in doc_data["data"]:
            data["humidity"] = float(data.get("humidity") or 0)
            data["soilHumidity"] = float(data.get("soilHumidity") or 0)
            data["temperature"] = float(data.get("temperature") or 0)
            data["light"] = float(data.get("light") or 0)
            data["water"] = float(data.get("water") or 0)
            data["timestamp"] = data["timestamp"]
            all_sensor_data.append(data)

    sensor_data_list = all_sensor_data
    return all_sensor_data


# 定義一個函數來計算給定時間範圍的統計資訊
def calculate_stats(time_range):
    # 過濾出在time_range範圍內的數據
    filtered_data = [
        data
        for data in sensor_data_list
        if data["timestamp"] > datetime.now(timezone.utc) - time_range
    ]
    # 計算平均值、中位數、最大值、最小值
    stats = {}
    for key in ["humidity", "light", "soilHumidity", "temperature", "water"]:
        values = [data[key] for data in filtered_data if key in data]
        if values:
            stats[key] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
            }
        else:
            stats[key] = {
                "mean": None,
                "median": None,
            }

    return stats


def dataAnalysistoText(db):
    get_data(db)  # get data from

    stats_hour = calculate_stats(timedelta(hours=1))
    stats_hour = calculate_stats(timedelta(hours=1))
    stats_three_hours = calculate_stats(timedelta(hours=3))
    stats_six_hours = calculate_stats(timedelta(hours=6))
    stats_twelve_hours = calculate_stats(timedelta(hours=12))
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
        f'【空氣溫度 三時】平均: {str(stats_three_hours["temperature"]["mean"])},中位: {str(stats_three_hours["temperature"]["mean"])},',
        f'【空氣溫度 六時】平均: {str(stats_six_hours["temperature"]["mean"])},中位: {str(stats_six_hours["temperature"]["median"])};',
        f'【空氣溫度 十二時】平均: {str(stats_twelve_hours["temperature"]["mean"])},中位: {str(stats_twelve_hours["temperature"]["median"])};'
        f'【空氣溫度 一日】平均: {str(stats_day["temperature"]["mean"])},中位:{str(stats_day["temperature"]["median"])};',
        f'【空氣溫度 三日】平均: {str(stats_three_days["temperature"]["mean"])},中位: {str(stats_three_days["temperature"]["median"])};',
        f'【空氣溫度 一週】平均: {str(stats_week["temperature"]["mean"])},中位: {str(stats_week["temperature"]["median"])};',
        f'【空氣溫度 三週】平均: {str(stats_three_week["temperature"]["mean"])},中位: {str(stats_three_week["temperature"]["median"])};',
        f'【空氣溫度 一個月】平均: {str(stats_month["temperature"]["mean"])},中位: {str(stats_month["temperature"]["median"])};',
        f'【空氣溫度 三個月】平均: {str(stats_three_month["temperature"]["mean"])},中位: {str(stats_three_month["temperature"]["median"])};',
        f'【空氣溫度 一年】平均: {str(stats_year["temperature"]["mean"])},中位: {str(stats_year["temperature"]["median"])};;',
        # 空氣濕度
        f'【空氣濕度 一時】平均: {str(stats_hour["humidity"]["mean"])},中位:{str(stats_hour["humidity"]["median"])};',
        f'【空氣濕度 三時】平均: {str(stats_three_hours["humidity"]["mean"])},中位: {str(stats_three_hours["humidity"]["mean"])},',
        f'【空氣濕度 六時】平均: {str(stats_six_hours["humidity"]["mean"])},中位: {str(stats_six_hours["humidity"]["median"])};',
        f'【空氣濕度 十二時】平均: {str(stats_twelve_hours["humidity"]["mean"])},中位: {str(stats_twelve_hours["humidity"]["median"])};'
        f'【空氣濕度 一日】平均: {str(stats_day["humidity"]["mean"])},中位:{str(stats_day["humidity"]["median"])};',
        f'【空氣濕度 三日】平均: {str(stats_three_days["humidity"]["mean"])},中位: {str(stats_three_days["humidity"]["median"])};',
        f'【空氣濕度 一週】平均: {str(stats_week["humidity"]["mean"])},中位: {str(stats_week["humidity"]["median"])};',
        f'【空氣濕度 三週】平均: {str(stats_three_week["humidity"]["mean"])},中位: {str(stats_three_week["humidity"]["median"])};',
        f'【空氣濕度 一個月】平均: {str(stats_month["humidity"]["mean"])},中位: {str(stats_month["humidity"]["median"])};',
        f'【空氣濕度 三個月】平均: {str(stats_three_month["humidity"]["mean"])},中位: {str(stats_three_month["humidity"]["median"])};',
        f'【空氣濕度 一年】平均: {str(stats_year["humidity"]["mean"])},中位: {str(stats_year["humidity"]["median"])};;',
        # 土壤濕度
        f'【土壤濕度 一時】平均: {str(stats_hour["soilHumidity"]["mean"])},中位:{str(stats_hour["soilHumidity"]["median"])};',
        f'【土壤濕度 三時】平均: {str(stats_three_hours["soilHumidity"]["mean"])},中位: {str(stats_three_hours["soilHumidity"]["mean"])},',
        f'【土壤濕度 六時】平均: {str(stats_six_hours["soilHumidity"]["mean"])},中位: {str(stats_six_hours["soilHumidity"]["median"])};',
        f'【土壤濕度 十二時】平均: {str(stats_twelve_hours["soilHumidity"]["mean"])},中位: {str(stats_twelve_hours["soilHumidity"]["median"])};'
        f'【土壤濕度 一日】平均: {str(stats_day["soilHumidity"]["mean"])},中位:{str(stats_day["soilHumidity"]["median"])};',
        f'【土壤濕度 三日】平均: {str(stats_three_days["soilHumidity"]["mean"])},中位: {str(stats_three_days["soilHumidity"]["median"])};',
        f'【土壤濕度 一週】平均: {str(stats_week["soilHumidity"]["mean"])},中位: {str(stats_week["soilHumidity"]["median"])};',
        f'【土壤濕度 三週】平均: {str(stats_three_week["soilHumidity"]["mean"])},中位: {str(stats_three_week["soilHumidity"]["median"])};',
        f'【土壤濕度 一個月】平均: {str(stats_month["soilHumidity"]["mean"])},中位: {str(stats_month["soilHumidity"]["median"])};',
        f'【土壤濕度 三個月】平均: {str(stats_three_month["soilHumidity"]["mean"])},中位: {str(stats_three_month["soilHumidity"]["median"])};',
        f'【土壤濕度 一年】平均: {str(stats_year["soilHumidity"]["mean"])},中位: {str(stats_year["soilHumidity"]["median"])};;',
        # 光照度
        f'【光照度 一時】平均: {str(stats_hour["light"]["mean"])},中位:{str(stats_hour["light"]["median"])};',
        f'【光照度 三時】平均: {str(stats_three_hours["light"]["mean"])},中位: {str(stats_three_hours["light"]["mean"])},',
        f'【光照度 六時】平均: {str(stats_six_hours["light"]["mean"])},中位: {str(stats_six_hours["light"]["median"])};',
        f'【光照度 十二時】平均: {str(stats_twelve_hours["light"]["mean"])},中位: {str(stats_twelve_hours["light"]["median"])};'
        f'【光照度 一日】平均: {str(stats_day["light"]["mean"])},中位:{str(stats_day["light"]["median"])};',
        f'【光照度 三日】平均: {str(stats_three_days["light"]["mean"])},中位: {str(stats_three_days["light"]["median"])};',
        f'【光照度 一週】平均: {str(stats_week["light"]["mean"])},中位: {str(stats_week["light"]["median"])};',
        f'【光照度 三週】平均: {str(stats_three_week["light"]["mean"])},中位: {str(stats_three_week["light"]["median"])};',
        f'【光照度 一個月】平均: {str(stats_month["light"]["mean"])},中位: {str(stats_month["light"]["median"])};',
        f'【光照度 三個月】平均: {str(stats_three_month["light"]["mean"])},中位: {str(stats_three_month["light"]["median"])};',
        f'【光照度 一年】平均: {str(stats_year["light"]["mean"])},中位: {str(stats_year["light"]["median"])};;',
        # 儲水量
        f'【儲水量 一時】平均: {str(stats_hour["water"]["mean"])},中位:{str(stats_hour["water"]["median"])};',
        f'【儲水量 三時】平均: {str(stats_three_hours["water"]["mean"])},中位: {str(stats_three_hours["water"]["mean"])},',
        f'【儲水量 六時】平均: {str(stats_six_hours["water"]["mean"])},中位: {str(stats_six_hours["water"]["median"])};',
        f'【儲水量 十二時】平均: {str(stats_twelve_hours["water"]["mean"])},中位: {str(stats_twelve_hours["water"]["median"])};'
        f'【儲水量 一日】平均: {str(stats_day["water"]["mean"])},中位:{str(stats_day["water"]["median"])};',
        f'【儲水量 三日】平均: {str(stats_three_days["water"]["mean"])},中位: {str(stats_three_days["water"]["median"])};',
        f'【儲水量 一週】平均: {str(stats_week["water"]["mean"])},中位: {str(stats_week["water"]["median"])};',
        f'【儲水量 三週】平均: {str(stats_three_week["water"]["mean"])},中位: {str(stats_three_week["water"]["median"])};',
        f'【儲水量 一個月】平均: {str(stats_month["water"]["mean"])},中位: {str(stats_month["water"]["median"])};',
        f'【儲水量 三個月】平均: {str(stats_three_month["water"]["mean"])},中位: {str(stats_three_month["water"]["median"])};',
        f'【儲水量 一年】平均: {str(stats_year["water"]["mean"])},中位: {str(stats_year["water"]["median"])};',
    )

    return analysis_text


# 趨勢分析
def analyze_trend(data_points):
    timestamps = [point["timestamp"] for point in data_points]
    values = [point["value"] for point in data_points]
    slope, intercept, r_value, p_value, std_err = linregress(timestamps, values)
    return slope, intercept, r_value, p_value, std_err


# 過濾指定時間範圍內數據
def filter_data_by_time(sensor_data, time_range):
    return [
        data
        for data in sensor_data
        if data["timestamp"].timestamp()
        > datetime.now(timezone.utc).timestamp() - time_range.total_seconds()
    ]


# 生成趨勢文本
def data_analysis_to_text(sensor_data):
    analysis_texts = []
    # 感測器參數及對應名稱
    sensortype_mapping = {
        "temperature": "空氣溫度",
        "humidity": "空氣濕度",
        "soilHumidity": "土壤濕度",
        "water": "儲水量",
    }
    # 欲分析的時間範圍
    time_ranges = {
        "一小時": timedelta(hours=1),
        "三小時": timedelta(hours=3),
        "六小時": timedelta(hours=6),
        "十二小時": timedelta(hours=6),
        "一天": timedelta(days=1),
        "三天": timedelta(days=3),
        "一週": timedelta(weeks=1),
        "一個月": timedelta(days=30),
        "三個月": timedelta(days=90),
    }

    # 循環分析設定的時間段
    for time_range_name, time_range in time_ranges.items():
        # 過濾指定時間段的數據
        filtered_data = filter_data_by_time(sensor_data, time_range)
        # 循環每個感測器的數據
        for sensor_type in ["temperature", "humidity", "soilHumidity", "water"]:
            # 將綜合數據過濾為單個感測器數據及Timestamp
            sensor_data_points = [
                {"timestamp": data["timestamp"].timestamp(), "value": data[sensor_type]}
                for data in filtered_data
                if sensor_type in data
            ]
            if sensor_data_points:
                slope, intercept, r_value, p_value, std_err = analyze_trend(
                    sensor_data_points
                )
                # 判斷趨勢的方向
                trend_direction = "上升" if slope > 0 else "下降"
                # 加入趨勢分析的文本
                analysis_texts.append(
                    f"最近{time_range_name}的“{sensortype_mapping[sensor_type]}”呈現「{trend_direction}」趨勢。"
                )

    return analysis_texts


# 資料趨勢分析
def dataTrendText(db):
    sensor_data = get_data(db)
    analysis_texts = data_analysis_to_text(sensor_data)
    all_analysis_text = ""
    for text in analysis_texts:
        all_analysis_text += text + "\n"

    return all_analysis_text
