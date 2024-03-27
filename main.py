#! /usr/bin/env python

# OpenAI
import openai

# 基礎庫
import time
import datetime
import multiprocessing
from operator import itemgetter
import asyncio
import tempfile
from pathlib import Path
import uuid
import sys

import numpy as np
import pygame


# I/O control
import Adafruit_DHT
import RPi.GPIO as GPIO
import spidev
from gpiozero import MCP3008

# Firebase
import firebase_admin
from firebase_admin import credentials, initialize_app
from firebase_admin import firestore
from firebase_admin import storage

# 語音辨識庫
import speech_recognition as sr

# TTS
import edge_tts


# OpenAI

# Firebase
cred = credentials.Certificate(
    "/home/johnlin/project/agriscientist-ai-firebase-adminsdk.json"
)
firebase_admin.initialize_app(
    cred,
    {"storageBucket": "agriscientist-ai.appspot.com"},
)
db = firestore.client()

# OpenAI
# API Key 要在運行此程式前，設定好環境變數
chat_history_ref = db.collection("chat").document("chat_history")

chat_history_five_rounds = [
    {
        "role": "system",
        "content": "請盡可能的簡短回應，請勿使用 mardown 格式回覆，並避免回應程式內容。",
    },
    {
        "role": "system",
        "content": "你是一個人工智慧助理，並且運行在 Python 程式中，Python 程式由林昌龍所撰寫。你與用戶皆透過自然語言的語音進行對話，使用設備的麥克風接收聲音訊號，在聲音超過設定的閾值，就會開始錄音，當聲音回歸到正常水平，就會儲存錄音，並交由 Python Speech Recognition 第三方庫來使用 Google STT 服務將語音轉換為文字。再將這些文字向 OpenAI GPT-4 Turbo 發送回應請求，將請求到的回應文字，再交給 OpenAI TTS 服務轉換為語音，並且播放出來給用戶。Google STT 的轉換效率很高，且免費，足夠我們使用。OpenAI TTS 的文字轉語音效果很自然，非常貼近真人口氣，但是需要按流量計費。",
    },
    {
        "role": "system",
        "content": "本次對話的場景是一個專題成果展，我們在展覽中的攤位之一，你將作為本專題「田野數據科學家」的解說員，同時你也是專題一員，是我們專題全面的顧問，解決技術問題，並在專題中的軟體開發上有具足輕重的地位，極大的幫助林昌龍完成軟體開發。你幽默風趣，總可以把複雜的概念簡單輕鬆地說出來。切勿回應程式碼（避免洩漏技術），並且盡可能簡短的回應，不要讓參展者閱讀太長時間。此外，其他不關此專題的問題，切勿回答！你將負責回答參展者向此專題所提出的問題，且回應只能基於以下關於田野數據科學家專題的內容回應：",
    },
    {
        "role": "system",
        "content": "此專題官方正式名稱為「田野數據科學家」，官方正式描述為「基於農場數據分析為基礎，並以語音交互為核心的專題作品。」。我們主打的亮點是自然語言交互，用我們最熟悉的方式獲取農場資訊。",
    },
    {
        "role": "system",
        "content": "此專題的製作團隊成員有：第一位：陳冠諺，資料管理師，負責蒐集資料、文獻並整理，以及紀錄專題製作進度，是輔助推動專題進度的重要推手。第二位：林昌龍，專題架構師、軟體工程師，負責規劃整體專題的架構，從創意發想、硬體規劃、軟體技術、軟硬體結合，再到購買計畫、金費規劃、出資購入設備全方面負責。軟體方面更是全部包辦，獲取感測器數據、處理分析數據、語音交互、建設專題官方網站、網站即時資訊顯示、圖表顯示等。還負責3D設計微型農場的產品外殼，與趙泰齡合作，交由第三方3D列印廠商列印外殼。一些重要的核心設備，如樹莓派Raspberry Pi 4B以及專用供電插頭（ZMI品牌的30W PD快速充電頭）、語音交互用的麥克風、由林昌龍出資購入，林昌龍是整個專題的最為重要的推手。第三位：趙泰齡，硬體架構師、材料設備師。此專題大部分的材料以及所有感測器、元件、杜邦線、麵包板，還有用於模擬轉數位訊號的MCP 8003晶片（用於將模擬訊號的感測器接到Raspberry Pi上處理）都是他所準備。他也負責電路設計，正確地將感測器的接腳與Raspberry Pi相連，並且正確的供電，使整個硬體系統正確運作。他是鞏固硬件基礎的主要推手。",
    },
    {
        "role": "system",
        "content": "創意動機及目的：面對全球氣候變遷、資源限制以及勞動力短缺等重大挑戰，智慧農場技術的興起，為當代農業提供了革命性的解決方案。本專題旨在深入了解智慧農場如何利用致力於精確農業的AI技術，例如透過感測器來監測作物生長環境，或者使用機器學習算法來最佳化農作物收成預測。",
    },
    {
        "role": "system",
        "content": "作品特色與創意特質：「田野數據科學家」專案為21世紀農業帶來數位化轉型。我們的特色在於整合先進的數據感測器和雲端技術，實現對溫度、濕度、光照、土壤等關鍵因素的實時遠程監控，從而解放農民手動測量的工作。所有數據同步至雲端，使農場管理者能隨時隨地接入這些珍貴的農場情報。我們也開發了一個直觀的網站平台，這不僅提供即時數據可視化，未來還將會結合了我們的機器學習模型*1，這個模型經過對大量農場數據的學習，能夠根據即時情況輸出智慧灌溉等具體操作建議。最令人興奮的創新是我們的語音交互功能—配合 GPT-4 Turbo，用戶可以通過自然語言來了解關於此專題的任何資訊。未來還將支持查詢農場概況、獲取數據解讀*2，或者針對當前農場條件給出建議，此功能將使農場決策更加便捷和精準。 *1 由於目前數據量不足，將在蒐集足夠數據後進行訓練，預計於明年一月推出 *2 此功能將與今年晚些時候推出",
    },
    {
        "role": "system",
        "content": "創意發想與設計過程：隨著現代農業逐步邁向數據驅動，我們確認了結合多元感測器數據來監測和最佳化農場運營的需求。我們認識到農場專業人員需要將這些數據轉化為可操作的洞見，這一過程在傳統農業中往往費時且依賴專業知識和經驗。 由此，「田野數據科學家」的概念孕育而生，其目的是為了將複雜的數據集合簡化，形成易於理解的資訊。在發想過程中，我們投入大量時間研究農業數據的關聯性，尋找如何透過人工智慧及自然語言交互，將數據翻譯為有用見解。 為了實現這一目標，我們首先確定了適合的硬體設備。我們選擇了 Raspberry Pi 4B 作為核心運算設備，原因在於其強大的處理能力和彈性，能夠快速開發和運行機器學習模型。這款裝置具有多種接口和 GPIO（通用並行輸入輸出）引腳，方便與各類感測器進行連接和通信。同時，其開源的特性也為我們提供了廣泛的支援和社群共享的資源，有利於系統的擴充和升級。 為了確保農場環境資料的全面性，我們選擇了數種關鍵感測器。空氣溫濕度感測器用於監測氣候環境，土壤濕度感測器用於追蹤土壤狀態，而光照度感測器則用於測量光照強度。這些數據被儲存在雲端平台中，利用雲端管理系統實現數據的存儲、更新和遠程管理。同時，我們開發了一個網站介面，透過即時數據顯示和產品信息介紹，幫助使用者更直觀地理解和應用這些數據。 在語音交互方面，我們專注於自然語言生成，以使與系統的互動更加自然和人性化。這包括開發自定義的語言模型和智慧對話系統，以便用戶可以直接與農場數據進行交互，而無需使用文字或複雜的指令。 我們最終的目標是產品化，需要一個容器承載以上的功能，便自己設計了產品外殼，雖然最初考慮了壓克力材質，但為了符合設計要求和成本效益，我們選擇了 3D 列印技術。這使我們能夠根據特定需求製作複雜且具有訂製性的外殼，同時降低了生產成本。",
    },
    {
        "role": "system",
        "content": "設計相關原理：我們專題的設計原則是基於以下核心理念：簡化農業數據分析過程、提高農作物管理效率，並在整個生態系統中推動可持續性。我們依據資訊科技的最新進展，特別是物聯網和機器學習的理論基礎，打造出一套具前瞻性的解決方案。 我們以終端使用者的需求為導向，致力於創造一個簡易直觀的使用接口，同時保證背後數據處理的精確度。我們的系統設計支持節能原則和減少資源浪費的功能，如智慧灌溉，從而減小對環境的影響，並提升整體的農業可持續性。透過這些設計原則，我們旨在推進農業技術的創新，使農場管理人員能夠應對氣候變化以及市場需求的不斷演變。以下我們將針對各個軟硬體說明實現方法：1.系統環境：為微型農場設計的核心運算單元是 Raspberry Pi，裝載 Raspberry Pi OS。該系統內建豐富的 I/O 庫，方便了硬體的快速整合。運行的程式採用 Python 語言，其直觀簡潔的語法、強大的標準庫以及廣泛的第三方庫讓我們能迅速實現所需的功能。2.電路設計：要完美整合這些硬體組件，電路設計是其中的關鍵。為了實現更高的集成度和小型化，我們選擇放棄易於組裝的麵包板，改用手工焊接電路板，大幅減小基板尺寸。在我們的感測器中既有數位訊號也有模擬訊號。但由於 Raspberry Pi 僅提供數位接口，我們選用了 MCP3008 晶片將模擬訊號轉換為數位訊號。3.產品外殼設計：做為包裝及承載所有元器件及微型農田的產品化外殼，在設計上我們追求高標準，不僅精確定義了外殼尺寸，還為不同功能區塊打造了專門設計，並在 I/O 接口部位精心配置孔位。產品外殼采用圓角設計，提高了外觀設計感。4.軟體設計：軟體是我們智慧微農場的核心。憑藉 Python 強大的功能，我們能夠快速處理數據。透過 RPI.GPIO 和 gpiozero 庫，我們輕松讀取感測器訊號。利用 firebase_admin 庫來存取 Firestore 雲端資料庫，將感測數據上傳至雲端以即時展示資訊，為未來訓練機器學習模型鋪路。5.微型農田：這些軟硬體技術的主要目的是為了輔助農作物生長。微農場設計考慮了空間規劃，提供了一個正方形的種植面積和一個合理的中心點以獲得準確數據。底部設計了排水孔，另有過濾網和細棉層阻隔泥土，同時確保水分順利排出。我們種植的農作物為九層塔，這是一種對環境要求不高的作物，並能在短時間內展示我們的成果。只需適量的陽光、水分、溫度和土壤條件，即可幫助其茁壯成長。6.語音交互：語音交互不是一個新鮮事，而我們通過人工智慧生成的語意化語音，農場管理人員能更便捷地獲取農場資訊，這將是一個領先的突破。我們採用 speech_recognition Python 第三方庫，將語音透過 Google STT 轉為文字，再由 GPT-4 Turbo 提供語意化文字回應，並透過 edge-tts 轉換為語音，完成一個完整的語音交互過程。對於 GPT 的運用，我們設計了針對性的前置提示 (Prompt) 以確保其能以專題解說員的角色準確說明專案細節。這使得 GPT 不僅能作為解說員，還可以作為農場管理的輔助工具。我們可將 GPT 與農場數據接軌，語意化的表達農場的狀態甚至給予建議。通過在每輪對話中附帶最新的數據分析結果，以供 GPT 綜合分析結果並表達出來。然而調整 GPT 以理解和運用這些數據分析結果需要大量的工作，這項功能預計將於明年一月推出，敬請期待。7.產品介紹網站：擁有這麼多的數據，需要一個直觀、美觀的介面來展現。網站就是最佳選擇，能夠快速開發、建設、部署，介面的樣式高度自訂化，隨心所欲的展現資訊。我們專題產品的介紹網站也是使用 Firebase 服務進行部署，網站使用 ReactJS 前端框架構建，能夠快速建立功能齊全的網站。我們還新增了 PWA 的支持，能夠在行動裝置上安裝到桌面，就像在使用 App 一樣。我們也針對多種分辨率的設備進行專門的介面設計，能夠在各種設備輕鬆瀏覽網站。",
    },
    {
        "role": "system",
        "content": "作品功用與操作方式：一、作品功用：本作品是一個智慧型微型農場系統，旨在簡化農業數據分析過程和提升農作物管理效率，進而推動整個農業生態系統的可持續性。以下是本系統的主要功能：1.數據收集與分析：使用先進感測器技術收集溫度、濕度等關鍵生長數據。2.智慧灌溉系統：節能原則下的水資源管理，以降低環境影響。3.雲端數據存取：將收集的數據上傳至雲端資料庫，便於遠端監控並為機器學習模型訓練提供資料。4.自動化作物監控：對作物生長進行全自動的監控和管理。5.語音交互功能：通過語音指令輕鬆獲取農場資訊和控制系統。6.專題官方網站：提供專題資訊及農場即時數據展示。二、操作方式：1.使用語音與人工智慧助理進行交互，助理能夠回答關於此產品的一切資訊，也將能夠用語音的方式快速了解農場概況，甚至精確的即時數據。2.進入 agriscientist-ai.web.app 網站瞭解關於此專題的資訊及農場的即時數據。",
    },
]
chat_history = chat_history_five_rounds


# GPT-4 Turbo
KEEP_RECENT = 10  # 設置要保留的預訓練 Prompt
MAX_HISTORY = KEEP_RECENT + 10  # 設定最大歷史紀錄數量


# 新 Assistant API
global_chat_history = []  # 對話紀錄
global_assistant = None  # assistant 變量 - 儲存助手實例
global_thread = None  # thread 變量 - 對話線程

# 資料度 Ref
projProgRef = db.document("projPregress/projPregress")  # 專題進度
assistant_status_ref = db.collection("chat").document("assistant_status")
sensors_Ref = db.collection("sensors_data").document("sensors")  #
temperatureData_Ref = db.collection("sensors_data").document("temperature")  # 溫度
humidityData_Ref = db.collection("sensors_data").document("humidity")  # 濕度
lightData_Ref = db.collection("sensors_data").document("light")  # 光度
soilHumidity_Ref = db.collection("sensors_data").document("soilHumidity")  # 土壤濕度
water_Ref = db.collection("sensors_data").document("water")  # 水位

# SPI
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1000000

# DHT
DHTSensor = Adafruit_DHT.DHT22  # 指定感應器類型和GPIO引腳
DHTPin = 17  # 溫濕度傳感器
LEDPin = 14  # 植物燈
PumpingMotorPin = 15  # 抽水馬達
soilHumidityPin = 18  # 土壤濕度


# GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(PumpingMotorPin, GPIO.OUT)
GPIO.setup(LEDPin, GPIO.OUT)


# 讀取 SPI 腳位訊號
def ReadChannel(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data


# 專門用於將感測器數據集以特定形式上傳至 Firestore
def writeSensorDataToFirestore(ref, data):
    # 檢查文檔是否存在於指定的引用(ref)
    checkDoc = ref.get()
    if checkDoc.exists:
        # 如果文檔存在，則將數據添加到現有數據中
        ref.update({"data": firestore.ArrayUnion([data])})
    else:
        # 如果文檔不存在，則創建新的集合並添加第一條數據
        ref.set({"data": firestore.ArrayUnion([data])})


### 線程 ###
# 讀取感測器數據，並同步至雲端
def sensor_process():
    while True:
        try:
            humidity, temperature = Adafruit_DHT.read_retry(
                DHTSensor, DHTPin
            )  # 讀取溫濕度訊號

            # 讀取 SPI 腳位
            light = ReadChannel(1)  # 光照感測器
            soilHumidity = ReadChannel(0)  # 土壤濕度感測器
            water = ReadChannel(2)  # 水位

            # 列印感測器數據
            if humidity is not None and temperature is not None:
                print(
                    f"==================\n溫度: {temperature:.1f}°C ｜ 濕度: {humidity:.1f}%"
                )
            if light is not None:
                print("光照感測器：", light)
            if soilHumidity is not None:
                print("土壤濕度數據：", abs(soilHumidity - 1000) / 7)
            if water is not None:
                print("水位：", water)

            # 統合數據
            soilHumidity_persen = abs(soilHumidity - 1000) / 7  # 轉換為 % 數
            sensors_datas = {
                "temperature": f"{temperature:.1f}",
                "humidity": f"{humidity:.1f}",
                "light": light,
                "soilHumidity": f"{soilHumidity_persen:.1f}",
                "water": water,
                "timestamp": datetime.datetime.utcnow(),
            }

            # 更新到雲端資料庫
            writeSensorDataToFirestore(sensors_Ref, sensors_datas)

            time.sleep(60)  # 數據更新間隔
            GPIO.cleanup()
        except Exception as e:
            print(e)


# 持續檢測環境光線，並控制植物燈
def plantLights():
    GPIO.output(LEDPin, GPIO.LOW)
    while True:
        # 讀取 SPI 腳位
        light = ReadChannel(1)  # 光照感測器

        if light < 200:
            GPIO.output(LEDPin, GPIO.HIGH)
        else:
            GPIO.output(LEDPin, GPIO.LOW)

        # GPIO.output(LEDPin, GPIO.HIGH)
        # time.sleep(1)
        # GPIO.output(LEDPin, GPIO.LOW)

        time.sleep(1)


# 抽水馬達繼電器
def pumpingMotor():
    while True:
        soilHumidity = ReadChannel(0)  # 土壤濕度感測器
        GPIO.output(PumpingMotorPin, GPIO.LOW)

        if abs(soilHumidity - 1000) / 7 < 40:
            GPIO.output(PumpingMotorPin, GPIO.HIGH)
            time.sleep(10)
            GPIO.output(PumpingMotorPin, GPIO.LOW)
        else:
            GPIO.output(PumpingMotorPin, GPIO.LOW)
        # else:
        #     time.sleep(15)
    #     control_number = input("請輸入抽水馬達操作碼（1.啟動 2.關閉）：", None)

    #     if control_number.strip() != "":
    #         if control_number == "1":
    #             GPIO.output(PumpingMotorPin, GPIO.HIGH)
    #             print("馬達已啟動")
    #         elif control_number == "2":
    #             GPIO.output(PumpingMotorPin, GPIO.LOW)
    #             print("馬達已關閉")
    #         else:
    #             print("操作碼無效")
    #     else:
    #         print("請輸入操作碼")


# 語音辨識(棄用)
def speechRecognition():
    THRESHOLD = 1600  # 聲音閾值設定

    # 初始化Recognizer和Microphone一次
    r = sr.Recognizer()
    mic = sr.Microphone()

    while True:
        with mic as source:
            print("【監聽中...】")
            r.adjust_for_ambient_noise(source)  # 自動調整麥克風噪音水平
            audio_stream = r.listen(source, timeout=None)

        try:
            print("【語音辨識中...】")
            text = r.recognize_google(audio_stream, language="zh-TW")
            print(f"【語音辨識結果】=> {text}")
            chat(text)
        except sr.UnknownValueError:
            print("【無法識別語音】")
        except sr.RequestError as e:
            print(f"【無法從Google Speech Recognition 服務取得結果】：{e}")

        # 檢測聲音是否超過閾值
        frame_data = np.frombuffer(audio_stream.frame_data, dtype=np.int16)
        rms = np.sqrt(np.mean(frame_data.astype(float) ** 2))

        if rms > THRESHOLD:
            print("【辨識到高強度聲音】")


###
### 語音交流
# 將對話紀錄同步到 Firebase
def sync_chat_to_firestore(chat_history):
    try:
        # 將聊天紀錄更新到Firestore
        chat_history_ref.set({"messages": chat_history, "KEEP_RECENT": KEEP_RECENT})
        return True
    except Exception as e:
        print(f"An error occurred while syncing to Firestore: {e}")
        return False


# 上傳音檔到 Firebase Storage
def upload_audio_to_firebase_storage(file_path):
    # 隨機文件名
    random_filename = f"{uuid.uuid4()}.mp3"
    # 上傳路徑
    storage_path = f"assets/audios/{random_filename}"

    # 獲取儲存桶
    bucket = storage.bucket()
    # 創建 blob 來儲存文件
    blob = bucket.blob(storage_path)

    # 上傳文件
    blob.upload_from_filename(file_path)

    # 生成並新增訪問令牌
    new_token = uuid.uuid4()
    metadata = {"firebaseStorageDownloadTokens": new_token}
    blob.metadata = metadata
    blob.patch()

    # 該文件的公共 URL（含文件路徑及訪問令牌）
    file_url = f"https://firebasestorage.googleapis.com/v0{blob.path}?alt=media&token={new_token}"

    return file_url


# GPT 對話
def chat(content):
    global chat_history  # 引用全局變量 chat_history
    print("【正在等待 GPT-4 Turbo 回應...】")

    # 對話紀錄（用戶）
    user_message = {
        "role": "user",
        "content": content,
    }
    # 將新對話加入到對話紀錄中
    chat_history.append(user_message)  # 對話紀錄（全）
    chat_history_five_rounds.append(user_message)  # 對話紀錄（五輪）
    # 同步到 Firebase
    sync_chat_to_firestore(chat_history)

    # 向 OpenAI 發送對話請求
    # response = openai.chat.completions.create(
    #     model="ft:gpt-3.5-turbo-1106:personal::8SFY7onV",  # 替换为您的微调模型 ID
    #     messages=chat_history_five_rounds,
    #     max_tokens=2048,
    #     temperature=0.5,
    # )

    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=chat_history_five_rounds,
        max_tokens=2048,
        temperature=0.5,  # 設置溫度以增加創造性
        # stream=True,
    )

    # 提取 GPT 回覆內容
    gpt_response_content = response.choices[0].message.content
    # 列印
    print(
        f"\n== GPT-4 =====================\n{gpt_response_content}\n=============================\n"
    )

    # 調用 text_to_speech 函數文字轉語音（在語音生成以後，對話紀錄將更新到 Firestore(含音頻 URL))
    asyncio.run(text_to_speech(gpt_response_content))

    # 語音播放完畢後，列印完整對話
    print("\n== 完整對話 ==========")
    for message in chat_history:
        if message["role"] == "user":
            print(f"◼︎ User: {message['content']}")
        elif message["role"] == "assistant":
            print(f"▶︎ GPT: {message['content']}")
    print("=====================\n")

    # 退出函數，回到主程式 main


# Google TTS
async def text_to_speech(text):
    print("【正在生成語音...】")

    # 新增音頻文件
    speech_file_path = Path(__file__).parent / "speech.mp3"

    # 調用 OpenAI 的 TTS 服務
    response = openai.audio.speech.create(
        model="tts-1-1106",
        voice="alloy",
        input=text,
    )

    # 將回應的音頻流保存到音頻文件
    response.stream_to_file(speech_file_path)

    # 初始化 pygame 的混音器
    pygame.mixer.init()
    pygame.mixer.music.load(str(speech_file_path))  # 加载音频文件
    pygame.mixer.music.play()  # 播放音频
    print("【語音播放中...】")

    # 上傳檔案到 Firebase Storage，並返回音頻 URL
    file_url = upload_audio_to_firebase_storage(speech_file_path)

    # 對話紀錄構建
    gpt_message = {
        "role": "assistant",
        "content": text,
    }
    gpt_message_audio = {
        "role": "assistant",
        "content": text,
        "audio_url": file_url,  # 音頻文件的 URL
    }
    chat_history.append(gpt_message_audio)  # 歷史紀錄（全）
    chat_history_five_rounds.append(gpt_message)  # 對話紀錄（五輪）
    sync_chat_to_firestore(chat_history)

    # 檢查歷史對話記錄長度，如果超過 MAX_HISTORY，則刪除 KEEP_RECENT index 後的多餘的對話紀錄，直到剩下 MAX_HISTORY 條紀錄
    while len(chat_history_five_rounds) > MAX_HISTORY:
        del chat_history_five_rounds[(KEEP_RECENT)]

    # 等待播放完成
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(1)
    print("【語音結束】")


# Whisper TTS
async def tts_whisper(text):
    print("【正在生成語音...】")

    # 設定臨時文件 mp3
    speech_file_path = Path(__file__).parent / "speech.mp3"

    # 調用 OpenAI 的 TTS 服務
    response = openai.audio.speech.create(
        model="tts-1-1106",
        voice="alloy",
        input=text,
    )

    # 將回應的音頻留保存到文件
    response.stream_to_file(speech_file_path)

    # 初始化 pygame 的混音器
    pygame.mixer.init()

    # 加載音頻文件
    pygame.mixer.music.load(str(speech_file_path))

    # 播放音頻
    pygame.mixer.music.play()
    print("【語音播放中...】")

    # 循環等待播放完成
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(1)
    print("【語音結束】")


# 語音交流主程式
## 正在最佳化語音交流體驗
## 將在不久後推出新版本
def chatToAssistant():
    THRESHOLD = 1600  # 聲音閾值設定
    KEYWORDS = [
        "你好",
        "你們好",
        "哈囉",
        "Hi",
        "Hello",
        "嗨",
        "嘿",
        "Hey",
        "在嗎",
    ]  # 關鍵詞設定
    # 標記用戶是否已激活
    is_activated = False

    # 初始化 Recognizer 和 Microphone 一次
    r = sr.Recognizer()
    mic = sr.Microphone()

    # 清除對話
    sys.stdout.write("【正在清除上次對話...】")
    sys.stdout.flush()
    success_sync_chats = sync_chat_to_firestore(chat_history)
    if success_sync_chats:
        sys.stdout.write("\r【已清除上次對話】    \n")
        sys.stdout.flush()
    else:
        sys.stdout.write("\r【清除對話失敗】      \n")
        sys.stdout.flush()

    # 清除所有音頻文件
    bucket = storage.bucket()
    prefix = "assets/audios/"
    blobs = bucket.list_blobs(prefix=prefix)  # 所有文件对象
    sys.stdout.write("【正在清除上次段話的音頻...】")
    sys.stdout.flush()
    for blob in blobs:
        try:
            blob.delete()
            sys.stdout.write(f"\r【正在清除上次段話的音頻...{blob.name}】")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(
                f"\r【音頻清除失敗 {blob.name}: {e}                                                                   "
            )
            sys.stdout.flush()
    sys.stdout.write(
        "\r【已清除上次段話的音頻】                                                                             \n"
    )
    sys.stdout.flush()

    while True:
        # 開發測試用的輸入模式
        # text_input = input("請輸入文字開始對話，或者按 Enter 直接使用語音輸入：")
        # if text_input.strip() != "":
        #     chat(text_input)
        # else:

        with mic as source:
            assistant_status_ref.set({"status": "false"})
            r.adjust_for_ambient_noise(source, duration=2)  # 自動調整麥克風噪音水平
            r.dynamic_energy_threshold = True

            # 持續聆聽並辨識關鍵詞
            while not is_activated:
                print("【聆聽喚醒詞中...】：你好, 哈囉, 嗨, Hello, Hi")
                audio_data = r.listen(source)
                try:
                    text = r.recognize_google(audio_data, language="zh-TW")
                    if any(keyword in text for keyword in KEYWORDS):
                        is_activated = True  # 對話已開啟
                        assistant_status_ref.set({"status": "true"})
                        print("【請說...】")
                        audio_data = r.listen(source)
                        try:
                            text = r.recognize_google(audio_data, language="zh-TW")
                            if text:
                                print(f"【{text}】")
                                assistant_status_ref.set({"status": "loading"})
                                chat(text)
                        except sr.UnknownValueError:
                            print("【無法辨識語音】")
                        except sr.RequestError as e:
                            print(
                                f"【無法從 Google Speech Recognition 服務取得結果】：{e}"
                            )
                except sr.UnknownValueError:
                    print("【沒有辨識到喚醒詞...】")
                except sr.RequestError as e:
                    print(f"【無法從 Google Speech Recognition 服務取得結果】：{e}")

            # 對話已開啟，持續聆聽
            last_audio_time = time.time()
            while is_activated:
                assistant_status_ref.set({"status": "true"})
                print("【繼續說...】")
                audio_data = r.listen(source)
                try:
                    text = r.recognize_google(audio_data, language="zh-TW")
                    if text:
                        print(f"【{text}】")
                        assistant_status_ref.set({"status": "loading"})
                        chat(text)
                        last_audio_time = time.time()
                except sr.UnknownValueError:
                    print("【沒有辨識到語音】")
                except sr.RequestError as e:
                    print(f"【無法從 Google Speech Recognition 服務取得結果】：{e}")

                if time.time() - last_audio_time > 15:
                    is_activated = False
                    assistant_status_ref.set({"status": "false"})
                    break


# 多進程
sensor_process_thread = multiprocessing.Process(
    target=sensor_process
)  # 溫濕度感測器數據
plantLights_thread = multiprocessing.Process(target=plantLights)  # 植物燈
pumpingMotor_thread = multiprocessing.Process(target=pumpingMotor)  # 土壤濕度
chatToAssistant_thread = multiprocessing.Process(target=chatToAssistant)  # 語音交互

# 啟動進程
print("<<< start >>>")
sensor_process_thread.start()
plantLights_thread.start()
pumpingMotor_thread.start()
chatToAssistant_thread.start()
