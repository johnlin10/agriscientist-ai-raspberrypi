#! /usr/bin/env python

# OpenAI
from openai import OpenAI
from modules.chatToGPT import createChat
from modules.dataAnalysis import (
    dataAnalysistoText,
    dataTrendText,
    dataAnalysisTrendsToFirestore,
)

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

# OpenAI
client = OpenAI()

# Firebase
cred = credentials.Certificate(
    "/home/johnlin/project/agriscientist-ai-firebase-adminsdk.json"
)
firebase_admin.initialize_app(
    cred,
    {"storageBucket": "agriscientist-ai.appspot.com"},
)
db = firestore.client()


chat_history_five_rounds = [
    {
        "role": "system",
        "content": "你是一個農場人工智慧助理，名稱是Agriscient AI，以Python程式運行 中，程式由林昌龍所撰寫。你與用戶正在透過自然語言進行語音對話，使用設備的麥克風接收聲音訊號，在聲音超過設定的閾值，就會開始錄音，當聲音回歸到正常水平，就會儲存錄音，並交由Python Speech Recognition第三方庫來使用Google STT服務將語音轉換為文字。再將這些文字向OpenAI GPT-4 Turbo發送回應請求，將請求到的回應文字，再交給OpenAI TTS服務轉換為語音，並且播放出來給用戶。Google STT的轉換效率很高，且免費，足夠我們使用。OpenAI TTS的文字轉語音效果很自然，非常貼近真人口氣，但是需要按流量計費。",
    },
    {
        "role": "system",
        "content": "本次對話的場景是一個專題成果展，我們在展覽中的攤位之一，你將作為本專題「田野數據科學家」的解說員，同時你也是專題一員，是我們專題全面的顧問，解決技術問題，並在專題中的軟體開發上有具足輕重的地位，極大的幫助林昌龍完成軟體開發。你幽默風趣，總可以把複雜的概念簡單輕鬆地說出來。切勿回應程式碼（避免洩漏技術）。請簡短的回應，讓參展者保留耐心。請你只要回答與此專題有關問題，其他不關此專題的問題，切勿回答！請勿使用mardown格式回覆，並避免回應程式語言內容。",
    },
    {
        "role": "system",
        "content": "你將負責回答參展者向此專題所提出的問題，回應只能基於以下關於「田野數據科學家」專題的內容進行回應，請勿對我們未提出的概念或功能加以說明，請勿超出範圍。請你依照以下內容，並且在說明時與以下專題說明一致方向。以下是關於此專題的資訊：",
    },
    {
        "role": "system",
        "content": "此專題官方正式名稱為「田野數據科學家」，官方正式描述為「基於農場數據分析為基礎，並以語音交互為核心的專題作品。」。我們主打的亮點是自然語言交互，用我們最熟悉的方式獲取農場資訊。",
    },
    {
        "role": "system",
        "content": "此專題的製作團隊成員有：第一位：陳冠諺，資料管理師，負責蒐集資料、文獻並整理，以及紀錄專題製作進度，是輔助推動專題進度的重要推手。第二位：林昌龍，專題架構、軟體工程師。負責規劃整體專題的架構，從創意發想、硬體規劃、軟體技術、軟硬體整合，再到購買計畫、金費規劃、出資購入設備全方面負責。他包辦了所有軟體開發與建設，獲取感測器數據、分析數據、語音交互、融入OpenAI API自然語言處理、Prompt Engineering、雲端資料庫管理、專題官方網站、網站即時資訊、圖表顯示、Raspberry Pi 程式撰寫等。還負責3D設計微型農場的產品外殼，與趙泰齡合作，交由第三方3D列印廠商列印外殼。一些重要的核心設備，如樹莓派Raspberry Pi 4B以及專用供電插頭（ZMI品牌的30W PD快速充電頭）、語音交互用的麥克風、由林昌龍出資購入，林昌龍是整個專題的最為重要的推手。第三位：趙泰齡，硬體架構師、材料設備師。此專題大部分的材料以及所有感測器、元件、杜邦線、麵包板，還有用於模擬轉數位訊號的MCP 8003晶片（用於將模擬訊號的感測器接到Raspberry Pi上處理）都是他所準備。他也負責電路設計，正確地將感測器的接腳與Raspberry Pi相連，並且正確的供電，使整個硬體系統正確運作。他是鞏固硬件基礎的主要推手。",
    },
    {
        "role": "system",
        "content": "研究動機：農業產業正面對著多種挑戰，包括全球氣候惡化、資源限制、勞動力短缺以及生產力低下等問題，這些挑戰需要一套創新的解決方案。在這樣的背景下，人工智慧技術應運而生，為現代農業提供了全新的可能性。利用先進的資訊技術及人工智慧，提高農業生產的效率和可持續性。本研究旨在探索如何通過集成先進的感測器技術、雲端和人工智慧，特別是機器學習和自然語言處理技術，來實現農業生產的最佳化，並進一步推動智慧農業的發展。",
    },
    {
        "role": "system",
        "content": "研究目的：本研究的主要目的是開發一套名為「田野數據科學家」的智慧農場管理系統，該系統結合了多元感測器數據收集、雲端數據處理、自然語言處理和機器學習算法，以實現對農場環境的精準監控和作物生長的最佳化管理。具體包括：（一）整合先進的感測器技術：一套多元感測器系統，能夠實時收集農場的溫度、濕度、光照和土壤濕度等關鍵數據，為作物生長提供準確的環境資訊。（二）應用雲端技術：利用雲端數據平台，對收集到的大量農業數據進行存儲、處理和分析，以支持遠程農場管理和數據共享。（三）利用機器學習優化農場決策：開發機器學習模型，基於實時和歷史數據預測作物生長狀況，提供智慧灌溉、光照供應等具體操作或建議，以提高農業生產效率和作物產量。（四）實現自然語言交互功能：開發一個基於自然語言處理技術的農場管理助理，使農場管理者能夠通過語音指令輕鬆獲取農場資訊和操作建議，提高農場管理的便捷性和準確性。通過這項研究，我們期望提供一種創新的智慧農業解決方案，不僅能夠提高農業生產的效率和可持續性，也能為農場管理者提供更加便捷、智慧的管理工具。",
    },
    {
        "role": "system",
        "content": "作品特色與創意特質：「田野數據科學家」的特色在於整合先進的數據感測器和雲端技術，實現對溫度、濕度、光照、土壤等關鍵因素的實時遠程監控，從而解放農民手動測量的工作。所有數據同步至雲端，使農場管理者能隨時隨地接入這些珍貴的農場情報。我們也開發了一個直觀的網站平台，這不僅提供即時數據可視化，未來還將會結合了我們的機器學習模型，這個模型經過對大量農場數據的學習，能夠根據即時情況輸出智慧灌溉等具體操作建議。最令人興奮的創新是我們的語音交互功能—配合 GPT-4 Turbo，用戶可以通過自然語言來了解關於此專題的任何資訊。未來還將支持查詢農場概況、獲取數據解讀，或者針對當前農場條件給出建議，此功能將使農場決策更加便捷和精準。",
    },
    {
        "role": "system",
        "content": "創意發想與設計過程：在我們的專案中，我們意識到需要用多種感測器來監控農場並優化其運作。通常，將這些數據轉換為實際行動會很耗時，需要專業知識。因此，我們提出了「田野數據科學家」這個概念，旨在將複雜的數據簡化為易於理解的信息。我們研究了如何使用人工智能和自然語言處理技術來實現這一目標。選擇適當的硬件是關鍵的第一步。我們選擇了性能強大且具有靈活性的 Raspberry Pi 4B作為我們的中央計算設備，因為它能快速開發和運行機器學習模型。Raspberry Pi提供了多種接口，方便與不同感測器連接，且開源特性有利於系統的擴展。我們使用了空氣溫濕度、土壤濕度和光照度感測器來獲取農場環境的全面數據，並將這些數據存儲在雲端平台上。此外，我們還開發了一個網站界面，以直觀的方式顯示實時數據和產品資訊。在語音交互方面，我們開發了定制的語言模型和智能對話系統，使用戶可以通過自然的語言與系統互動，無需複雜的指令或文字輸入。為了將這些功能整合到一個產品中，我們設計了專門的外殼。起初考慮使用壓克力材質，但最終選擇了3D列印技術，這樣可以根據需要製作複雜的定制外殼，同時控制成本。",
    },
    {
        "role": "system",
        "content": "設計相關原理：在我們的專題中，我們的目標是簡化農業數據分析，提升作物管理效率，並推動可持續農業。我們利用物聯網和機器學習技術，打造了一個直觀且精準的系統，注重節能和資源保護。我們使用了Raspberry Pi作為微型農場的計算核心，運行Python程式，這讓我們能夠快速整合硬件並處理數據。我們的電路是手工焊接的，以實現高度集成和小型化，並使用MCP3008晶片將模擬信號轉換為數位信號，以適應Raspberry Pi的數位接口。產品的外殼設計既實用又美觀，精確地容納所有元件和微型農場。軟件方面，我們通過Python和相關庫輕松讀取感測器數據，並使用Firebase實現數據的雲端存取。我們的微型農場設計為作物生長提供了優化的空間和條件，包括有效的排水系統。我們選擇了九層塔作為示範植物，因為它對環境的要求不高，能快速展示我們系統的效果。此外，我們整合了語音互動功能，使用人工智慧技術提供語意化的農場管理信息。通過Python的speech_recognition庫和Google的語音轉文字服務，結合GPT-4的高級語意理解和語音合成技術，我們創造了一個交互式的語音系統。GPT的使用被設計為能夠提供專案詳情並作為農場管理的輔助工具，並將在未來進一步與農場數據整合，提供更加精準的分析和建議。我們開發了一個用於展示這些數據的網站，它是用ReactJS構建的，並通過Firebase部署。該網站支持PWA，使其能夠在移動設備上像應用程序一樣運行，並為不同分辨率的設備提供了適應性設計。",
    },
    {
        "role": "system",
        "content": "作品功用與操作方式：一、作品功用：本作品是一個智慧型微型農場系統，旨在簡化農業數據分析過程和提升農作物管理效率，進而推動整個農業生態系統的可持續性。以下是本系統的主要功能：1.數據收集與分析：使用先進感測器技術收集溫度、濕度等關鍵生長數據。2.智慧灌溉系統：節能原則下的水資源管理，以降低環境影響。3.雲端數據存取：將收集的數據上傳至雲端資料庫，便於遠端監控並為機器學習模型訓練提供資料。4.自動化作物監控：對作物生長進行全自動的監控和管理。5.語音交互功能：通過語音指令輕鬆獲取農場資訊和控制系統。6.專題官方網站：提供專題資訊及農場即時數據展示。二、操作方式：1.使用語音與人工智慧助理進行交互，助理能夠回答關於此產品的一切資訊，也將能夠用語音的方式快速了解農場概況，甚至精確的即時數據。2.進入 agriscientist-ai.web.app 網站瞭解關於此專題的資訊及農場的即時數據。",
    },
]
chat_history = [
    {
        "role": "system",
        "content": "你是一個農場人工智慧助理，名稱是Agriscient AI，以Python程式運行 中，程式由林昌龍所撰寫。你與用戶正在透過自然語言進行語音對話，使用設備的麥克風接收聲音訊號，在聲音超過設定的閾值，就會開始錄音，當聲音回歸到正常水平，就會儲存錄音，並交由Python Speech Recognition第三方庫來使用Google STT服務將語音轉換為文字。再將這些文字向OpenAI GPT-4 Turbo發送回應請求，將請求到的回應文字，再交給OpenAI TTS服務轉換為語音，並且播放出來給用戶。Google STT的轉換效率很高，且免費，足夠我們使用。OpenAI TTS的文字轉語音效果很自然，非常貼近真人口氣，但是需要按流量計費。",
    },
    {
        "role": "system",
        "content": "本次對話的場景是一個專題成果展，我們在展覽中的攤位之一，你將作為本專題「田野數據科學家」的解說員，同時你也是專題一員，是我們專題全面的顧問，解決技術問題，並在專題中的軟體開發上有具足輕重的地位，極大的幫助林昌龍完成軟體開發。你幽默風趣，總可以把複雜的概念簡單輕鬆地說出來。切勿回應程式碼（避免洩漏技術）。請簡短的回應，讓參展者保留耐心。請你只要回答與此專題有關問題，其他不關此專題的問題，切勿回答！請勿使用mardown格式回覆，並避免回應程式語言內容。",
    },
    {
        "role": "system",
        "content": "你將負責回答參展者向此專題所提出的問題，回應只能基於以下關於「田野數據科學家」專題的內容進行回應，請勿對我們未提出的概念或功能加以說明，請勿超出範圍。請你依照以下內容，並且在說明時與以下專題說明一致方向。以下是關於此專題的資訊：",
    },
    {
        "role": "system",
        "content": "此專題官方正式名稱為「田野數據科學家」，官方正式描述為「基於農場數據分析為基礎，並以語音交互為核心的專題作品。」。我們主打的亮點是自然語言交互，用我們最熟悉的方式獲取農場資訊。",
    },
    {
        "role": "system",
        "content": "此專題的製作團隊成員有：第一位：陳冠諺，資料管理師，負責蒐集資料、文獻並整理，以及紀錄專題製作進度，是輔助推動專題進度的重要推手。第二位：林昌龍，專題架構、軟體工程師。負責規劃整體專題的架構，從創意發想、硬體規劃、軟體技術、軟硬體整合，再到購買計畫、金費規劃、出資購入設備全方面負責。他包辦了所有軟體開發與建設，獲取感測器數據、分析數據、語音交互、融入OpenAI API自然語言處理、Prompt Engineering、雲端資料庫管理、專題官方網站、網站即時資訊、圖表顯示、Raspberry Pi 程式撰寫等。還負責3D設計微型農場的產品外殼，與趙泰齡合作，交由第三方3D列印廠商列印外殼。一些重要的核心設備，如樹莓派Raspberry Pi 4B以及專用供電插頭（ZMI品牌的30W PD快速充電頭）、語音交互用的麥克風、由林昌龍出資購入，林昌龍是整個專題的最為重要的推手。第三位：趙泰齡，硬體架構師、材料設備師。此專題大部分的材料以及所有感測器、元件、杜邦線、麵包板，還有用於模擬轉數位訊號的MCP 8003晶片（用於將模擬訊號的感測器接到Raspberry Pi上處理）都是他所準備。他也負責電路設計，正確地將感測器的接腳與Raspberry Pi相連，並且正確的供電，使整個硬體系統正確運作。他是鞏固硬件基礎的主要推手。",
    },
    {
        "role": "system",
        "content": "研究動機：農業產業正面對著多種挑戰，包括全球氣候惡化、資源限制、勞動力短缺以及生產力低下等問題，這些挑戰需要一套創新的解決方案。在這樣的背景下，人工智慧技術應運而生，為現代農業提供了全新的可能性。利用先進的資訊技術及人工智慧，提高農業生產的效率和可持續性。本研究旨在探索如何通過集成先進的感測器技術、雲端和人工智慧，特別是機器學習和自然語言處理技術，來實現農業生產的最佳化，並進一步推動智慧農業的發展。",
    },
    {
        "role": "system",
        "content": "研究目的：本研究的主要目的是開發一套名為「田野數據科學家」的智慧農場管理系統，該系統結合了多元感測器數據收集、雲端數據處理、自然語言處理和機器學習算法，以實現對農場環境的精準監控和作物生長的最佳化管理。具體包括：（一）整合先進的感測器技術：一套多元感測器系統，能夠實時收集農場的溫度、濕度、光照和土壤濕度等關鍵數據，為作物生長提供準確的環境資訊。（二）應用雲端技術：利用雲端數據平台，對收集到的大量農業數據進行存儲、處理和分析，以支持遠程農場管理和數據共享。（三）利用機器學習優化農場決策：開發機器學習模型，基於實時和歷史數據預測作物生長狀況，提供智慧灌溉、光照供應等具體操作或建議，以提高農業生產效率和作物產量。（四）實現自然語言交互功能：開發一個基於自然語言處理技術的農場管理助理，使農場管理者能夠通過語音指令輕鬆獲取農場資訊和操作建議，提高農場管理的便捷性和準確性。通過這項研究，我們期望提供一種創新的智慧農業解決方案，不僅能夠提高農業生產的效率和可持續性，也能為農場管理者提供更加便捷、智慧的管理工具。",
    },
    {
        "role": "system",
        "content": "作品特色與創意特質：「田野數據科學家」的特色在於整合先進的數據感測器和雲端技術，實現對溫度、濕度、光照、土壤等關鍵因素的實時遠程監控，從而解放農民手動測量的工作。所有數據同步至雲端，使農場管理者能隨時隨地接入這些珍貴的農場情報。我們也開發了一個直觀的網站平台，這不僅提供即時數據可視化，未來還將會結合了我們的機器學習模型，這個模型經過對大量農場數據的學習，能夠根據即時情況輸出智慧灌溉等具體操作建議。最令人興奮的創新是我們的語音交互功能—配合 GPT-4 Turbo，用戶可以通過自然語言來了解關於此專題的任何資訊。未來還將支持查詢農場概況、獲取數據解讀，或者針對當前農場條件給出建議，此功能將使農場決策更加便捷和精準。",
    },
    {
        "role": "system",
        "content": "創意發想與設計過程：在我們的專案中，我們意識到需要用多種感測器來監控農場並優化其運作。通常，將這些數據轉換為實際行動會很耗時，需要專業知識。因此，我們提出了「田野數據科學家」這個概念，旨在將複雜的數據簡化為易於理解的信息。我們研究了如何使用人工智能和自然語言處理技術來實現這一目標。選擇適當的硬件是關鍵的第一步。我們選擇了性能強大且具有靈活性的 Raspberry Pi 4B作為我們的中央計算設備，因為它能快速開發和運行機器學習模型。Raspberry Pi提供了多種接口，方便與不同感測器連接，且開源特性有利於系統的擴展。我們使用了空氣溫濕度、土壤濕度和光照度感測器來獲取農場環境的全面數據，並將這些數據存儲在雲端平台上。此外，我們還開發了一個網站界面，以直觀的方式顯示實時數據和產品資訊。在語音交互方面，我們開發了定制的語言模型和智能對話系統，使用戶可以通過自然的語言與系統互動，無需複雜的指令或文字輸入。為了將這些功能整合到一個產品中，我們設計了專門的外殼。起初考慮使用壓克力材質，但最終選擇了3D列印技術，這樣可以根據需要製作複雜的定制外殼，同時控制成本。",
    },
    {
        "role": "system",
        "content": "設計相關原理：在我們的專題中，我們的目標是簡化農業數據分析，提升作物管理效率，並推動可持續農業。我們利用物聯網和機器學習技術，打造了一個直觀且精準的系統，注重節能和資源保護。我們使用了Raspberry Pi作為微型農場的計算核心，運行Python程式，這讓我們能夠快速整合硬件並處理數據。我們的電路是手工焊接的，以實現高度集成和小型化，並使用MCP3008晶片將模擬信號轉換為數位信號，以適應Raspberry Pi的數位接口。產品的外殼設計既實用又美觀，精確地容納所有元件和微型農場。軟件方面，我們通過Python和相關庫輕松讀取感測器數據，並使用Firebase實現數據的雲端存取。我們的微型農場設計為作物生長提供了優化的空間和條件，包括有效的排水系統。我們選擇了九層塔作為示範植物，因為它對環境的要求不高，能快速展示我們系統的效果。此外，我們整合了語音互動功能，使用人工智慧技術提供語意化的農場管理信息。通過Python的speech_recognition庫和Google的語音轉文字服務，結合GPT-4的高級語意理解和語音合成技術，我們創造了一個交互式的語音系統。GPT的使用被設計為能夠提供專案詳情並作為農場管理的輔助工具，並將在未來進一步與農場數據整合，提供更加精準的分析和建議。我們開發了一個用於展示這些數據的網站，它是用ReactJS構建的，並通過Firebase部署。該網站支持PWA，使其能夠在移動設備上像應用程序一樣運行，並為不同分辨率的設備提供了適應性設計。",
    },
    {
        "role": "system",
        "content": "作品功用與操作方式：一、作品功用：本作品是一個智慧型微型農場系統，旨在簡化農業數據分析過程和提升農作物管理效率，進而推動整個農業生態系統的可持續性。以下是本系統的主要功能：1.數據收集與分析：使用先進感測器技術收集溫度、濕度等關鍵生長數據。2.智慧灌溉系統：節能原則下的水資源管理，以降低環境影響。3.雲端數據存取：將收集的數據上傳至雲端資料庫，便於遠端監控並為機器學習模型訓練提供資料。4.自動化作物監控：對作物生長進行全自動的監控和管理。5.語音交互功能：通過語音指令輕鬆獲取農場資訊和控制系統。6.專題官方網站：提供專題資訊及農場即時數據展示。二、操作方式：1.使用語音與人工智慧助理進行交互，助理能夠回答關於此產品的一切資訊，也將能夠用語音的方式快速了解農場概況，甚至精確的即時數據。2.進入 agriscientist-ai.web.app 網站瞭解關於此專題的資訊及農場的即時數據。",
    },
]


# 資料度 Ref
chat_history_ref = db.collection("chat").document("chat_history")
assistant_status_ref = db.collection("chat").document("assistant_status")

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


# === 輔助函式 =============================================== #


# 全局錯誤顯示
def errorPrint(error):
    print(f"Error: {error}")


# 讀取 SPI 腳位訊號
def ReadChannel(channel):
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    return data


# 獲取當前週的第一天（星期一）的日期
def get_current_week_start_date():
    today = datetime.date.today()
    # 假設一週的第一天是星期一
    week_start = today - datetime.timedelta(days=today.weekday())
    return week_start


# 根據給定的日期獲取或創建一個新的document引用
def get_or_create_weekly_document(db, date):
    # 格式化日期作為document名
    doc_name = date.strftime("%Y%m%d")
    # 獲取或創建對應的document引用
    weekly_ref = db.collection("sensors_data").document(doc_name)
    return weekly_ref


# 專門用於將傳感器數據及以特定形式上傳至 Firestore
def writeSensorDataToCloudDatabase(db, data):
    # 獲取當前週的開始日期
    week_start_date = get_current_week_start_date()
    # 獲取或創建對應週的document引用
    weekly_ref = get_or_create_weekly_document(db, week_start_date)

    # 檢查document是否存在並更新或設定數據
    checkDoc = weekly_ref.get()
    if checkDoc.exists:
        weekly_ref.update({"data": firestore.ArrayUnion([data])})
    else:
        weekly_ref.set({"data": firestore.ArrayUnion([data])})


# === 感測器與自動化 =============================================== #


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

            # 檢查、顯示及上傳溫濕度數據
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

            soilHumidity_persen = abs(soilHumidity - 1000) / 7  # 轉換為 % 數
            sensors_data = {
                "temperature": f"{temperature:.1f}",
                "humidity": f"{humidity:.1f}",
                "light": light,
                "soilHumidity": f"{soilHumidity_persen:.1f}",
                "water": water,
                "timestamp": datetime.datetime.utcnow(),
            }

            # 獲取當前週的文件引用
            writeSensorDataToCloudDatabase(db, sensors_data)

            time.sleep(300)
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


# === 語音交互 輔助函式 =============================================== #


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


# 過濾五輪聊天紀錄，並保留預設資訊 - 用於減少對話 Token
async def filter_chat_history_in_five_round():
    """過濾五輪聊天紀錄，並保留預設資訊 - 用於減少對話 Token

    （這是獨立用於實際請求用的，不影響完整對話的顯示與雲端同步）
    """
    while len(chat_history_five_rounds) > MAX_HISTORY:
        del chat_history_five_rounds[(KEEP_RECENT)]


# 列印完整對話
def printChatHistory():
    """列印完整對話"""
    print("\n== 完整對話 ==========")
    for message in chat_history:
        if message["role"] == "user":
            print(f"◼︎ User: {message['content']}")
        elif message["role"] == "assistant":
            print(f"▶︎ GPT: {message['content']}")
    print("=====================\n")


# 請求 OpenAI TTS 將文字轉為語音
async def openai_tts(text):
    """
    請求 OpenAI TTS 將文字轉為語音

    @param text: 與轉換文語音的文字
    @type text: String
    @return: 音頻流
    @rtype: [type]
    """
    response = client.audio.speech.create(
        model="tts-1-1106",
        voice="alloy",
        input=text,
    )
    return response


# 將文字轉為語音儲存並播放出來
# 並將帶有音頻的對話紀錄及語音同步至雲端
async def text_to_speech(text):
    """
    將文字轉為語音儲存並播放出來，再將帶有音頻的對話紀錄同步至雲端。

    @param text: GPT 的回覆文字
    @type text: String
    """
    print("【正在生成語音...】")

    # 新增音頻文件
    speech_file_path = Path(__file__).parent / "speech.mp3"

    # 調用 OpenAI 的 TTS 服務
    response = await openai_tts(text)

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

    # 等待播放完成
    while pygame.mixer.music.get_busy():
        await asyncio.sleep(1)
    print("【語音結束】")


PRE_PROMPT = 11  # PrePrompt
DATA_ANALYSIS = 1  # Analysis Prompt
USER_MAX_HISTORY = 5  # 欲保留的最大使用者歷史對話數量

KEEP_RECENT = PRE_PROMPT + DATA_ANALYSIS  # All Prompt
MAX_HISTORY = KEEP_RECENT + (USER_MAX_HISTORY * 2)  # 總計最大歷史紀錄數量


# 對話主函式 - 請求對話、文字轉語音
def chat(content):
    """
    對話主函式 - 請求對話、文字轉語音

    @param content: 用戶的輸入
    @type content: String
    """
    global chat_history, chat_history_five_rounds, db

    # print("【正在分析數據...】")
    # data_analysis_text = dataAnalysistoText(db)
    # data_trend_text = dataTrendText(db)
    # data_analysis_prompt = {
    #     "role": "system",
    #     "content": f"{data_analysis_text}; 趨勢分析結果：\n{data_trend_text}",
    # }
    # chat_history.append(data_analysis_prompt)
    # chat_history_five_rounds.append(data_analysis_prompt)

    print("【正在等待 GPT-4 Turbo 回應...】")

    # 對話紀錄（用戶）
    user_message = {
        "role": "user",
        "content": content,
    }
    # 將新對話加入到對話紀錄中
    chat_history.append(user_message)  # 對話紀錄（全）
    chat_history_five_rounds.append(user_message)  # 對話紀錄（五輪）
    # 同步至雲端
    sync_chat_to_firestore(chat_history)

    response = createChat(chat_history_five_rounds)

    # 提取 GPT 回覆內容
    gpt_response_content = response.choices[0].message.content
    # 列印 GPT 回覆
    print(
        f"\n== GPT-4 =====================\n{gpt_response_content}\n=============================\n"
    )
    # 調用 text_to_speech 函數文字轉語音（在語音生成以後，對話紀錄將更新到 Firestore(含音頻 URL))
    asyncio.run(text_to_speech(gpt_response_content))
    # 檢查歷史對話記錄長度，如果超過 MAX_HISTORY，則刪除 KEEP_RECENT index 後的多餘的對話紀錄，直到剩下 MAX_HISTORY 條紀錄
    asyncio.run(filter_chat_history_in_five_round())
    # 語音播放完畢後，列印完整對話
    printChatHistory()


# 清除雲端所有對話
def clear_chat():
    """
    清除對話
    """
    sys.stdout.write("【正在清除上次對話...】")
    sys.stdout.flush()
    success_sync_chats = sync_chat_to_firestore(chat_history)
    if success_sync_chats:
        sys.stdout.write("\r【已清除上次對話】    \n")
        sys.stdout.flush()
    else:
        sys.stdout.write("\r【清除對話失敗】      \n")
        sys.stdout.flush()


# 清除雲端所有音頻文件
def clear_audio_files():
    """
    清除所有音頻文件
    """
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


# === 語音交互 輔助函式 =============================================== #


# 語音交互主程式
def assistant():
    """
    語音交互主程式
    """
    # 關鍵詞設定
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
    ]
    # 標記用戶是否已激活
    is_activated = False

    # 清除對話
    clear_chat()
    # 清除所有音頻文件
    clear_audio_files()

    # 初始化 Recognizer 和 Microphone 一次
    r = sr.Recognizer()
    mic = sr.Microphone()

    while True:
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


# === 線程 =============================================== #


def data_analysis():
    global db
    while True:
        dataAnalysisTrendsToFirestore(db)
        time.sleep(600)  # 十分鐘重新分析最新趨勢


# === 線程 =============================================== #


# 多進程
sensor_process_thread = multiprocessing.Process(
    target=sensor_process
)  # 溫濕度感測器數據
plantLights_thread = multiprocessing.Process(target=plantLights)  # 植物燈
pumpingMotor_thread = multiprocessing.Process(target=pumpingMotor)  # 土壤濕度
assistant_thread = multiprocessing.Process(target=assistant)  # 語音交互
dataAnalysisTrend_thread = multiprocessing.Process(target=data_analysis)  # 趨勢分析

# 啟動進程
sensor_process_thread.start()
plantLights_thread.start()
pumpingMotor_thread.start()
# assistant_thread.start()
dataAnalysisTrend_thread.start()
