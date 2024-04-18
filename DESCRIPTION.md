# 程式說明

每個程式中，我們都寫了非常詳細的註釋說明。  
我們也正在努力改善程式的可讀性，我們將會逐步調整。

以下是各目錄及程式的簡介，  
以方便理解我們 Raspberry Pi Engine 的架構：

- [`main.py`](/main.py)：Raspberry Pi 正在運行的主程式。裡面建立了多個線程，以運行多個獨立功能，包含：語音交互、感測器監測、光照控制、澆水控制、數據趨勢分析。

- [`/modules/`](/modules/)：內含多個功能模組。包含：OpenAI交互功能、數據分析模組。

  - [`chatToGPT.py`](/modules/chatToGPT.py)：用於處理與GPT的對話請求
  - [`dataAnalysis.py`](/modules/dataAnalysis.py)：負責對數據進行分析運算，返回生成文本或分析結果，又或是將分析結果同步至雲端。

- [`/localTest`](/localTest/)：用於存放內部測試的功能，程式較為零散，不建議參考。
