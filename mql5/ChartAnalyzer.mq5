//+------------------------------------------------------------------+
//|  ChartAnalyzer.mq5                                               |
//|  Captures OHLC + screenshot on candle close → Python API        |
//+------------------------------------------------------------------+
#property copyright "MT5Bot"
#property version   "1.00"
#property strict

//--- Input parameters
input string           ServerURL      = "http://192.168.1.3:8000/analyze";  // Python backend URL
input string           Symbol_        = "XAUUSD";                          // Symbol to analyze
input ENUM_TIMEFRAMES  Timeframe_     = PERIOD_M5;                         // Timeframe (M1, M5, M15...)
input int              OHLCBars       = 50;                                // Number of bars to send
input int              RequestTimeout = 60000;                             // HTTP timeout ms
input bool             EnableLogging  = true;                              // Verbose logging

//--- Internal state
string   g_ScreenshotFile  = "";
datetime g_LastCandleTime  = 0;   // Thời gian mở của nến cuối cùng đã xử lý

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   Log("ChartAnalyzer EA initialising. Server=" + ServerURL);
   g_ScreenshotFile = TerminalInfoString(TERMINAL_DATA_PATH) +
                      "\\MQL5\\Files\\chart_snapshot.png";

   // Timer 1 giây để detect đúng thời điểm mở nến mới
   if(!EventSetTimer(1))
   {
      Print("ERROR: EventSetTimer failed.");
      return INIT_FAILED;
   }

   // Lấy thời gian nến hiện tại để không chạy ngay khi khởi động
   g_LastCandleTime = iTime(Symbol_, Timeframe_, 0);
   Log("Timer set to 1s (" + EnumToString(Timeframe_) + " candle sync). EA ready. Current candle: " + TimeToString(g_LastCandleTime));
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Log("ChartAnalyzer EA stopped. Reason=" + IntegerToString(reason));
}

//+------------------------------------------------------------------+
//| Timer event – chạy mỗi 1 giây, detect nến mới đóng             |
//+------------------------------------------------------------------+
void OnTimer()
{
   //--- Lấy thời gian mở của nến hiện tại (index 0)
   datetime currentCandleTime = iTime(Symbol_, Timeframe_, 0);

   //--- Chưa có nến mới → bỏ qua
   if(currentCandleTime == 0 || currentCandleTime <= g_LastCandleTime)
      return;

   //--- Nến mới vừa mở = nến trước vừa đóng → chạy pipeline
   Log("=== Nến " + EnumToString(Timeframe_) + " mới: " + TimeToString(currentCandleTime) +
       " (nến trước đóng: " + TimeToString(g_LastCandleTime) + ") ===");

   g_LastCandleTime = currentCandleTime;

   //--- 1. Capture chart screenshot
   if(!CaptureChart())
   {
      Print("ERROR: CaptureChart() failed. Aborting cycle.");
      return;
   }

   //--- 2. Get OHLC data as JSON string
   string ohlcJson = GetOHLC();
   if(ohlcJson == "")
   {
      Print("ERROR: GetOHLC() returned empty. Aborting cycle.");
      return;
   }

   //--- 3. Encode screenshot to Base64
   string b64Image = EncodeBase64(g_ScreenshotFile);
   if(b64Image == "")
   {
      Print("ERROR: EncodeBase64() failed. Aborting cycle.");
      return;
   }

   //--- 4. Build and send HTTP request
   SendRequest(ohlcJson, b64Image);
}

//+------------------------------------------------------------------+
//| Capture the current chart to PNG file                            |
//+------------------------------------------------------------------+
bool CaptureChart()
{
   //--- Use file name relative to MQL5\Files\
   string relFile = "chart_snapshot.png";
   long   chartId = ChartID();

   if(!ChartScreenShot(chartId, relFile, 1280, 720, ALIGN_LEFT))
   {
      int err = GetLastError();
      Print("ERROR: ChartScreenShot() failed. Code=", err);
      return false;
   }

   Log("Screenshot saved: " + relFile);
   return true;
}

//+------------------------------------------------------------------+
//| Build JSON array of last N OHLC bars (theo Timeframe_ đã chọn)  |
//+------------------------------------------------------------------+
string GetOHLC()
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   int copied = CopyRates(Symbol_, Timeframe_, 0, OHLCBars, rates);
   if(copied <= 0)
   {
      int err = GetLastError();
      Print("ERROR: CopyRates() failed. Copied=", copied, " Code=", err);
      return "";
   }

   Log("CopyRates copied " + IntegerToString(copied) + " bars.");

   string json = "[";
   for(int i = copied - 1; i >= 0; i--)   // oldest → newest
   {
      string ts   = TimeToString(rates[i].time, TIME_DATE | TIME_MINUTES);
      string open  = DoubleToString(rates[i].open,  5);
      string high  = DoubleToString(rates[i].high,  5);
      string low   = DoubleToString(rates[i].low,   5);
      string close = DoubleToString(rates[i].close, 5);
      long   vol   = rates[i].tick_volume;

      json += "{";
      json += "\"time\":\"" + ts + "\",";
      json += "\"open\":"   + open  + ",";
      json += "\"high\":"   + high  + ",";
      json += "\"low\":"    + low   + ",";
      json += "\"close\":"  + close + ",";
      json += "\"volume\":" + IntegerToString(vol);
      json += "}";

      if(i > 0)
         json += ",";
   }
   json += "]";

   return json;
}

//+------------------------------------------------------------------+
//| Read file from disk and encode bytes to Base64 string            |
//+------------------------------------------------------------------+
string EncodeBase64(const string filePath)
{
   //--- Open file from MQL5\Files\ (use short name only)
   string shortName = "chart_snapshot.png";
   int    handle    = FileOpen(shortName, FILE_READ | FILE_BIN);

   if(handle == INVALID_HANDLE)
   {
      int err = GetLastError();
      Print("ERROR: FileOpen failed for [", shortName, "]. Code=", err);
      return "";
   }

   ulong fileSize = FileSize(handle);
   if(fileSize == 0)
   {
      FileClose(handle);
      Print("ERROR: Screenshot file is empty.");
      return "";
   }

   uchar fileBytes[];
   ArrayResize(fileBytes, (int)fileSize);
   uint bytesRead = FileReadArray(handle, fileBytes, 0, (int)fileSize);
   FileClose(handle);

   if(bytesRead == 0)
   {
      Print("ERROR: FileReadArray read 0 bytes.");
      return "";
   }

   //--- Encode to Base64
   string result = "";
   if(!CryptEncode(CRYPT_BASE64, fileBytes, fileBytes, result))
   {
      Print("ERROR: CryptEncode(BASE64) failed.");
      return "";
   }

   Log("Base64 encoded " + IntegerToString(bytesRead) + " bytes → " +
       IntegerToString(StringLen(result)) + " chars.");

   return result;
}

//+------------------------------------------------------------------+
//| Build JSON payload and POST via WebRequest                       |
//+------------------------------------------------------------------+
void SendRequest(const string ohlcJson, const string b64Image)
{
   //--- Build JSON payload
   string symbol_val    = Symbol_;
   string timeframe_val = EnumToString(Timeframe_);  // "PERIOD_M1", "PERIOD_M5"...
   // Trim "PERIOD_" prefix cho gọn: M1, M5, H1...
   StringReplace(timeframe_val, "PERIOD_", "");

   string payload = "{";
   payload += "\"symbol\":\""    + symbol_val    + "\",";
   payload += "\"timeframe\":\"" + timeframe_val + "\",";
   payload += "\"ohlc\":"        + ohlcJson      + ",";
   payload += "\"image\":\""     + b64Image      + "\"";
   payload += "}";

   Log("Payload size: " + IntegerToString(StringLen(payload)) + " chars.");

   //--- Prepare headers
   string headers  = "Content-Type: application/json\r\n";
   string response = "";
   char   postData[];
   char   responseData[];

   StringToCharArray(payload, postData, 0, StringLen(payload), CP_UTF8);

   //--- Strip null terminator that StringToCharArray appends
   int dataLen = ArraySize(postData);
   if(dataLen > 0 && postData[dataLen - 1] == 0)
      ArrayResize(postData, dataLen - 1);

   //--- Fire HTTP POST
   int httpCode = WebRequest(
      "POST",
      ServerURL,
      headers,
      RequestTimeout,
      postData,
      responseData,
      response
   );

   if(httpCode == -1)
   {
      int err = GetLastError();
      Print("ERROR: WebRequest failed. Code=", err,
            ". Ensure URL is whitelisted in Tools→Options→Expert Advisors.");
      return;
   }

   //--- Parse response
   string responseStr = CharArrayToString(responseData, 0, WHOLE_ARRAY, CP_UTF8);

   if(httpCode == 200)
   {
      Log("SUCCESS [HTTP 200]. Response: " + responseStr);
      //--- Display signal on chart comment
      Comment("MT5Bot Signal:\n" + responseStr);
   }
   else
   {
      Print("WARNING: HTTP ", httpCode, ". Body: ", responseStr);
   }
}

//+------------------------------------------------------------------+
//| Conditional logger                                               |
//+------------------------------------------------------------------+
void Log(const string msg)
{
   if(EnableLogging)
      Print("[ChartAnalyzer] ", msg);
}

//+------------------------------------------------------------------+
//| CryptEncode wrapper for MQL5 Base64                              |
//+------------------------------------------------------------------+
bool CryptEncode(ENUM_CRYPT_METHOD method,
                 const uchar& data[],
                 uchar&       key[],
                 string&      result)
{
   uchar encoded[];
   int   encLen = CryptEncode(method, data, key, encoded);
   if(encLen <= 0)
      return false;

   result = CharArrayToString(encoded, 0, WHOLE_ARRAY, CP_UTF8);
   return true;
}
//+------------------------------------------------------------------+
