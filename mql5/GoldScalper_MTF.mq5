//+------------------------------------------------------------------+
//|  GoldScalper_MTF.mq5                                             |
//|  Multi-Timeframe EMA Pullback Strategy – XAUUSD                  |
//|                                                                  |
//|  FLOW:                                                           |
//|   1. MQL5 computes indicators (H1 trend, S/R, M5 pullback,      |
//|      candle pattern, score)                                      |
//|   2. If pre-filters pass → send OHLC + screenshot +             |
//|      indicator context to Python AI server                       |
//|   3. AI confirms or rejects the trade idea                      |
//|   4. Execute only if AI approves (confidence >= MinConfidence)  |
//+------------------------------------------------------------------+
#property copyright "MT5Bot"
#property version   "2.00"
#property strict

//=== Python AI Server ================================================
input string ServerURL      = "http://192.168.1.3:8000/analyze"; // Python backend URL
input int    OHLCBars       = 200;                             // OHLC bars to send (200 = ~16h M5, enough for EMA50+RSI warmup)
input int    RequestTimeout = 60000;                           // HTTP timeout ms
input double MinConfidence  = 0.65;                            // Min AI confidence to trade

//=== Risk Management =================================================
input double RiskPercent         = 1.0;    // Risk per trade (% of balance)
input double RR_Ratio            = 2.0;    // Risk:Reward ratio for TP

//=== Indicators ======================================================
input int    EMA_Fast            = 20;     // Fast EMA period (M5 & H1)
input int    EMA_Slow            = 50;     // Slow EMA period (M5 & H1)
input int    ATR_Period          = 14;     // ATR period (M5)
input int    SwingLookback       = 10;     // M5 bars back to find swing SL
input int    H1_SRLookback       = 50;     // H1 bars back for S/R detection

//=== Filters =========================================================
input double SidewayThreshold    = 0.3;   // Sideways: |EMA20-EMA50| < factor*ATR
input double SR_DistMultiplier   = 0.5;   // Min distance to S/R (factor*ATR)
input double MinBodyRatio        = 0.3;   // Min candle body/range ratio
input bool   UseSessionFilter    = true;  // Trade only London+NY sessions
input double MaxSpreadUSD        = 3.0;   // Max allowed spread in price units (e.g. 3.0 = $3 for XAUUSD)

//=== EA Settings =====================================================
input int    MagicNumber         = 20260319;
input bool   EnableLogging       = true;
input bool   EnableMLLog         = false; // Log trade features for ML training

//--- Symbol / Timeframe constants
#define SYMBOL    _Symbol
#define TF_M5     PERIOD_M5
#define TF_H1     PERIOD_H1

//--- Indicator handles
int g_hEMA20_M5  = INVALID_HANDLE;
int g_hEMA50_M5  = INVALID_HANDLE;
int g_hATR_M5    = INVALID_HANDLE;
int g_hEMA20_H1  = INVALID_HANDLE;
int g_hEMA50_H1  = INVALID_HANDLE;

//--- State
datetime g_LastBarTime    = 0;
string   g_ScreenshotFile = "";

//--- S/R level struct
struct SRLevel
{
   double price;
   bool   isResistance;   // true = resistance, false = support
};

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   Log("GoldScalper_MTF v2 initialising | Symbol=" + SYMBOL +
       " | Risk=" + DoubleToString(RiskPercent, 1) + "%" +
       " | RR=" + DoubleToString(RR_Ratio, 1) +
       " | Server=" + ServerURL +
       " | MinConf=" + DoubleToString(MinConfidence, 2));

   //--- Screenshot path (EA saves here, encoded and sent to server)
   g_ScreenshotFile = TerminalInfoString(TERMINAL_DATA_PATH) +
                      "\\MQL5\\Files\\gs_mtf_snapshot.png";

   //--- Create indicator handles
   g_hEMA20_M5 = iMA(SYMBOL, TF_M5, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   g_hEMA50_M5 = iMA(SYMBOL, TF_M5, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
   g_hATR_M5   = iATR(SYMBOL, TF_M5, ATR_Period);
   g_hEMA20_H1 = iMA(SYMBOL, TF_H1, EMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   g_hEMA50_H1 = iMA(SYMBOL, TF_H1, EMA_Slow, 0, MODE_EMA, PRICE_CLOSE);

   if(g_hEMA20_M5 == INVALID_HANDLE || g_hEMA50_M5 == INVALID_HANDLE ||
      g_hATR_M5   == INVALID_HANDLE || g_hEMA20_H1 == INVALID_HANDLE ||
      g_hEMA50_H1 == INVALID_HANDLE)
   {
      Print("ERROR: Indicator handle creation failed. Code=", GetLastError());
      return INIT_FAILED;
   }

   Log("All handles created. EA ready. Waiting for next M5 bar...");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   IndicatorRelease(g_hEMA20_M5);
   IndicatorRelease(g_hEMA50_M5);
   IndicatorRelease(g_hATR_M5);
   IndicatorRelease(g_hEMA20_H1);
   IndicatorRelease(g_hEMA50_H1);
   Log("GoldScalper_MTF stopped. Reason=" + IntegerToString(reason));
}

//+------------------------------------------------------------------+
//| OnTick – main strategy loop                                      |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Execute only on new M5 bar close
   datetime currentBarTime = iTime(SYMBOL, TF_M5, 0);
   if(currentBarTime == g_LastBarTime)
      return;
   g_LastBarTime = currentBarTime;

   //--- Spread check (compare in price units, works for any symbol)
   double ask    = SymbolInfoDouble(SYMBOL, SYMBOL_ASK);
   double bid    = SymbolInfoDouble(SYMBOL, SYMBOL_BID);
   double spread = ask - bid;
   if(spread > MaxSpreadUSD)
   {
      Log("Spread " + DoubleToString(spread, _Digits) +
          " exceeds limit " + DoubleToString(MaxSpreadUSD, _Digits) + ". Skip.");
      return;
   }

   //--- Session filter
   if(UseSessionFilter && !IsActiveSession())
      return;

   //--- Only one position at a time
   if(HasOpenPosition())
      return;

   //--- Read M5 indicators (bar index 1 = last completed bar)
   double ema20_m5 = GetBuffer(g_hEMA20_M5, 1);
   double ema50_m5 = GetBuffer(g_hEMA50_M5, 1);
   double atr_m5   = GetBuffer(g_hATR_M5,   1);

   //--- Read H1 indicators
   double ema20_h1 = GetBuffer(g_hEMA20_H1, 1);
   double ema50_h1 = GetBuffer(g_hEMA50_H1, 1);

   if(ema20_m5 == 0 || ema50_m5 == 0 || atr_m5 == 0 ||
      ema20_h1 == 0 || ema50_h1 == 0)
   {
      Log("Indicators not ready yet. Skip.");
      return;
   }

   if(atr_m5 < 0.50)
   {
      Log("ATR too low (" + DoubleToString(atr_m5, 2) + "). Skip.");
      return;
   }

   //=================================================================
   // STEP 1: MQL5 Pre-filters
   //=================================================================

   //--- H1 trend
   int h1Trend = DetectTrendH1(ema20_h1, ema50_h1, atr_m5);
   if(h1Trend == 0) { Log("H1 sideways – no trade."); return; }

   //--- H1 S/R zones
   SRLevel srLevels[];
   DetectSupportResistanceH1(srLevels, atr_m5);

   //--- M5 pullbacks
   bool pullbackBuy  = DetectPullbackM5(ema20_m5, ema50_m5, true);
   bool pullbackSell = DetectPullbackM5(ema20_m5, ema50_m5, false);

   //--- Candle patterns
   int patternBuy  = DetectCandlePattern(true);
   int patternSell = DetectCandlePattern(false);

   double askPrice = SymbolInfoDouble(SYMBOL, SYMBOL_ASK);
   double bidPrice = SymbolInfoDouble(SYMBOL, SYMBOL_BID);

   //--- Determine direction and check S/R distance
   string direction  = "";
   int    pattern    = 0;
   double slPrice    = 0;
   double tpPrice    = 0;
   int    score      = 0;

   if(h1Trend == 1 && pullbackBuy && patternBuy > 0)
   {
      double distToRes = DistanceToSR(srLevels, askPrice, true);
      if(distToRes <= SR_DistMultiplier * atr_m5)
      {
         Log("BUY pre-filter blocked: too close to H1 resistance. Skip.");
         return;
      }
      direction = "BUY";
      pattern   = patternBuy;
      double swingLow = GetSwingLow(TF_M5, SwingLookback);
      slPrice   = NormalizeDouble(swingLow - atr_m5 * 0.2, _Digits);
      tpPrice   = NormalizeDouble(askPrice + (askPrice - slPrice) * RR_Ratio, _Digits);
      score     = ComputeScore(h1Trend, 1, ema20_m5, ema50_m5, atr_m5, true, pattern);
   }
   else if(h1Trend == -1 && pullbackSell && patternSell > 0)
   {
      double distToSup = DistanceToSR(srLevels, bidPrice, false);
      if(distToSup <= SR_DistMultiplier * atr_m5)
      {
         Log("SELL pre-filter blocked: too close to H1 support. Skip.");
         return;
      }
      direction = "SELL";
      pattern   = patternSell;
      double swingHigh = GetSwingHigh(TF_M5, SwingLookback);
      slPrice   = NormalizeDouble(swingHigh + atr_m5 * 0.2, _Digits);
      tpPrice   = NormalizeDouble(bidPrice - (slPrice - bidPrice) * RR_Ratio, _Digits);
      score     = ComputeScore(h1Trend, -1, ema20_m5, ema50_m5, atr_m5, true, pattern);
   }

   if(direction == "")
      return;   // No valid setup

   //--- Validate SL/TP positions
   double entryPrice = (direction == "BUY") ? askPrice : bidPrice;
   if(direction == "BUY"  && (slPrice >= entryPrice || tpPrice <= entryPrice)) return;
   if(direction == "SELL" && (slPrice <= entryPrice || tpPrice >= entryPrice)) return;

   //=================================================================
   // STEP 2: All pre-filters passed → consult Python AI server
   //=================================================================

   Log("Pre-filter PASSED | Dir=" + direction +
       " | Score=" + IntegerToString(score) +
       " | Pattern=" + PatternName(pattern) +
       " | H1 trend=" + IntegerToString(h1Trend) +
       " | Sending to AI server...");

   //--- 2a. Capture chart screenshot
   if(!CaptureChart())
   {
      Print("ERROR: CaptureChart() failed. Aborting.");
      return;
   }

   //--- 2b. Build OHLC JSON
   string ohlcJson = BuildOHLCJson();
   if(ohlcJson == "")
   {
      Print("ERROR: BuildOHLCJson() failed. Aborting.");
      return;
   }

   //--- 2c. Encode screenshot to Base64
   string b64Image = EncodeBase64();
   if(b64Image == "")
   {
      Print("ERROR: EncodeBase64() failed. Aborting.");
      return;
   }

   //--- 2d. Package all indicator context + send to server
   string responseJson = SendToServer(
      ohlcJson, b64Image, direction,
      h1Trend, ema20_h1, ema50_h1,
      ema20_m5, ema50_m5, atr_m5,
      pattern, score, slPrice
   );

   if(responseJson == "")
   {
      Print("ERROR: No response from AI server. Aborting.");
      return;
   }

   //=================================================================
   // STEP 3: Parse AI response → execute if approved
   //=================================================================

   string aiSignal     = ExtractStringField(responseJson, "signal");
   double aiConfidence = ExtractDoubleField(responseJson, "confidence");
   string aiReason     = ExtractStringField(responseJson, "reason");

   Log("AI Response | Signal=" + aiSignal +
       " | Confidence=" + DoubleToString(aiConfidence, 2) +
       " | Reason: " + aiReason);

   //--- Display AI analysis on chart
   Comment("GoldScalper AI:\n" +
           "Pre-filter: " + direction + " (score=" + IntegerToString(score) + ")\n" +
           "AI Signal: " + aiSignal + " (" + DoubleToString(aiConfidence * 100, 0) + "%)\n" +
           "Reason: " + aiReason);

   //--- Execute only when AI agrees with direction and is confident enough
   if(aiSignal != direction)
   {
      Log("AI disagreed: proposed=" + direction + " AI=" + aiSignal + ". Trade skipped.");
      return;
   }

   if(aiConfidence < MinConfidence)
   {
      Log("AI confidence too low: " + DoubleToString(aiConfidence, 2) +
          " < " + DoubleToString(MinConfidence, 2) + ". Trade skipped.");
      return;
   }

   Log("AI APPROVED | Executing " + direction + " trade.");

   if(EnableMLLog)
      LogTradeFeature(direction, ema20_h1 - ema50_h1, ema20_m5 - ema50_m5,
                      atr_m5, pattern, score, entryPrice, slPrice, tpPrice,
                      aiConfidence, aiReason);

   ENUM_ORDER_TYPE orderType = (direction == "BUY") ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
   ExecuteTrade(orderType, entryPrice, slPrice, tpPrice);
}

//+------------------------------------------------------------------+
//| Capture current chart to PNG file                                |
//+------------------------------------------------------------------+
bool CaptureChart()
{
   string relFile = "gs_mtf_snapshot.png";
   long   chartId = ChartID();

   if(!ChartScreenShot(chartId, relFile, 1280, 720, ALIGN_LEFT))
   {
      Print("ERROR: ChartScreenShot() failed. Code=", GetLastError());
      return false;
   }

   Log("Screenshot saved: " + relFile);
   return true;
}

//+------------------------------------------------------------------+
//| Build JSON array of last N OHLC bars                             |
//+------------------------------------------------------------------+
string BuildOHLCJson()
{
   MqlRates rates[];
   ArraySetAsSeries(rates, true);

   int copied = CopyRates(SYMBOL, TF_M5, 0, OHLCBars, rates);
   if(copied <= 0)
   {
      Print("ERROR: CopyRates() failed. Code=", GetLastError());
      return "";
   }

   string json = "[";
   for(int i = copied - 1; i >= 0; i--)
   {
      string ts    = TimeToString(rates[i].time, TIME_DATE | TIME_MINUTES);
      json += "{";
      json += "\"time\":\""  + ts + "\",";
      json += "\"open\":"    + DoubleToString(rates[i].open,  5) + ",";
      json += "\"high\":"    + DoubleToString(rates[i].high,  5) + ",";
      json += "\"low\":"     + DoubleToString(rates[i].low,   5) + ",";
      json += "\"close\":"   + DoubleToString(rates[i].close, 5) + ",";
      json += "\"volume\":"  + IntegerToString(rates[i].tick_volume);
      json += "}";
      if(i > 0) json += ",";
   }
   json += "]";
   return json;
}

//+------------------------------------------------------------------+
//| Read screenshot file and encode to Base64                        |
//+------------------------------------------------------------------+
string EncodeBase64()
{
   string shortName = "gs_mtf_snapshot.png";
   int    handle    = FileOpen(shortName, FILE_READ | FILE_BIN);

   if(handle == INVALID_HANDLE)
   {
      Print("ERROR: FileOpen failed. Code=", GetLastError());
      return "";
   }

   ulong fileSize = FileSize(handle);
   if(fileSize == 0) { FileClose(handle); Print("ERROR: Screenshot empty."); return ""; }

   uchar fileBytes[];
   ArrayResize(fileBytes, (int)fileSize);
   uint bytesRead = FileReadArray(handle, fileBytes, 0, (int)fileSize);
   FileClose(handle);

   if(bytesRead == 0) { Print("ERROR: FileReadArray read 0 bytes."); return ""; }

   string result = "";
   uchar  key[];
   uchar  encoded[];
   int encLen = CryptEncode(CRYPT_BASE64, fileBytes, key, encoded);
   if(encLen <= 0) { Print("ERROR: CryptEncode(BASE64) failed."); return ""; }

   result = CharArrayToString(encoded, 0, WHOLE_ARRAY, CP_UTF8);
   Log("Base64 encoded " + IntegerToString(bytesRead) + " bytes.");
   return result;
}

//+------------------------------------------------------------------+
//| Build full JSON payload and POST to Python server                |
//| Returns raw response JSON string, or "" on failure              |
//+------------------------------------------------------------------+
string SendToServer(
   const string ohlcJson,
   const string b64Image,
   const string direction,
   int    h1Trend,
   double ema20H1, double ema50H1,
   double ema20M5, double ema50M5,
   double atrM5,
   int    candlePattern,
   int    score,
   double swingSL
)
{
   //--- Timeframe string: "PERIOD_M5" → "M5"
   string tfStr = EnumToString(TF_M5);
   StringReplace(tfStr, "PERIOD_", "");

   //--- Build "indicators" sub-object
   string indicators = "{";
   indicators += "\"h1_trend\":"      + IntegerToString(h1Trend)           + ",";
   indicators += "\"h1_ema20\":"      + DoubleToString(ema20H1, 5)         + ",";
   indicators += "\"h1_ema50\":"      + DoubleToString(ema50H1, 5)         + ",";
   indicators += "\"m5_ema20\":"      + DoubleToString(ema20M5, 5)         + ",";
   indicators += "\"m5_ema50\":"      + DoubleToString(ema50M5, 5)         + ",";
   indicators += "\"atr_m5\":"        + DoubleToString(atrM5, 5)           + ",";
   indicators += "\"direction\":\""   + direction                          + "\",";
   indicators += "\"candle_pattern\":" + IntegerToString(candlePattern)    + ",";
   indicators += "\"score\":"         + IntegerToString(score)             + ",";
   indicators += "\"swing_sl\":"      + DoubleToString(swingSL, _Digits);
   indicators += "}";

   //--- Build full payload
   string payload = "{";
   payload += "\"symbol\":\""    + SYMBOL        + "\",";
   payload += "\"timeframe\":\"" + tfStr         + "\",";
   payload += "\"ohlc\":"        + ohlcJson      + ",";
   payload += "\"image\":\""     + b64Image      + "\",";
   payload += "\"indicators\":"  + indicators;
   payload += "}";

   Log("Sending payload to server. Size=" + IntegerToString(StringLen(payload)) + " chars.");

   //--- Prepare HTTP request
   string headers = "Content-Type: application/json\r\n";
   string response = "";
   char   postData[], responseData[];

   StringToCharArray(payload, postData, 0, StringLen(payload), CP_UTF8);
   int dataLen = ArraySize(postData);
   if(dataLen > 0 && postData[dataLen - 1] == 0)
      ArrayResize(postData, dataLen - 1);

   int httpCode = WebRequest(
      "POST", ServerURL, headers, RequestTimeout,
      postData, responseData, response
   );

   if(httpCode == -1)
   {
      Print("ERROR: WebRequest failed. Code=", GetLastError(),
            ". Whitelist URL in Tools→Options→Expert Advisors.");
      return "";
   }

   string responseStr = CharArrayToString(responseData, 0, WHOLE_ARRAY, CP_UTF8);

   if(httpCode == 200)
   {
      Log("Server responded [HTTP 200].");
      return responseStr;
   }
   else
   {
      Print("WARNING: Server returned HTTP ", httpCode, ". Body: ", responseStr);
      return "";
   }
}

//+------------------------------------------------------------------+
//| Extract a string value from flat JSON: "key":"value"            |
//+------------------------------------------------------------------+
string ExtractStringField(const string json, const string key)
{
   string search = "\"" + key + "\":\"";
   int pos = StringFind(json, search);
   if(pos < 0) return "";
   pos += StringLen(search);
   int end = StringFind(json, "\"", pos);
   if(end < 0) return "";
   return StringSubstr(json, pos, end - pos);
}

//+------------------------------------------------------------------+
//| Extract a numeric value from flat JSON: "key":number            |
//+------------------------------------------------------------------+
double ExtractDoubleField(const string json, const string key)
{
   string search = "\"" + key + "\":";
   int pos = StringFind(json, search);
   if(pos < 0) return 0.0;
   pos += StringLen(search);
   int end = pos;
   while(end < StringLen(json))
   {
      ushort ch = StringGetCharacter(json, end);
      if(ch == ',' || ch == '}' || ch == ' ' || ch == '\r' || ch == '\n')
         break;
      end++;
   }
   return StringToDouble(StringSubstr(json, pos, end - pos));
}

//+------------------------------------------------------------------+
//| Detect H1 trend                                                  |
//| Returns: 1=UP, -1=DOWN, 0=SIDEWAYS                              |
//+------------------------------------------------------------------+
int DetectTrendH1(double ema20, double ema50, double atr_m5)
{
   double gap = MathAbs(ema20 - ema50);
   if(gap < SidewayThreshold * atr_m5)
      return 0;
   return (ema20 > ema50) ? 1 : -1;
}

//+------------------------------------------------------------------+
//| Detect swing high/low S/R zones from H1 bars using fractal logic |
//+------------------------------------------------------------------+
void DetectSupportResistanceH1(SRLevel &levels[], double atr_m5)
{
   ArrayResize(levels, 0);

   int    total      = H1_SRLookback + 2;
   double highs[], lows[];
   ArraySetAsSeries(highs, true);
   ArraySetAsSeries(lows,  true);

   if(CopyHigh(SYMBOL, TF_H1, 0, total, highs) < total ||
      CopyLow (SYMBOL, TF_H1, 0, total, lows)  < total)
      return;

   double clusterThresh = atr_m5 * SR_DistMultiplier;

   for(int i = 1; i < H1_SRLookback + 1; i++)
   {
      //--- Fractal high: higher than both neighbours
      if(highs[i] > highs[i-1] && highs[i] > highs[i+1])
      {
         if(!LevelExists(levels, highs[i], clusterThresh))
         {
            int n = ArraySize(levels);
            ArrayResize(levels, n + 1);
            levels[n].price        = highs[i];
            levels[n].isResistance = true;
         }
      }

      //--- Fractal low: lower than both neighbours
      if(lows[i] < lows[i-1] && lows[i] < lows[i+1])
      {
         if(!LevelExists(levels, lows[i], clusterThresh))
         {
            int n = ArraySize(levels);
            ArrayResize(levels, n + 1);
            levels[n].price        = lows[i];
            levels[n].isResistance = false;
         }
      }
   }

   Log("H1 S/R detected: " + IntegerToString(ArraySize(levels)) + " levels.");
}

//--- Helper: check if a price is already clustered in existing levels
bool LevelExists(const SRLevel &levels[], double price, double threshold)
{
   for(int i = 0; i < ArraySize(levels); i++)
      if(MathAbs(levels[i].price - price) < threshold)
         return true;
   return false;
}

//+------------------------------------------------------------------+
//| M5 pullback: price inside EMA20–EMA50 zone on last closed bar    |
//+------------------------------------------------------------------+
bool DetectPullbackM5(double ema20, double ema50, bool isBuy)
{
   double close = iClose(SYMBOL, TF_M5, 1);
   double low   = iLow  (SYMBOL, TF_M5, 1);
   double high  = iHigh (SYMBOL, TF_M5, 1);

   if(isBuy)
   {
      //--- Need uptrend (ema20>ema50) and price dipped into or below EMA zone
      double zoneTop = MathMax(ema20, ema50);
      double zoneBot = MathMin(ema20, ema50);
      return (ema20 > ema50) && (low <= zoneTop) && (close >= zoneBot * 0.998);
   }
   else
   {
      double zoneTop = MathMax(ema20, ema50);
      double zoneBot = MathMin(ema20, ema50);
      return (ema20 < ema50) && (high >= zoneBot) && (close <= zoneTop * 1.002);
   }
}

//+------------------------------------------------------------------+
//| Detect candle pattern on last closed M5 bar                      |
//| Returns: 0=none, 1=pin bar, 2=engulfing                         |
//+------------------------------------------------------------------+
int DetectCandlePattern(bool isBullish)
{
   double o1 = iOpen (SYMBOL, TF_M5, 1);
   double h1 = iHigh (SYMBOL, TF_M5, 1);
   double l1 = iLow  (SYMBOL, TF_M5, 1);
   double c1 = iClose(SYMBOL, TF_M5, 1);

   double o2 = iOpen (SYMBOL, TF_M5, 2);
   double c2 = iClose(SYMBOL, TF_M5, 2);

   double range      = h1 - l1;
   if(range < _Point * 3)
      return 0;

   double body       = MathAbs(c1 - o1);
   double upperWick  = h1 - MathMax(o1, c1);
   double lowerWick  = MathMin(o1, c1) - l1;

   //--- Body too small → no momentum
   if(body < MinBodyRatio * range)
   {
      //--- Only pin bars allowed with small body (they rely on wick)
   }

   if(isBullish)
   {
      //--- Bullish pin bar: small body, long lower wick, close above midpoint
      bool isPinBar   = (lowerWick >= 2.0 * body) &&
                        (lowerWick >= 0.55 * range) &&
                        (c1 > (l1 + range * 0.5));

      //--- Bullish engulfing: prior bar bearish, current fully engulfs it
      bool isEngulf   = (c2 < o2) &&              // Prior: bearish
                        (c1 > o1) &&              // Current: bullish
                        (c1 >= o2) &&             // Close above prior open
                        (o1 <= c2);               // Open below prior close

      if(isPinBar)   return 1;
      if(isEngulf)   return 2;
   }
   else
   {
      //--- Bearish pin bar: small body, long upper wick, close below midpoint
      bool isPinBar   = (upperWick >= 2.0 * body) &&
                        (upperWick >= 0.55 * range) &&
                        (c1 < (l1 + range * 0.5));

      //--- Bearish engulfing
      bool isEngulf   = (c2 > o2) &&
                        (c1 < o1) &&
                        (c1 <= o2) &&
                        (o1 >= c2);

      if(isPinBar)   return 1;
      if(isEngulf)   return 2;
   }

   return 0;
}

//+------------------------------------------------------------------+
//| Distance from price to nearest relevant S/R level                |
//| For BUY: nearest resistance above price                          |
//| For SELL: nearest support below price                            |
//+------------------------------------------------------------------+
double DistanceToSR(const SRLevel &levels[], double price, bool isBuy)
{
   double minDist = DBL_MAX;
   for(int i = 0; i < ArraySize(levels); i++)
   {
      if(isBuy  && levels[i].isResistance && levels[i].price > price)
         minDist = MathMin(minDist, levels[i].price - price);
      if(!isBuy && !levels[i].isResistance && levels[i].price < price)
         minDist = MathMin(minDist, price - levels[i].price);
   }
   return minDist;
}

//+------------------------------------------------------------------+
//| Recent swing low on M5 (anchor for BUY stop loss)               |
//+------------------------------------------------------------------+
double GetSwingLow(ENUM_TIMEFRAMES tf, int lookback)
{
   double lows[];
   ArraySetAsSeries(lows, true);
   if(CopyLow(SYMBOL, tf, 1, lookback, lows) <= 0)
      return iLow(SYMBOL, tf, 1);

   double result = lows[0];
   for(int i = 1; i < ArraySize(lows); i++)
      if(lows[i] < result) result = lows[i];
   return result;
}

//+------------------------------------------------------------------+
//| Recent swing high on M5 (anchor for SELL stop loss)             |
//+------------------------------------------------------------------+
double GetSwingHigh(ENUM_TIMEFRAMES tf, int lookback)
{
   double highs[];
   ArraySetAsSeries(highs, true);
   if(CopyHigh(SYMBOL, tf, 1, lookback, highs) <= 0)
      return iHigh(SYMBOL, tf, 1);

   double result = highs[0];
   for(int i = 1; i < ArraySize(highs); i++)
      if(highs[i] > result) result = highs[i];
   return result;
}

//+------------------------------------------------------------------+
//| Scoring system (0–100)                                           |
//| Components: H1 trend strength, M5 EMA gap, pullback, pattern    |
//+------------------------------------------------------------------+
int ComputeScore(int h1Trend, int direction,
                 double ema20_m5, double ema50_m5, double atr_m5,
                 bool pullback, int pattern)
{
   if(h1Trend != direction) return 0;

   int score = 0;

   //--- H1 trend confirmed (30 pts)
   score += 30;

   //--- M5 EMA gap strength (25 pts)
   double m5Gap = MathAbs(ema20_m5 - ema50_m5);
   double gapRatio = m5Gap / (atr_m5 > 0 ? atr_m5 : 1.0);
   if(gapRatio > 0.5)  score += 25;
   else if(gapRatio > 0.2) score += 15;
   else score += 5;

   //--- Pullback into zone (25 pts)
   if(pullback) score += 25;

   //--- Candle pattern quality (20 pts)
   if(pattern == 2) score += 20;       // Engulfing – strongest
   else if(pattern == 1) score += 12;  // Pin bar

   return score;
}

//+------------------------------------------------------------------+
//| Lot size based on % risk and SL distance                         |
//+------------------------------------------------------------------+
double CalcLotSize(double entryPrice, double slPrice)
{
   double balance    = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = balance * RiskPercent / 100.0;
   double slPoints   = MathAbs(entryPrice - slPrice) / _Point;
   double tickValue  = SymbolInfoDouble(SYMBOL, SYMBOL_TRADE_TICK_VALUE);

   if(slPoints < 1 || tickValue <= 0)
   {
      Print("WARNING: Invalid SL distance or tick value. Using min lot.");
      return SymbolInfoDouble(SYMBOL, SYMBOL_VOLUME_MIN);
   }

   double lots    = riskAmount / (slPoints * tickValue);
   double minLot  = SymbolInfoDouble(SYMBOL, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(SYMBOL, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(SYMBOL, SYMBOL_VOLUME_STEP);

   lots = MathMax(minLot, MathMin(maxLot, MathFloor(lots / lotStep) * lotStep));
   return lots;
}

//+------------------------------------------------------------------+
//| Send a market order with SL and TP                               |
//+------------------------------------------------------------------+
void ExecuteTrade(ENUM_ORDER_TYPE type, double price, double sl, double tp)
{
   double lots = CalcLotSize(price, sl);
   string dir  = (type == ORDER_TYPE_BUY) ? "BUY" : "SELL";

   Log(">>> " + dir +
       " | Lots=" + DoubleToString(lots, 2) +
       " | Entry=" + DoubleToString(price, _Digits) +
       " | SL=" + DoubleToString(sl, _Digits) +
       " | TP=" + DoubleToString(tp, _Digits));

   //--- Determine broker-supported filling mode
   ENUM_ORDER_TYPE_FILLING filling;
   long fillMode = SymbolInfoInteger(SYMBOL, SYMBOL_FILLING_MODE);
   if((fillMode & SYMBOL_FILLING_FOK) != 0)
      filling = ORDER_FILLING_FOK;
   else if((fillMode & SYMBOL_FILLING_IOC) != 0)
      filling = ORDER_FILLING_IOC;
   else
      filling = ORDER_FILLING_RETURN;

   MqlTradeRequest req = {};
   MqlTradeResult  res = {};
   req.action       = TRADE_ACTION_DEAL;
   req.symbol       = SYMBOL;
   req.volume       = lots;
   req.type         = type;
   req.price        = price;
   req.sl           = sl;
   req.tp           = tp;
   req.deviation    = 10;
   req.magic        = MagicNumber;
   req.comment      = "GoldScalper_MTF";
   req.type_filling = filling;

   if(!OrderSend(req, res))
      Print("ERROR: OrderSend failed. Code=", GetLastError(),
            " Retcode=", res.retcode);
   else
      Log("Order sent. Ticket=" + IntegerToString((int)res.order) +
          " Retcode=" + IntegerToString(res.retcode));
}

//+------------------------------------------------------------------+
//| Check for an existing EA position                                |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(!PositionSelectByTicket(ticket)) continue;
      if(PositionGetString(POSITION_SYMBOL)  == SYMBOL &&
         PositionGetInteger(POSITION_MAGIC)  == MagicNumber)
         return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Session filter – London 07:00-16:00 + New York 13:00-22:00 UTC  |
//+------------------------------------------------------------------+
bool IsActiveSession()
{
   MqlDateTime dt;
   TimeToStruct(TimeGMT(), dt);
   int h = dt.hour;
   return (h >= 7 && h < 16) ||   // London
          (h >= 13 && h < 22);    // New York
}

//+------------------------------------------------------------------+
//| Read first buffer value from indicator handle at given bar shift |
//+------------------------------------------------------------------+
double GetBuffer(int handle, int shift)
{
   double buf[];
   ArraySetAsSeries(buf, true);
   if(CopyBuffer(handle, 0, shift, 1, buf) <= 0)
      return 0.0;
   return buf[0];
}

//+------------------------------------------------------------------+
//| Human-readable candle pattern name                               |
//+------------------------------------------------------------------+
string PatternName(int pattern)
{
   switch(pattern)
   {
      case 1:  return "PinBar";
      case 2:  return "Engulfing";
      default: return "None";
   }
}

//+------------------------------------------------------------------+
//| Log trade features to CSV file for ML training                   |
//+------------------------------------------------------------------+
void LogTradeFeature(string signal, double h1_strength, double m5_gap,
                     double atr, int pattern, int score,
                     double entry, double sl, double tp,
                     double ai_confidence, string ai_reason)
{
   string fname = "GoldScalper_ML_Log.csv";
   int    fh    = FileOpen(fname, FILE_WRITE | FILE_READ | FILE_CSV | FILE_ANSI, ',');
   if(fh == INVALID_HANDLE)
   {
      Print("WARNING: Cannot open ML log file.");
      return;
   }

   FileSeek(fh, 0, SEEK_END);
   string row = TimeToString(TimeCurrent(), TIME_DATE | TIME_MINUTES) + "," +
                signal + "," +
                DoubleToString(h1_strength, 5) + "," +
                DoubleToString(m5_gap, 5) + "," +
                DoubleToString(atr, 5) + "," +
                IntegerToString(pattern) + "," +
                IntegerToString(score) + "," +
                DoubleToString(entry, _Digits) + "," +
                DoubleToString(sl, _Digits) + "," +
                DoubleToString(tp, _Digits) + "," +
                DoubleToString(ai_confidence, 2) + "," +
                "\"" + ai_reason + "\"";
   FileWriteString(fh, row + "\n");
   FileClose(fh);
}

//+------------------------------------------------------------------+
//| Conditional logger                                               |
//+------------------------------------------------------------------+
void Log(const string msg)
{
   if(EnableLogging)
      Print("[GoldScalper_MTF] ", msg);
}
//+------------------------------------------------------------------+
