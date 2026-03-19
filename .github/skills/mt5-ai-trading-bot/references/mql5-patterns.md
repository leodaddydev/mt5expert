# MQL5 Patterns — Reference

## Indicator Lifecycle

```mql5
// OnInit: create handles
int g_hEMA20 = iMA(_Symbol, PERIOD_M5, 20, 0, MODE_EMA, PRICE_CLOSE);
if(g_hEMA20 == INVALID_HANDLE) { Print("Handle failed"); return INIT_FAILED; }

// OnDeinit: always release
IndicatorRelease(g_hEMA20);
```

## Reading Buffer Values

```mql5
double GetBuffer(int handle, int shift) {
    double buf[];
    ArraySetAsSeries(buf, true);
    if(CopyBuffer(handle, 0, shift, 1, buf) <= 0) return 0.0;
    return buf[0];
}
// shift=0 is current (forming) bar, shift=1 is last CLOSED bar → use 1 for signals
```

## Screenshot + Base64 Encode

```mql5
// Capture
ChartScreenShot(ChartID(), "snapshot.png", 1280, 720, ALIGN_LEFT);

// Read file
int fh = FileOpen("snapshot.png", FILE_READ | FILE_BIN);
ulong sz = FileSize(fh);
uchar bytes[]; ArrayResize(bytes, (int)sz);
FileReadArray(fh, bytes, 0, (int)sz);
FileClose(fh);

// Base64 encode
uchar key[], encoded[];
CryptEncode(CRYPT_BASE64, bytes, key, encoded);
string b64 = CharArrayToString(encoded, 0, WHOLE_ARRAY, CP_UTF8);
```

## WebRequest POST

```mql5
string headers = "Content-Type: application/json\r\n";
char postData[], responseData[];
string response;
StringToCharArray(payload, postData, 0, StringLen(payload), CP_UTF8);
// Remove null terminator
int len = ArraySize(postData);
if(len > 0 && postData[len-1] == 0) ArrayResize(postData, len - 1);

int code = WebRequest("POST", url, headers, timeout_ms, postData, responseData, response);
// code == 200 → success
// code == -1 → whitelist URL in Tools → Options → Expert Advisors
```

## Simple JSON Extraction (no library)

```mql5
string ExtractStringField(const string json, const string key) {
    string search = "\"" + key + "\":\"";
    int pos = StringFind(json, search);
    if(pos < 0) return "";
    pos += StringLen(search);
    int end = StringFind(json, "\"", pos);
    return StringSubstr(json, pos, end - pos);
}

double ExtractDoubleField(const string json, const string key) {
    string search = "\"" + key + "\":";
    int pos = StringFind(json, search);
    if(pos < 0) return 0.0;
    pos += StringLen(search);
    int end = pos;
    while(end < StringLen(json)) {
        ushort ch = StringGetCharacter(json, end);
        if(ch == ',' || ch == '}' || ch == ' ') break;
        end++;
    }
    return StringToDouble(StringSubstr(json, pos, end - pos));
}
```

## OHLC Copy (for sending to server)

```mql5
MqlRates rates[];
ArraySetAsSeries(rates, true);
int copied = CopyRates(_Symbol, PERIOD_M5, 0, N_BARS, rates);
// Iterate from oldest to newest: for(int i = copied-1; i >= 0; i--)
```

## Order Execution with Dynamic Filling

```mql5
ENUM_ORDER_TYPE_FILLING filling;
long fillMode = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
if((fillMode & SYMBOL_FILLING_FOK) != 0) filling = ORDER_FILLING_FOK;
else if((fillMode & SYMBOL_FILLING_IOC) != 0) filling = ORDER_FILLING_IOC;
else filling = ORDER_FILLING_RETURN;

MqlTradeRequest req = {};
MqlTradeResult  res = {};
req.action = TRADE_ACTION_DEAL; req.symbol = _Symbol;
req.type   = ORDER_TYPE_BUY;    req.volume = lots;
req.price  = entry; req.sl = sl; req.tp = tp;
req.deviation = 10; req.magic = MagicNumber;
req.type_filling = filling;
if(!OrderSend(req, res)) Print("Error: ", GetLastError());
```

## New Bar Detection

```mql5
datetime g_LastBarTime = 0;
void OnTick() {
    datetime currentBar = iTime(_Symbol, PERIOD_M5, 0);
    if(currentBar == g_LastBarTime) return;  // same bar, skip
    g_LastBarTime = currentBar;
    // ... process new bar close
}
```

## Session Filter (UTC-based)

```mql5
bool IsActiveSession() {
    MqlDateTime dt; TimeToStruct(TimeGMT(), dt);
    int h = dt.hour;
    return (h >= 7 && h < 16) ||   // London
           (h >= 13 && h < 22);    // New York
}
```

## Fractal S/R Detection

```mql5
// Bar i is a fractal high if higher than bar i-1 and bar i+1
if(highs[i] > highs[i-1] && highs[i] > highs[i+1])  // → resistance
if(lows[i]  < lows[i-1]  && lows[i]  < lows[i+1])   // → support
// Always ArraySetAsSeries(highs, true) first
```

## Risk-Based Lot Sizing

```mql5
double CalcLotSize(double entry, double sl) {
    double risk    = AccountInfoDouble(ACCOUNT_BALANCE) * RiskPercent / 100.0;
    double slPts   = MathAbs(entry - sl) / _Point;
    double tickVal = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double lots    = risk / (slPts * tickVal);
    // Clamp to broker limits
    lots = MathFloor(lots / lotStep) * lotStep;
    return MathMax(minLot, MathMin(maxLot, lots));
}
```

## MQL5 Compilation Notes

- Always whitelist the server URL in MetaEditor → Tools → Options → Expert Advisors → Allow WebRequest
- Files saved by the EA go to `MQL5\Files\` under the terminal data path
- `TerminalInfoString(TERMINAL_DATA_PATH)` returns the root data folder path
