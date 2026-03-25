#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <OneWire.h>      
#include <DallasTemperature.h>


// ====================== USER SETTINGS ======================
#define USE_SIM_MODE 1          // 1 = simulate sensors (works now). Set 0 when hardware arrives.

const char* WIFI_SSID = "BraLogger_AP";
const char* WIFI_PASS = "12345678"; // >= 8 chars

// DS18B20 (4 sensors on one wire)
static const uint8_t ONE_WIRE_PIN = 4;

// Pressure mux (CD74HC4067)
static const uint8_t MUX_S0 = 14;
static const uint8_t MUX_S1 = 27;
static const uint8_t MUX_S2 = 26;
static const uint8_t MUX_S3 = 25;
static const uint8_t MUX_SIG_ADC = 34;   // ADC input (ESP32 input-only pins are good)

// Sampling
static const uint32_t SAMPLE_INTERVAL_MS = 500; // 2 Hz
// ===========================================================

WebServer server(80);

// DS18B20 setup
OneWire oneWire(ONE_WIRE_PIN);
DallasTemperature dallas(&oneWire);

DeviceAddress tempAddr[4];
bool tempFound[4] = {false, false, false, false};

// Last sample buffer
float tempsC[4] = {NAN, NAN, NAN, NAN};
uint16_t pressRaw[8] = {0};

uint32_t lastSampleMs = 0;

// ---------------------- Helpers ----------------------
void muxSelectChannel(uint8_t channel) {
  // channel: 0..15
  digitalWrite(MUX_S0, channel & 0x01);
  digitalWrite(MUX_S1, (channel >> 1) & 0x01);
  digitalWrite(MUX_S2, (channel >> 2) & 0x01);
  digitalWrite(MUX_S3, (channel >> 3) & 0x01);
  delayMicroseconds(5); // settle
}

uint16_t readPressureChannel(uint8_t ch0to7) {
  muxSelectChannel(ch0to7);
  // ESP32 ADC: 0..4095 (12-bit) by default
  return (uint16_t)analogRead(MUX_SIG_ADC);
}

float simTemp(uint8_t i, float t) {
  // Simple synthetic patterns: baseline + slow drift + small difference per sensor
  return 34.0f + 0.2f * sinf(t / 20.0f) + 0.05f * i + 0.1f * sinf((t + i * 3) / 7.0f);
}

uint16_t simPressure(uint8_t i, float t) {
  // Simulated contact changes
  float base = 1400 + 120 * sinf(t / 8.0f);
  float per = 30 * i + 40 * sinf((t + i * 2) / 5.0f);
  float noise = 20 * sinf((t + i) * 1.7f);
  float v = base + per + noise;
  if (v < 0) v = 0;
  if (v > 4095) v = 4095;
  return (uint16_t)v;
}

// ---------------------- Sampling ----------------------
void sampleSensors() {
  const float t = millis() / 1000.0f;

#if USE_SIM_MODE
  for (int i = 0; i < 4; i++) tempsC[i] = simTemp(i, t);
  for (int i = 0; i < 8; i++) pressRaw[i] = simPressure(i, t);
#else
  // Temperatures
  dallas.requestTemperatures();
  for (int i = 0; i < 4; i++) {
    if (tempFound[i]) {
      float c = dallas.getTempC(tempAddr[i]);
      tempsC[i] = c; // if disconnected, library returns DEVICE_DISCONNECTED_C
    } else {
      tempsC[i] = NAN;
    }
  }

  // Pressures
  for (int i = 0; i < 8; i++) {
    pressRaw[i] = readPressureChannel(i);
  }
#endif
}

// ---------------------- HTTP Handlers ----------------------
String jsonNow() {
  // Minimal JSON payload for logging
  // You can add IMU/ambient later without changing your receiver much.
  uint32_t ms = millis();

  String s = "{";
  s += "\"ts_ms\":" + String(ms) + ",";
  s += "\"temps_c\":[";

  for (int i = 0; i < 4; i++) {
    if (i) s += ",";
    if (isnan(tempsC[i])) s += "null";
    else s += String(tempsC[i], 3);
  }
  s += "],";

  s += "\"press_raw\":[";
  for (int i = 0; i < 8; i++) {
    if (i) s += ",";
    s += String(pressRaw[i]);
  }
  s += "]";

  s += "}";
  return s;
}

void handleRoot() {
  String html =
    "<html><body>"
    "<h2>Bra Logger (ESP32)</h2>"
    "<p>Endpoints:</p>"
    "<ul>"
    "<li><a href=\"/data\">/data</a> - latest sample JSON</li>"
    "<li><a href=\"/health\">/health</a> - status</li>"
    "</ul>"
    "</body></html>";
  server.send(200, "text/html", html);
}

void handleHealth() {
  String s = "{";
  s += "\"wifi_ssid\":\"" + String(WIFI_SSID) + "\",";
  s += "\"ip\":\"" + WiFi.softAPIP().toString() + "\",";
  s += "\"sim_mode\":" + String((int)USE_SIM_MODE);
  s += "}";
  server.send(200, "application/json", s);
}

void handleData() {
  server.send(200, "application/json", jsonNow());
}

// ---------------------- Setup ----------------------
void setup() {
  Serial.begin(115200);

  // MUX pins
  pinMode(MUX_S0, OUTPUT);
  pinMode(MUX_S1, OUTPUT);
  pinMode(MUX_S2, OUTPUT);
  pinMode(MUX_S3, OUTPUT);

  // ADC pin: no pinMode needed on ESP32, but ok to omit.

#if !USE_SIM_MODE
  dallas.begin();
  // Discover up to 4 DS18B20 sensors on the bus
  int count = dallas.getDeviceCount();
  Serial.printf("DS18B20 found: %d\n", count);

  // Try to get 4 unique addresses
  for (int i = 0; i < 4; i++) {
    tempFound[i] = dallas.getAddress(tempAddr[i], i);
    Serial.printf("Temp sensor %d address %s\n", i, tempFound[i] ? "OK" : "NOT FOUND");
  }

  // Optional: set resolution
  dallas.setResolution(12); // 9..12 bits
#endif

  // Start Wi-Fi AP (WiFi direct)
  WiFi.mode(WIFI_AP);
  WiFi.softAP(WIFI_SSID, WIFI_PASS);
  IPAddress ip = WiFi.softAPIP();
  Serial.print("AP IP: ");
  Serial.println(ip);

  // HTTP routes
  server.on("/", handleRoot);
  server.on("/health", handleHealth);
  server.on("/data", handleData);
  server.begin();

  // Initial sample
  sampleSensors();
  lastSampleMs = millis();
}

// ---------------------- Loop ----------------------
void loop() {
  server.handleClient();

  uint32_t now = millis();
  if (now - lastSampleMs >= SAMPLE_INTERVAL_MS) {
    lastSampleMs = now;
    sampleSensors();
  }
}