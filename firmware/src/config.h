// ============================================================
// config.h  —  BRA Logger Firmware Configuration
// ============================================================
// Edit this file ONLY — do not scatter magic numbers in .cpp files.
// ============================================================

#pragma once

// ── Wi-Fi Access Point ────────────────────────────────────────
#define WIFI_SSID "BraLogger_AP"
#define WIFI_PASS "12345678"          // >= 8 chars (WPA2 requirement)

// ── Simulation mode ───────────────────────────────────────────
// 1 = generate synthetic sensor data (no hardware needed for testing)
// 0 = read real sensors (DS18B20, FSR via MUX, AD5933, BME280, MPU6050)
#define USE_SIM_MODE 1

// ── Sampling ─────────────────────────────────────────────────
#define SAMPLE_INTERVAL_MS 500        // 2 Hz

// ── DS18B20 Temperature (1-Wire) ─────────────────────────────
#define ONE_WIRE_PIN     4
#define NUM_TEMP_SENSORS 4            // 2 left breast + 2 right breast
#define DS18B20_RESOLUTION 12         // 9, 10, 11, or 12 bits

// ── FSR Pressure via CD74HC4067 MUX ──────────────────────────
#define MUX_S0          14
#define MUX_S1          27
#define MUX_S2          26
#define MUX_S3          25
#define MUX_SIG_ADC     34            // ADC input pin (input-only, fine)
#define NUM_PRESSURE_CH  8            // 4 left breast + 4 right breast

// ── AD5933 Bioimpedance (I2C) ─────────────────────────────────
#define I2C_SDA         21
#define I2C_SCL         22
#define NUM_IMPEDANCE_CH 4            // 2 left breast electrode pairs + 2 right
// (AD5933 has one channel; multiplexing electrode pairs is done in firmware)

// ── BME280 Ambient Sensor (I2C, shared bus) ───────────────────
// Uses same I2C bus as AD5933 — different I2C address (0x76 or 0x77)
#define BME280_ADDR     0x76

// ── MPU6050 IMU (I2C, shared bus) ────────────────────────────
// Used to detect motion artifacts
#define MPU6050_ADDR    0x68

// ── HTTP Server ───────────────────────────────────────────────
#define HTTP_PORT       80
