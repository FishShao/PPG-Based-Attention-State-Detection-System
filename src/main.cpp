#include <Arduino.h>

// Wiring (Seeed XIAO ESP32-C3):
//   Sensor A0  →  XIAO A0 (GPIO2)
//   Sensor VCC →  XIAO 3V3
//   Sensor GND →  XIAO GND
#define PPG_PIN     2   // A0 = GPIO2 on XIAO ESP32-C3
#define SAMPLE_HZ   256
#define INTERVAL_US (1000000 / SAMPLE_HZ)

void setup() {
  Serial.begin(57600);   // 57600 to match your previous setup
  delay(500);
  analogReadResolution(12);        // ESP32: 12-bit = 0–4095
  analogSetAttenuation(ADC_11db);  // full 3.3V range
  // No CSV header — output single float per line (matches your old Python reader)
}

void loop() {
  static uint32_t lastTime = 0;
  uint32_t now = micros();

  if (now - lastTime >= INTERVAL_US) {
    lastTime = now;
    // Scale 12-bit (0-4095) → 16-bit range (0-65535) to match CircuitPython values
    int raw = analogRead(PPG_PIN);
    int scaled = raw * 16;  // 4095*16 = 65520 ≈ 65535
    Serial.println(scaled);
  }
}