#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Initialize PCA9685
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Servo configuration (Adjust these limits based on your MG996R safe range)
#define US_MIN     50
#define US_MAX     600
#define SERVO_FREQ  50  // 50Hz for analog servos

// Potentiometer connected to ESP32 ADC pin
const int potPin = 34;

// Default starting pulse width in microseconds (Neutral ~90°)
int currentPulse = 102; 

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("--- MG996R Direct PWM & Potentiometer Tool ---");
  Serial.println("Type a PWM value in microseconds (e.g., 1000, 1500, 2000) and press Enter:");

  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);

  // Configure potentiometer pin
  pinMode(potPin, INPUT);
  analogSetAttenuation(ADC_11db); // Full range: 0 - 3.3V
}

void loop() {
  // 1. Check for keyboard input from Serial Monitor
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (input.length() > 0) {
      int targetPulse = input.toInt();
      if (targetPulse > 0) {
        // Constrain to safe hardware limits
        currentPulse = constrain(targetPulse, US_MIN, US_MAX);
        
        // Command the PCA9685 channel 0
        pwm.setPWM(1, 0, currentPulse);
        
        Serial.print("-> Set PWM to: ");
        Serial.print(currentPulse);
        Serial.println(" us");
      }
    }
  }

  // 2. Read the raw potentiometer value
  int rawPotValue = analogRead(potPin);

  // 3. Output current status
  Serial.print("Current Servo PWM: ");
  Serial.print(currentPulse);
  Serial.print(" us | Raw Potentiometer Value: ");
  Serial.println(rawPotValue);

  delay(250); // Delay to keep the serial stream stable and readable
}