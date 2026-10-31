#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Called this object, default address 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

// Servo configuration for MG996R
#define US_MIN    600   // Safe minimum pulse width in microseconds (~0 degrees)
#define US_MAX    2400  // Safe maximum pulse width in microseconds (~180 degrees)
#define SERVO_FREQ 50   // Analog servos run at ~50 Hz updates

// Potentiometer pin on ESP32
const int potPin = 34;

// Current pulse width in microseconds
int currentPulse = 1500; // Start at neutral (approx 90 degrees)

void setup() {
  Serial.begin(115200);
  while (!Serial);
  
  Serial.println("--- MG996R Servo PWM Serial Control ---");
  Serial.println("Type any pulse width value in microseconds (e.g., 1000, 1500, 2000) and press Enter:");
  
  pwm.begin();
  pwm.setOscillatorFrequency(27000000);
  pwm.setPWMFreq(SERVO_FREQ);

  // Configure ADC for ESP32
  pinMode(potPin, INPUT);
  analogSetAttenuation(ADC_11db); // Full range: 0 - 3.3V
  
  delay(10);
}

void loop() {
  // 1. Read potentiometer value for telemetry
  int potValue = analogRead(potPin);
  int potMapped = map(potValue, 0, 4095, 0, 180);

  // 2. Handle direct PWM input from Serial Monitor
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    
    if (input.length() > 0) {
      int targetPulse = input.toInt();
      
      // Validate that the input is a valid number
      if (targetPulse > 0) {
        // Constrain within safe MG996R hardware limits to prevent stalling/damage
        currentPulse = constrain(targetPulse, US_MIN, US_MAX);
        
        Serial.print("-> PWM Set to: ");
        Serial.print(currentPulse);
        Serial.println(" us");
      } else {
        Serial.println("Invalid input. Enter a number between 600 and 2400.");
      }
    }
  }

  // 3. Write pulse width to the PCA9685 (Channel 0)
  pwm.writeMicroseconds(0, currentPulse);

  // 4. Print telemetry for reference
  Serial.print("Pot Raw: ");
  Serial.print(potValue);
  Serial.print(" | Pot Angle: ");
  Serial.print(potMapped);
  Serial.print("° | Current Servo PWM: ");
  Serial.print(currentPulse);
  Serial.println(" us");

  delay(200); // Small delay to keep the serial stream readable
}