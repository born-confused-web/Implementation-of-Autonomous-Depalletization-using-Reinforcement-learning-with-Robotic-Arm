#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>


Adafruit_PWMServoDriver PWM = Adafruit_PWMServoDriver();

const int total_joints = 7;
const int feedback_joints = 4;
const float max_down_velocity = 40; //degree/s
const int down_control_pins[total_joints] = {0, -1, 1, 0, 0, 0, 0};
const int feedbackpins[feedback_joints] = {32,33,34,35};
const float homepos[total_joints] = {90.0, 180.0, 180.0, 90.0, 180.0, 90.0, 90.0};
float currentpos[feedback_joints];
String inputBuffer = "";

float readpot(int i){
  int rawADC = analogRead(i);
  Serial.println(rawADC);
  float rawangle = map(rawADC, 45, 3571, 0, 180);
  return rawangle;
}

void get_current_position(float* arr, const int* pins){
  for(int i=0; i<feedback_joints;i++){
      arr[i] = readpot(pins[i]);
  }
}

void printcurrentpos(float* pos){
  Serial.print("currentpos: ");
  for(int i=0; i<feedback_joints; i++){
    Serial.print(pos[i]);
    Serial.print(", ");
  }
  Serial.println(".");
}

uint16_t angletopwm(float angle){
  int pulse = map((long)(angle*100), 0, 18000, 100, 500);
  return constrain(pulse, 100, 500);
}

void move_arm(const float* target){
  get_current_position(currentpos, feedbackpins);
  float DT = 1000/max_down_velocity;
  for(int i=0; i<total_joints; i++){
    float dist = abs(target[i]-currentpos[i]);
    if(down_control_pins[i]!=0){
      if((down_control_pins[i]*(target[i]-currentpos[i]))>0){
        while((down_control_pins[i]*abs(target[i]-currentpos[i]))>0){
          PWM.setPWM(i, 0, angletopwm(currentpos[i]));
          delay(DT);
          currentpos[i]+=(float)down_control_pins[i];
        }
      }
      else{
        while((down_control_pins[i]*(target[i]-currentpos[i]))<0 && dist>60){
          PWM.setPWM(i, 0, angletopwm(currentpos[i]));
          delay(DT*max_down_velocity);
          currentpos[i]-=(float)down_control_pins[i]*max_down_velocity;
        }
      }
    }
    PWM.setPWM(i, 0, angletopwm(target[i]));
  }
}

void gohome(){
  move_arm(homepos);
  Serial.println("reached home");
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  PWM.begin();
  PWM.setPWMFreq(50);
  analogReadResolution(12);
  gohome();
  delay(5000);
  inputBuffer.reserve(128);
}

void loop() {
  while (Serial.available()>0){
    char c = Serial.read();

    if (c=='\n'){
      processLine(inputBuffer);
      inputBuffer = "";
    }
    else{
      inputBuffer += c;
    }
  }
}

void processLine(const String &line){
  float angle[total_joints];
  int count = 0;

  int startIdx = 0;
  for(int i=0; i <= line.length(); i++){
    if(i==line.length() || line.charAt(i)==','){
      if (count < total_joints){
        String token = line.substring(startIdx, i);
        angle[i] = token.toFloat();
        count++;
      }
      startIdx = i + 1;
    }
  }

  if(count==total_joints){
    move_arm(angle);
  }
}