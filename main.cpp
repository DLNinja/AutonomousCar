#include <Arduino.h>
#include <Wire.h>
#define pwm1 6
#define dira1 2
#define dira2 3
#define pwm2 10
#define dirb1 4
#define dirb2 5

int nr[3], x, val = 125;

void receiveEvent(int howMany) {
  int k = 0;
  while (Wire.available()) {

    nr[k] = Wire.read();
    k++;
  }
  x = map(nr[2], 0, 224, 0, val);
  if (x)
    go(val + x, val - x);
  else
    go(0, 0);
  delay(100);
}


void go(int speedLeft, int speedRight) {
  if (speedLeft > 0)
  {
    digitalWrite(dira1, HIGH);
    digitalWrite(dira2, LOW);
    analogWrite(pwm1, speedLeft);
  }
  else
  {
    digitalWrite(dira2, HIGH);
    digitalWrite(dira1, LOW);
    analogWrite(pwm1, -speedLeft);
  }
  if (speedRight > 0)
  {
    digitalWrite(dirb1, HIGH);
    digitalWrite(dirb2, LOW);
    analogWrite(pwm2, speedRight);
  }
  else
  {
    digitalWrite(dirb2, HIGH);
    digitalWrite(dirb1, LOW);
    analogWrite(pwm2, -speedRight);
  }

}


void setup() {
  Wire.begin(8);                // join i2c bus with address #8
  Wire.onReceive(receiveEvent); // register event
  // Serial.begin(9600);           // start serial for output
}

void loop() {

}
