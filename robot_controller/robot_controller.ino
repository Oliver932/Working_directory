#include <AccelStepper.h>

//---------------------------------------------------------------------------------
// Hardware & Library Setup
//---------------------------------------------------------------------------------

// Create stepper motor objects using the AccelStepper library.
// The format is AccelStepper(interface, step_pin, direction_pin).
// AccelStepper::DRIVER signifies a standard step/dir driver (e.g., A4988, DRV8825).
AccelStepper stepper1(AccelStepper::DRIVER, 2, 3);
AccelStepper stepper2(AccelStepper::DRIVER, 4, 5);
AccelStepper stepper3(AccelStepper::DRIVER, 6, 7);
AccelStepper stepper4(AccelStepper::DRIVER, 8, 9);

// Define the Arduino pins connected to the limit switches for each axis.
const int homeSwitch1 = A0;
const int homeSwitch2 = A1;
const int homeSwitch3 = A2;
const int homeSwitch4 = A3;


//---------------------------------------------------------------------------------
// Motion & Position Variables
//---------------------------------------------------------------------------------

// Variables to hold the target position (in steps) for each motor.
long targetPosition1 = 0;
long targetPosition2 = 0;
long targetPosition3 = 0;
long targetPosition4 = 0;

// Set the default maximum speed and acceleration for normal movements.
// Speed is in steps per second.
// Acceleration is in steps per second per second.
long maxSpeed = 4500;
long accel = 8000;

//---------------------------------------------------------------------------------
// Homing Parameters & State Flags
//---------------------------------------------------------------------------------

// Define parameters specifically for the homing sequence for safety and precision.
const int homingSpeed = 1000;         // Slower speed for homing.
const int homingAcceleration = 2000;  // Gentler acceleration for homing.
const int backoffSteps = 200;         // Steps to move away from the switch after homing.

// Boolean flags to manage the machine's state.
bool isHomed = false;          // True if the machine has been successfully homed.
bool homingInProgress = false; // True while the homing sequence is active.
bool motor1Homed = false;      // Individual flags to track each motor's homing status.
bool motor2Homed = false;
bool motor3Homed = false;
bool motor4Homed = false;

// Flag to track if a coordinate move command is currently being executed.
bool moveInProgress = false;

//---------------------------------------------------------------------------------
// End Effector (Gripper) Variables
//---------------------------------------------------------------------------------

// Variables to manage the timed operation of the end effector (e.g., a gripper).
unsigned long triggerStartTime = 0;           // Records when the gripper action started.
const unsigned long triggerDuration = 5000;   // Duration (in ms) to keep the gripper motor active.

/**
 * @brief Runs once at startup to initialize hardware and software settings.
 */
void setup() {
  // Start serial communication for receiving commands and sending feedback.
  Serial.begin(115200);
  pinMode(13, OUTPUT); // Onboard LED, can be used for status indication.

  // Apply the default speed and acceleration settings to each stepper motor.
  stepper1.setMaxSpeed(maxSpeed);
  stepper1.setAcceleration(accel);
  stepper2.setMaxSpeed(maxSpeed);
  stepper2.setAcceleration(accel);
  stepper3.setMaxSpeed(maxSpeed);
  stepper3.setAcceleration(accel);
  stepper4.setMaxSpeed(maxSpeed);
  stepper4.setAcceleration(accel);

  // Configure the limit switch pins as inputs with internal pull-up resistors.
  // The switch should be wired to connect the pin to GND when pressed.
  pinMode(homeSwitch1, INPUT_PULLUP);
  pinMode(homeSwitch2, INPUT_PULLUP);
  pinMode(homeSwitch3, INPUT_PULLUP);
  pinMode(homeSwitch4, INPUT_PULLUP);

  // Initialize the pins for the End Effector (EE) motor as outputs.
  pinMode(10, OUTPUT);
  pinMode(11, OUTPUT);

  // Ensure the EE motor pins start in the LOW (off) state.
  digitalWrite(10, LOW);
  digitalWrite(11, LOW);
}

/**
 * @brief Initiates the multi-axis homing sequence.
 * This function sets the homing flags and parameters, then commands all motors
 * to move towards their respective limit switches.
 */
void startHoming() {
  Serial.println ("Homing Now");
  homingInProgress = true;
  motor1Homed = motor2Homed = motor3Homed = motor4Homed = false;
  
  // Temporarily set slower, safer motion parameters for the homing process.
  stepper1.setMaxSpeed(homingSpeed);
  stepper1.setAcceleration(homingAcceleration);
  stepper2.setMaxSpeed(homingSpeed);
  stepper2.setAcceleration(homingAcceleration);
  stepper3.setMaxSpeed(homingSpeed);
  stepper3.setAcceleration(homingAcceleration);
  stepper4.setMaxSpeed(homingSpeed);
  stepper4.setAcceleration(homingAcceleration);

  // Command all motors to move to a large negative position. This ensures they
  // travel towards the home switches regardless of their starting position.
  stepper1.moveTo(-100000);
  stepper2.moveTo(-100000);
  stepper3.moveTo(-100000);
  stepper4.moveTo(-100000);
}

/**
 * @brief Manages the active homing process.
 * This function is called repeatedly from the main loop while homing is in progress.
 * It checks each motor's limit switch and stops it upon contact, then performs
 * a final back-off maneuver once all axes are homed.
 */
void handleHoming() {
  // Check each motor's limit switch individually.
  if (!motor1Homed) {
    if (digitalRead(homeSwitch1) == LOW) { // Switch is pressed
      stepper1.stop();
      stepper1.setCurrentPosition(0); // Set this position as the temporary zero
      motor1Homed = true;
      Serial.println("Motor 1 homed");
    } else {
      stepper1.run(); // Continue moving towards the switch
    }
  }

  if (!motor2Homed) {
    if (digitalRead(homeSwitch2) == LOW) {
      stepper2.stop();
      stepper2.setCurrentPosition(0);
      motor2Homed = true;
      Serial.println("Motor 2 homed");
    } else {
      stepper2.run();
    }
  }

  if (!motor3Homed) {
    if (digitalRead(homeSwitch3) == LOW) {
      stepper3.stop();
      stepper3.setCurrentPosition(0);
      motor3Homed = true;
      Serial.println("Motor 3 homed");
    } else {
      stepper3.run();
    }
  }

  if (!motor4Homed) {
    if (digitalRead(homeSwitch4) == LOW) {
      stepper4.stop();
      stepper4.setCurrentPosition(0);
      motor4Homed = true;
      Serial.println("Motor 4 homed");
    } else {
      stepper4.run();
    }
  }

  // This block executes only after ALL motors have hit their switches.
  if (motor1Homed && motor2Homed && motor3Homed && motor4Homed) {
    // Command all motors to back off from the switches by a small amount.
    stepper1.moveTo(backoffSteps);
    stepper2.moveTo(backoffSteps);
    stepper3.moveTo(backoffSteps);
    stepper4.moveTo(backoffSteps);

    // This is a short blocking loop to ensure the back-off move is complete.
    while (stepper1.isRunning() || stepper2.isRunning() || 
           stepper3.isRunning() || stepper4.isRunning()) {
      stepper1.run();
      stepper2.run();
      stepper3.run();
      stepper4.run();
    }

    // After backing off, reset the current position to 0 for all axes.
    // This is the true, final home position.
    stepper1.setCurrentPosition(0);
    stepper2.setCurrentPosition(0);
    stepper3.setCurrentPosition(0);
    stepper4.setCurrentPosition(0);

    // Finalize the homing sequence by updating state flags.
    homingInProgress = false;
    isHomed = true;

    Serial.println("DONE"); // Signal that the machine is ready for commands.
  }
}

/**
 * @brief The main program loop, which runs continuously.
 * It listens for serial commands, manages machine states (homing vs. moving),
 * controls the end effector timer, and calls the run() methods for the steppers.
 */
void loop() {
  unsigned long currentTime = millis(); // Get current time for timer logic.

  // --- 1. Command Parsing ---
  // Check if there is a new command from the serial port.
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim(); // Remove any whitespace.

    // --- Gripper Commands ---
    if (command == "GRIP") {
      digitalWrite(10, HIGH);
      digitalWrite(11, LOW);
      triggerStartTime = currentTime; // Start the 5-second timer.
      Serial.println("CLAMPING IN PROGRESS");
    } else if (command == "UNGRIP") {
      digitalWrite(10, LOW);
      digitalWrite(11, HIGH);
      triggerStartTime = currentTime; // Start the 5-second timer.
      Serial.println("RELEASING IN PROGRESS");
    
    // --- Homing Command ---
    } else if (command == "HOME" && !homingInProgress) {
      startHoming();
    
    // --- Coordinate Move Command ---
    } else if (command != "0,0,0,0") {
      // Find the positions of the commas to parse the coordinate string.
      int comma1 = command.indexOf(',');
      int comma2 = command.indexOf(',', comma1 + 1);
      int comma3 = command.indexOf(',', comma2 + 1);
      
      // Ensure the command format is valid (contains three commas).
      if (comma1 == -1 || comma2 == -1 || comma3 == -1) {
        Serial.println("Invalid input format");
      }
      else {
        // Parse the string into integer positions for each motor.
        targetPosition1 = command.substring(0, comma1).toInt();
        targetPosition2 = command.substring(comma1 + 1, comma2).toInt();
        targetPosition3 = command.substring(comma2 + 1, comma3).toInt();
        targetPosition4 = command.substring(comma3 + 1).toInt();

        // Set the new target positions for each motor.
        stepper1.moveTo(targetPosition1);
        stepper2.moveTo(targetPosition2);
        stepper3.moveTo(targetPosition3);
        stepper4.moveTo(targetPosition4);

        // A new move has started, so set the flag to true.
        moveInProgress = true;
      }
    } else {
      Serial.println("Invalid command received");
    }
  }

  // --- 2. State Handling ---
  // If a homing sequence is active, continue processing it.
  if (homingInProgress) {
    handleHoming();
  }

  // --- 3. End Effector Timer ---
  // Check if the 5-second gripper duration has elapsed since it was triggered.
  if (triggerStartTime > 0 && (currentTime - triggerStartTime >= triggerDuration)) {
    digitalWrite(10, LOW); // Turn off the gripper motor.
    digitalWrite(11, LOW);
    triggerStartTime = 0; // Reset the timer start time.
    Serial.println("DONE");
  }

  // --- 4. Move Completion Check ---
  // If a coordinate move is in progress, check if all motors have reached their destination.
  if (moveInProgress) {
    // distanceToGo() returns 0 only when the motor has arrived at its target.
    if (stepper1.distanceToGo() == 0 &&
        stepper2.distanceToGo() == 0 &&
        stepper3.distanceToGo() == 0 &&
        stepper4.distanceToGo() == 0) {
      
      Serial.println("DONE"); // Send completion signal.
      moveInProgress = false; // Reset the flag, ready for the next move command.
    }
  }

  // --- 5. Stepper Execution ---
  // This section must be called as often as possible. The .run() function
  // calculates and executes steps based on the current speed and acceleration.
  // It only runs the motors if the machine has been homed.
  // if (isHomed) {
  //   stepper1.run();
  //   stepper2.run();
  //   stepper3.run();
  //   stepper4.run();
  // }

  stepper1.run();
  stepper2.run();
  stepper3.run();
  stepper4.run();
}