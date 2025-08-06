#include <AccelStepper.h>
#include <MultiStepper.h>
#include <Bounce2.h>

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

// MultiStepper object for coordinated multi-axis movements
MultiStepper steppers;

// Define the Arduino pins connected to the limit switches for each axis.
const int NUM_MOTORS = 4;
const int homeSwitchPins[NUM_MOTORS] = {A0, A1, A2, A3};

// Bounce2 objects for debounced switch reading
Bounce homeSwitches[NUM_MOTORS];


//---------------------------------------------------------------------------------
// Motion & Position Variables
//---------------------------------------------------------------------------------

// Set the default maximum speed and acceleration for normal movements.
// Speed is in steps per second.
// Acceleration is in steps per second per second.
const long maxSpeed = 4500;
const long accel = 8000;

// Array to store target positions for all motors (for MultiStepper)
long targetPositions[NUM_MOTORS] = {0, 0, 0, 0};

//---------------------------------------------------------------------------------
// Homing Parameters & State Flags
//---------------------------------------------------------------------------------

// Define parameters specifically for the homing sequence for safety and precision.
const int homingSpeed = 1000;         // Slower speed for homing.
const int homingAcceleration = 2000;  // Gentler acceleration for homing.
const int backoffSteps = 200;         // Steps to move away from the switch after homing.

// Machine state enumeration for better state management
enum MachineState {
  STATE_UNKNOWN,      // Initial state, not yet homed
  STATE_HOMING,       // Currently performing homing sequence
  STATE_READY,        // Homed and ready for commands
  STATE_MOVING,       // Executing a move command
  STATE_EMERGENCY     // Emergency stop activated
};

MachineState currentState = STATE_UNKNOWN;

// Individual motor homing status flags
bool motorHomed[NUM_MOTORS] = {false, false, false, false};

//---------------------------------------------------------------------------------
// Safety & Emergency Stop Variables
//---------------------------------------------------------------------------------

// Emergency stop flag - when true, all motion stops immediately
bool emergencyStop = false;

// Position limits for safety (adjust these based on your machine)
const long MIN_POSITION = -1000;      // Minimum allowed position (steps)
const long MAX_POSITION = 20000;      // Maximum allowed position (steps)

//---------------------------------------------------------------------------------
// End Effector (Gripper) Variables
//---------------------------------------------------------------------------------

// Variables to manage the timed operation of the end effector (e.g., a gripper).
unsigned long triggerStartTime = 0;           // Records when the gripper action started.
const unsigned long triggerDuration = 3000;   // Duration (in ms) to keep the gripper motor active.

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

  // Add all steppers to the MultiStepper object for coordinated movement
  steppers.addStepper(stepper1);
  steppers.addStepper(stepper2);
  steppers.addStepper(stepper3);
  steppers.addStepper(stepper4);

  // Initialize Bounce2 objects for limit switches with debouncing
  for (int i = 0; i < NUM_MOTORS; i++) {
    pinMode(homeSwitchPins[i], INPUT_PULLUP);
    homeSwitches[i].attach(homeSwitchPins[i], INPUT_PULLUP);
    homeSwitches[i].interval(10); // 10ms debounce interval
  }

  // Initialize the pins for the End Effector (EE) motor as outputs.
  pinMode(10, OUTPUT);
  pinMode(11, OUTPUT);

  // Ensure the EE motor pins start in the LOW (off) state.
  digitalWrite(10, LOW);
  digitalWrite(11, LOW);

  Serial.println("Robot Controller Initialized - Send HOME command to begin");
}

//---------------------------------------------------------------------------------
// Utility Functions
//---------------------------------------------------------------------------------

/**
 * @brief Gets a pointer to the stepper motor object by index
 * @param index Motor index (0-3)
 * @return Pointer to AccelStepper object
 */
AccelStepper* getStepper(int index) {
  switch (index) {
    case 0: return &stepper1;
    case 1: return &stepper2;
    case 2: return &stepper3;
    case 3: return &stepper4;
    default: return nullptr;
  }
}

/**
 * @brief Emergency stop function - immediately halts all motion
 */
void emergencyStop() {
  emergencyStop = true;
  currentState = STATE_EMERGENCY;
  
  // Stop all motors immediately
  stepper1.stop();
  stepper2.stop();
  stepper3.stop();
  stepper4.stop();
  
  // Turn off gripper
  digitalWrite(10, LOW);
  digitalWrite(11, LOW);
  
  Serial.println("EMERGENCY STOP ACTIVATED");
  digitalWrite(13, HIGH); // Turn on status LED
}

/**
 * @brief Checks if any limit switches are pressed during normal operation
 * @return true if any switch is pressed unexpectedly
 */
bool checkUnexpectedSwitchHit() {
  if (currentState != STATE_MOVING) return false;
  
  for (int i = 0; i < NUM_MOTORS; i++) {
    homeSwitches[i].update();
    if (homeSwitches[i].fell()) {
      Serial.print("EMERGENCY: Limit switch ");
      Serial.print(i + 1);
      Serial.println(" hit during operation!");
      return true;
    }
  }
  return false;
}

/**
 * @brief Validates that target positions are within safe limits
 * @param positions Array of target positions
 * @return true if all positions are valid
 */
bool validatePositions(long positions[]) {
  for (int i = 0; i < NUM_MOTORS; i++) {
    if (positions[i] < MIN_POSITION || positions[i] > MAX_POSITION) {
      Serial.print("ERROR: Position out of range for motor ");
      Serial.print(i + 1);
      Serial.print(" (");
      Serial.print(positions[i]);
      Serial.println(")");
      return false;
    }
  }
  return true;
}
//---------------------------------------------------------------------------------
// Homing Functions
//---------------------------------------------------------------------------------

/**
 * @brief Initiates the multi-axis homing sequence.
 * This function sets the homing flags and parameters, then commands all motors
 * to move towards their respective limit switches.
 */
void startHoming() {
  Serial.println("Starting homing sequence...");
  currentState = STATE_HOMING;
  
  // Reset all motor homed flags
  for (int i = 0; i < NUM_MOTORS; i++) {
    motorHomed[i] = false;
  }
  
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
 * @brief Manages the active homing process using Bounce2 for debounced switch reading.
 * This function is called repeatedly from the main loop while homing is in progress.
 * It checks each motor's limit switch and stops it upon contact, then performs
 * a final back-off maneuver once all axes are homed.
 */
void handleHoming() {
  // Update all Bounce2 objects and check each motor's limit switch individually
  for (int i = 0; i < NUM_MOTORS; i++) {
    homeSwitches[i].update();
    
    if (!motorHomed[i]) {
      if (homeSwitches[i].fell()) { // Switch just pressed (debounced)
        AccelStepper* stepper = getStepper(i);
        stepper->stop();
        stepper->setCurrentPosition(0); // Set this position as the temporary zero
        motorHomed[i] = true;
        Serial.print("Motor ");
        Serial.print(i + 1);
        Serial.println(" homed");
      } else {
        // Continue moving towards the switch
        getStepper(i)->run();
      }
    }
  }

  // Check if all motors are homed
  bool allHomed = true;
  for (int i = 0; i < NUM_MOTORS; i++) {
    if (!motorHomed[i]) {
      allHomed = false;
      break;
    }
  }

  // This block executes only after ALL motors have hit their switches.
  if (allHomed) {
    Serial.println("All motors homed, backing off from switches...");
    
    // Command all motors to back off from the switches by a small amount.
    for (int i = 0; i < NUM_MOTORS; i++) {
      targetPositions[i] = backoffSteps;
    }
    steppers.moveTo(targetPositions);

    // This is a blocking operation to ensure the back-off move is complete.
    steppers.runSpeedToPosition();

    // After backing off, reset the current position to 0 for all axes.
    // This is the true, final home position.
    stepper1.setCurrentPosition(0);
    stepper2.setCurrentPosition(0);
    stepper3.setCurrentPosition(0);
    stepper4.setCurrentPosition(0);

    // Restore normal motion parameters
    stepper1.setMaxSpeed(maxSpeed);
    stepper1.setAcceleration(accel);
    stepper2.setMaxSpeed(maxSpeed);
    stepper2.setAcceleration(accel);
    stepper3.setMaxSpeed(maxSpeed);
    stepper3.setAcceleration(accel);
    stepper4.setMaxSpeed(maxSpeed);
    stepper4.setAcceleration(accel);

    // Finalize the homing sequence by updating state.
    currentState = STATE_READY;
    Serial.println("DONE"); // Signal that the machine is ready for commands.
  }
}

//---------------------------------------------------------------------------------
// Command Processing Functions
//---------------------------------------------------------------------------------

/**
 * @brief Processes gripper commands (GRIP/UNGRIP)
 * @param command The command string
 * @param currentTime Current timestamp for timer management
 */
void processGripperCommand(String command, unsigned long currentTime) {
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
  }
}

/**
 * @brief Processes coordinate move commands using MultiStepper for synchronized movement
 * @param command The coordinate string (e.g., "1000,2000,3000,4000")
 * @return true if command was processed successfully
 */
bool processCoordinateCommand(String command) {
  // Only allow movement if robot is ready (homed) and not in emergency state
  if (currentState != STATE_READY) {
    if (currentState == STATE_UNKNOWN) {
      Serial.println("ERROR: Robot not homed - send HOME command first");
    } else if (currentState == STATE_EMERGENCY) {
      Serial.println("ERROR: Emergency stop active - reset required");
    } else {
      Serial.println("ERROR: Robot busy");
    }
    return false;
  }

  // Find the positions of the commas to parse the coordinate string.
  int comma1 = command.indexOf(',');
  int comma2 = command.indexOf(',', comma1 + 1);
  int comma3 = command.indexOf(',', comma2 + 1);
  
  // Ensure the command format is valid (contains three commas).
  if (comma1 == -1 || comma2 == -1 || comma3 == -1) {
    Serial.println("ERROR: Invalid coordinate format - use: x,y,z,w");
    return false;
  }

  // Parse the string into integer positions for each motor.
  targetPositions[0] = command.substring(0, comma1).toInt();
  targetPositions[1] = command.substring(comma1 + 1, comma2).toInt();
  targetPositions[2] = command.substring(comma2 + 1, comma3).toInt();
  targetPositions[3] = command.substring(comma3 + 1).toInt();

  // Validate positions are within safe limits
  if (!validatePositions(targetPositions)) {
    return false;
  }

  // Use MultiStepper for coordinated movement
  steppers.moveTo(targetPositions);
  currentState = STATE_MOVING;
  
  Serial.print("Moving to: ");
  for (int i = 0; i < NUM_MOTORS; i++) {
    Serial.print(targetPositions[i]);
    if (i < NUM_MOTORS - 1) Serial.print(",");
  }
  Serial.println();
  
  return true;
}

/**
 * @brief Main command processing function
 * @param command The command string to process
 * @param currentTime Current timestamp
 */
void processCommand(String command, unsigned long currentTime) {
  command.trim(); // Remove any whitespace.
  
  // Emergency stop command - highest priority
  if (command == "STOP") {
    emergencyStop();
    return;
  }
  
  // Prevent most commands during emergency state
  if (currentState == STATE_EMERGENCY && command != "RESET") {
    Serial.println("ERROR: Emergency stop active - send RESET to clear");
    return;
  }
  
  // Reset from emergency state
  if (command == "RESET") {
    if (currentState == STATE_EMERGENCY) {
      emergencyStop = false;
      currentState = STATE_UNKNOWN;
      digitalWrite(13, LOW); // Turn off status LED
      Serial.println("Emergency stop cleared - send HOME to re-home robot");
    } else {
      Serial.println("No emergency stop to clear");
    }
    return;
  }

  // Gripper commands
  if (command == "GRIP" || command == "UNGRIP") {
    processGripperCommand(command, currentTime);
    return;
  }
  
  // Homing command
  if (command == "HOME") {
    if (currentState == STATE_HOMING) {
      Serial.println("ERROR: Homing already in progress");
      return;
    }
    startHoming();
    return;
  }
  
  // Status command
  if (command == "STATUS") {
    Serial.print("State: ");
    switch (currentState) {
      case STATE_UNKNOWN: Serial.println("UNKNOWN (not homed)"); break;
      case STATE_HOMING: Serial.println("HOMING"); break;
      case STATE_READY: Serial.println("READY"); break;
      case STATE_MOVING: Serial.println("MOVING"); break;
      case STATE_EMERGENCY: Serial.println("EMERGENCY"); break;
    }
    return;
  }
  
  // Coordinate move command (anything that looks like coordinates)
  if (command.indexOf(',') != -1 && command != "0,0,0,0") {
    if (!processCoordinateCommand(command)) {
      // Error already printed in processCoordinateCommand
      return;
    }
    return;
  }
  
  // Ignore null move command
  if (command == "0,0,0,0") {
    return;
  }
  
  // Unknown command
  Serial.println("ERROR: Unknown command");
  Serial.println("Valid commands: HOME, STOP, RESET, GRIP, UNGRIP, STATUS, x,y,z,w");
}
//---------------------------------------------------------------------------------
// Main Program Loop
//---------------------------------------------------------------------------------

/**
 * @brief The main program loop, which runs continuously.
 * It listens for serial commands, manages machine states (homing vs. moving),
 * controls the end effector timer, and calls the run() methods for the steppers.
 */
void loop() {
  unsigned long currentTime = millis(); // Get current time for timer logic.

  // --- 1. Safety Check - Emergency limit switch monitoring ---
  // Check for unexpected limit switch hits during normal operation
  if (checkUnexpectedSwitchHit()) {
    emergencyStop();
    return;
  }

  // --- 2. Command Processing ---
  // Check if there is a new command from the serial port.
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    processCommand(command, currentTime);
  }

  // --- 3. State-Based Processing ---
  
  // Handle homing sequence if in progress
  if (currentState == STATE_HOMING) {
    handleHoming();
  }

  // Handle move completion check
  if (currentState == STATE_MOVING) {
    // Check if all motors have reached their destination
    if (stepper1.distanceToGo() == 0 &&
        stepper2.distanceToGo() == 0 &&
        stepper3.distanceToGo() == 0 &&
        stepper4.distanceToGo() == 0) {
      
      currentState = STATE_READY;
      Serial.println("DONE"); // Send completion signal.
    }
  }

  // --- 4. End Effector Timer Management ---
  // Check if the 5-second gripper duration has elapsed since it was triggered.
  if (triggerStartTime > 0 && (currentTime - triggerStartTime >= triggerDuration)) {
    digitalWrite(10, LOW); // Turn off the gripper motor.
    digitalWrite(11, LOW);
    triggerStartTime = 0; // Reset the timer start time.
    Serial.println("DONE");
  }

  // --- 5. Motor Execution ---
  // Only run motors if not in emergency stop
  // The .run() function calculates and executes steps based on current speed and acceleration.
  if (!emergencyStop) {
    stepper1.run();
    stepper2.run();
    stepper3.run();
    stepper4.run();
  }
}