import serial
import time

# You may need to install the pyserial library first:
# pip install pyserial

def main():
    """
    Main function to connect to Arduino and send commands.
    """
    print("--- Arduino Serial Communication ---")
    
    # Prompt the user for the COM port.
    # On Windows, it will be 'COMX' (e.g., 'COM3').
    # You can find the port in the Arduino IDE under Tools > Port.
    try:
        port_name = input("Enter the Arduino's serial port (should be COM4): ")
        
        # Establish the serial connection.
        # The baud rate (115200) must match the one in the Arduino sketch.
        # The timeout is set to 1 second to avoid waiting forever for a response.
        arduino = serial.Serial(port=port_name, baudrate=115200, timeout=1)
        
        # Wait a moment for the connection to initialize.
        time.sleep(2)
        print(f"Successfully connected to Arduino on {port_name}")

    except serial.SerialException as e:
        print(f"Error: Could not open serial port '{port_name}'.")
        print(f"Details: {e}")
        print("Please make sure the Arduino is connected and you've entered the correct port name.")
        return # Exit the script if connection fails

    print("\nEnter commands to send to the Arduino (e.g., 'HOME', '200,200,200,200', 'RIGHT_TRIGGER', 'LEFT_TRIGGER').")
    print("Type 'exit' to quit.")

    # Main loop to send and receive data
    while True:
        # Get user input
        command = input("Enter command: ")

        if command.lower() == 'exit':
            break

        # Send the command to the Arduino.
        # It needs to be encoded into bytes and have a newline character at the end.
        arduino.write((command + '\n').encode('utf-8'))
        
        # Wait briefly for the Arduino to process and respond.
        time.sleep(0.05) 

        # Read any response lines from the Arduino.
        while arduino.in_waiting > 0:
            try:
                response = arduino.readline().decode('utf-8').strip()
                if response:
                    print(f"Arduino says: {response}")
            except UnicodeDecodeError:
                # Occasionally, the first connection might have some noise.
                print("Arduino says: (Could not decode response)")


    # Cleanly close the connection when done.
    arduino.close()
    print("Serial connection closed. Goodbye!")


if __name__ == '__main__':
    main()
