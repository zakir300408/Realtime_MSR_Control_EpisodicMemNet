import serial
import serial.tools.list_ports
import time


class SignalGenerator:
    TARGET_DESCRIPTION = "USB-SERIAL CH340"
    CMD_PREFIXES = {
        "amplitude": {1: "WMA", 2: "WFA", 3: "TFA"},
        "phase": {1: "WMP", 2: "WFP", 3: "TFP"},
    }
    WAVEFORM_PREFIXES = {1: "WMW", 2: "WFW", 3: "TFW"}
    FREQUENCY_PREFIXES = {1: "WMF", 2: "WFF", 3: "TFF"}
    OUTPUT_PREFIXES = {1: "WMN", 2: "WFN", 3: "TFN"}
    FACTORS = {1: 1.2, 2: 2.6, 3: 2.15}

    def __init__(self, baudrate=115200):
        ports = list(serial.tools.list_ports.comports())
        port = None
        for p in ports:
            if self.TARGET_DESCRIPTION in p.description:
                port = p.device
                break
        if port is None:
            raise Exception(
                f"No device with description {self.TARGET_DESCRIPTION} found."
            )

        self.ser = serial.Serial(port, baudrate, timeout=1)

    def _send_command(self, command):
        #print(f"Sending command: {command.strip()}")
        self.ser.write(command.encode("ascii"))
        
        # Add delay after sending command
        time.sleep(0.3)  # 100ms delay

        response = self.ser.read(1)  # Read the first byte
        if response == b'\x0a':
            #print(f"Received expected response: {response}")
            return response
        else:
            full_response = response + self.ser.read(self.ser.in_waiting)  # Read remaining bytes
            print(f"Received unexpected response: {full_response}")
            raise Exception(f"Command {command.strip()} failed with response: {full_response}")

    def close(self):
        self.ser.close()

    def set_parameter(self, channel, parameter, value):
        if (
                parameter not in self.CMD_PREFIXES
                or channel not in self.CMD_PREFIXES[parameter]
        ):
            return

        cmd_prefix = self.CMD_PREFIXES[parameter][channel]

        if parameter == "amplitude" and isinstance(value, str) and "ampere" in value:
            recorded_ampere = float(value.replace("ampere", ""))
            value = recorded_ampere / self.FACTORS.get(channel, 1)

        cmd = None
        if parameter in ["amplitude", "offset"]:
            cmd = f"{cmd_prefix}{float(value):.3f}\x0a"
        elif parameter == "phase":
            cmd = f"{cmd_prefix}{float(value):05.1f}\x0a"

        return self._send_command(cmd)

    def initialize_channel(self, channel, waveform, frequency):
        if channel not in self.WAVEFORM_PREFIXES:
            return

        cmds = [
            f"{self.WAVEFORM_PREFIXES[channel]}{int(waveform):02}\x0a",
            f"{self.FREQUENCY_PREFIXES[channel]}{int(frequency * 1e6):014}\x0a",
            f"{self.OUTPUT_PREFIXES[channel]}1\x0a",
        ]

        for cmd in cmds:
            self._send_command(cmd)

    def set_channel_parameters(self, channel, parameters):
        for parameter, value in parameters.items():
            self.set_parameter(channel, parameter, value)

    def move(self, settings):
        for channel, params in settings.items():
            self.set_channel_parameters(channel, params)

    def first_init(self):
        initial_settings = {
            1: {"amplitude": 10, "phase": 0},
            2: {"amplitude": 5, "phase": 0},
            3: {"amplitude": 10, "phase": 0},
        }

        for channel, params in initial_settings.items():
            self.set_channel_parameters(channel, params)

        user_confirmation = input("Have you set generator X=20V, Y=10V, Z=10V? (Y/N): ")
        while user_confirmation.upper() != "Y":
            user_confirmation = input(
                "Have you set generator X=20V, Y=10V, Z=10V? (Y/N): "
            )

        zero_settings = {
            1: {"amplitude": 0},
            2: {"amplitude": 0},
            3: {"amplitude": 0},
        }

        for channel, params in zero_settings.items():
            self.set_channel_parameters(channel, params)
