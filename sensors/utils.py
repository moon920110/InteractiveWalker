import h5py
import numpy as np
import json5
import json
import serial
import datetime
from datetime import datetime
import subprocess


def tactile_reading(path):
    f = h5py.File(path, 'r')
    fc = f['frame_count'][0]
    ts = np.array(f['ts'][:fc])
    pressure = np.array(f['pressure'][:fc]).astype(np.float32)

    return pressure, fc, ts

def find_closest_index(array, value):
    index = (np.abs(array - value)).argmin()
    return index, array[index]

def getUnixTimestamp():
    return np.datetime64(datetime.now()).astype(np.int64) / 1e6  # unix TS in secs and microsecs

def start_nextjs():
    try:
        subprocess.Popen(['npm', 'run', 'next-dev'], cwd='./ui/nextjs-flask', shell=True)
    except Exception as e:
        print(f"Failed to start Next.js: {e}")

def programSensor(sensor_id, config="sensors/WiSensConfigClean.json"):
    # Read the JSON file
    with open(config, 'r') as file:
        data = json5.load(file)
    
    # Find the sensor with the given ID
    sensor = next((s for s in data['sensors'] if s['id'] == sensor_id), None)
    if not sensor:
        raise ValueError(f"Sensor with ID {sensor_id} not found.")

    # Determine the protocol
    protocol = sensor.get('protocol')
    protocol_key = f"{protocol}Options"
    if protocol_key not in data:
        raise ValueError(f"Protocol '{protocol}' not supported.")

    # Get the protocol options
    protocol_options = data.get(protocol_key, {})

    #get readout options
    readout_options = data.get("readoutOptions",{})

    # Merge sensor data with the protocol options
    merged_data = {**sensor, **protocol_options, **readout_options}

    # Convert the merged data to a JSON string with proper quoting
    json_string = json.dumps(merged_data)
    print(json_string)
    # Send the JSON string over the serial port
    ser = serial.Serial(baudrate=data['serialOptions']['baudrate'], timeout=1)
    if "serialPort" in sensor:
        ser.port=sensor["serialPort"]
    else:
        ser.port=data['serialOptions']['port']
    ser.dtr = False
    ser.rts = False
    ser.open()
    ser.write(json_string.encode('utf-8'))
    ser.close()
    