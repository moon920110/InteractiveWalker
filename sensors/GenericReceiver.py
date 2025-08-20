import time
import struct
from .Sensor import Sensor
import aioconsole
import asyncio
from typing import List

class GenericReceiverClass():
    def __init__(self, numNodes, sensors: List[Sensor], record):
        self.frameRate = None
        self.startTime = time.time()
        self.numNodes = numNodes
        self.sensors = {sensor.id: sensor for sensor in sensors}
        self.caxs = []

        self.record = record
        self.pressure_min = 0
        self.pressure_max = 4096
        self.use_log = True

        self.partialData = b''
        self.buffer = asyncio.Queue()

    def startReceiver(self):
        raise NotImplementedError("Receivers must implement a startReceiver method")
    async def stopReceiver(self):
        raise NotImplementedError("Receivers must implement a stop receiver method")
    

    def unpackBytesPacket(self, byteString):
        format_string = '=b' + 'H' * (1+self.numNodes) + 'I'  # 'b' for int8_t, 'H' for uint16_t
        tupData = struct.unpack(format_string,byteString)
        sendId = tupData[0]
        startIdx = tupData[1]
        sensorReadings = tupData[2:-1]
        packetNumber = tupData[-1]
        return sendId, startIdx,sensorReadings, packetNumber

    async def process_line(self, line):
        if len(line) == 1+(1+self.numNodes)*2+4:
            sendId, startIdx, readings, packetID = self.unpackBytesPacket(line)
            sensor = self.sensors[sendId]
            await sensor.processRowAsync(startIdx, readings, packetID)

    async def read_lines(self):
        print("Reading lines")
        while not self.stopFlag.is_set():
            data = await self.buffer.get()
            self.partialData += data
            lines = self.partialData.split(b'wr')
            self.partialData = lines.pop()
            for line in lines:
                await self.process_line(line)

    async def listen_for_stop(self):
        print("Listening for stop")
        while not self.stopFlag.is_set():
            input_str = await aioconsole.ainput("Press Enter to stop...\n")
            if input_str == "":
                self.stopFlag.set()
                await self.stopReceiver()



    
    