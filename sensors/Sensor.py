import numpy as np
import h5py
import time
import asyncio
from .utils import getUnixTimestamp

class Sensor():
    def __init__(self, selWires:int, readWires:int,numNodes, id, deviceName = "Esp1", intermittent = False, p=15, fileName=None, port="COM3",baud="250000"):
        self.id = id
        self.readWires = readWires
        self.selWires = selWires
        self.deviceName = deviceName
        self.path = f'./{fileName}.hdf5' if fileName is not None else f'sensors/recordings/recordings_{id}_{str(time.time())}.hdf5'
        self.file = None
        self.pressure = np.zeros(readWires*selWires)
        self.fc = 0
        self.init = False
        self.filledSize = 0
        self.bufferSize = numNodes
        self.pressureLength = readWires*selWires
        self.left_to_fill = self.pressureLength
        self.block_size=1024
        self.packetCount = 0
        self.maxPackets = int(np.ceil(self.pressureLength/self.bufferSize))
        self.receivedPackets = np.zeros(self.maxPackets)
        self.lock = asyncio.Lock()

        #intermittent 
        self.intermittent = intermittent
        self.receivedIdxs = np.zeros(self.maxPackets)
        self.prevPressure = np.zeros(readWires*selWires)
        self.intermittentInit = False
        self.expectedPacket = 1
        self.nextStartIdx = 0
        self.p=p
        self.predCount=0
        self.lastTs = None

        #serial options
        self.port = port
        self.baudrate = baud



    def append_data(self, ts,reading, packet):
        print(ts)
        f = self.file
        block_size = self.block_size
        fc = self.fc
        init = self.init
        if not init:
            self.file = h5py.File(self.path, 'w')
            f=self.file
            self.init=True
            sz = [block_size, ]
            maxShape = sz.copy()
            maxShape[0] = None
            maxshapePressure = [None,self.selWires,self.readWires]
            f.create_dataset('frame_count', (1,),maxshape=maxShape, dtype=np.uint32)
            f.create_dataset('ts', tuple([block_size, ]), maxshape = maxShape, dtype=ts.dtype, chunks=True)
            f.create_dataset('predCount', tuple([block_size, ]), maxshape = maxShape, dtype=ts.dtype, chunks=True)
            f.create_dataset('pressure', tuple([block_size, self.selWires,self.readWires]), maxshape=maxshapePressure, dtype=np.int32, chunks=True)
            if packet is not None:
                maxshapePackets = [None, self.maxPackets]
                f.create_dataset('packetNumber', tuple([block_size, self.maxPackets]), maxshape=maxshapePackets, dtype=np.uint32, chunks=True)

        # Check size
        oldSize = f['ts'].shape[0]
        if oldSize == fc:
            newSize = oldSize + block_size
            f['ts'].resize(newSize, axis=0)
            f['pressure'].resize(newSize, axis=0)
            f['predCount'].resize(newSize, axis=0)
            if packet is not None:
                f['packetNumber'].resize(newSize, axis=0)

        f['frame_count'][0] = fc
        f['ts'][fc] = ts
        f['predCount'][fc] = self.predCount
        f['pressure'][fc] = reading.reshape(1, self.selWires,self.readWires)
        if packet is not None:
            f['packetNumber'][fc] = self.receivedPackets
        f.flush()

    def fillBuffer(self, startIdx, amountToFill, readings):
        if startIdx + amountToFill <= self.pressureLength:
            self.pressure[startIdx:startIdx+amountToFill] = np.array(readings[:amountToFill])
        else:
            firstSize = self.pressureLength - startIdx
            secondSize = amountToFill- firstSize
            self.pressure[startIdx:]=np.array(readings[:firstSize])
            self.pressure[:secondSize]=np.array(readings[firstSize:amountToFill])

    def processRow(self, startIdx,readings, packet=None, record=True):
        if packet is not None:
                self.receivedPackets[self.packetCount]=packet
                self.packetCount+=1
        if self.left_to_fill <= self.bufferSize:

            if self.left_to_fill > 0:
                self.fillBuffer(startIdx,self.left_to_fill,readings)
            ts = getUnixTimestamp()
            if record:
                self.append_data(ts,self.pressure,packet)
            self.fc+=1
            self.packetCount = 0
            self.receivedPackets=np.zeros(self.maxPackets)
            remaining = self.bufferSize - self.left_to_fill
            self.fillBuffer((startIdx+self.left_to_fill)%self.pressureLength, remaining, readings[self.left_to_fill:])
            self.left_to_fill = self.pressureLength-remaining
        else:
            self.fillBuffer(startIdx,self.bufferSize, readings)
            self.left_to_fill -= self.bufferSize

    def processRowReadNode(self,readings,packet,record=True):
        for i in range(0,len(readings),2):
            nodeLocation = readings[i]
            nodeReading = readings[i+1]
            self.pressure[nodeLocation] = nodeReading
            if nodeLocation == self.pressureLength-1 and record:
                ts = getUnixTimestamp()
                self.append_data(ts,self.pressure,packet)
                self.fc+=1

    
    def processRowIntermittent(self, startIdx, readings, packet, record=True):
         # If the packet id is not what we're expecting (during intermittent sending), then we should predict all of the missed packets in between
        if packet != self.expectedPacket and self.intermittentInit:
            currTs = getUnixTimestamp()
            for packetIdx in range(self.expectedPacket, packet):
                #Predict Packet
                predicted = self.predictPacket(self.nextStartIdx)
                #Estimate Timestamp
                predTs = self.lastTs + ((currTs-self.lastTs)*(packetIdx-self.expectedPacket)/(packet-self.expectedPacket))
                self.predCount+=1
                self.packetHandle(self.nextStartIdx,predicted,packetIdx, predTs, record)
                self.nextStartIdx = (startIdx+self.bufferSize)%self.pressureLength
                self.receivedPackets[self.packetCount]=packet
                self.packetCount+=1
            self.lastTs = currTs
        else:
            ts = getUnixTimestamp()
            self.packetHandle(startIdx,readings,packet, ts, record)
            self.nextStartIdx = (startIdx+self.bufferSize)%self.pressureLength
            self.receivedPackets[self.packetCount]=packet
            self.packetCount+=1
            self.lastTs = ts
        self.expectedPacket = packet+1
        
            


        
    def packetHandle(self,startIdx,readings,packet, ts, record):
        print(ts)
        if self.left_to_fill <= self.bufferSize:
            if self.left_to_fill > 0:
                self.fillBuffer(startIdx,self.left_to_fill,readings)
            if record:
                self.append_data(ts,self.pressure,packet)
            self.prevPressure = self.pressure
            self.fc+=1
            if self.fc==2:
                self.intermittentInit=True
            self.packetCount = 0
            self.receivedPackets=np.zeros(self.maxPackets)
            remaining = self.bufferSize - self.left_to_fill
            self.fillBuffer((startIdx+self.left_to_fill)%self.pressureLength, remaining, readings[self.left_to_fill:])
            self.left_to_fill = self.pressureLength-remaining
        else:
            self.fillBuffer(startIdx,self.bufferSize, readings)
            self.left_to_fill -= self.bufferSize

    def predictPacket(self,startIdx):
        predicted=np.zeros(self.bufferSize)
        endIdx=min(self.bufferSize, self.pressureLength-startIdx)
        predicted[:endIdx]=self.pressure[startIdx:startIdx+endIdx] + 1/self.p*(self.pressure[startIdx:startIdx+endIdx]-self.prevPressure[startIdx:startIdx+endIdx])
        #Handle overflow
        if startIdx+self.bufferSize>self.pressureLength:
            newIdx = endIdx
            endIdx = self.bufferSize-(self.pressureLength-startIdx)
            predicted[newIdx:] = self.pressure[:endIdx] + 1/self.p*(self.pressure[:endIdx]-self.prevPressure[:endIdx])
        return predicted

    async def processRowAsync(self, startIdx,readings, packet=None):
        async with self.lock:
             self.processRow(startIdx,readings,packet)
            
            
