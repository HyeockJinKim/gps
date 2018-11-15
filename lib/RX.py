import math, random, os, binascii

from Crypto.Cipher import AES
from bitstring import BitArray

import numpy as np
import lib.Parameters as PM

from threading import Timer,Thread,Event

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pandas as pd
from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D

corr_OS = np.zeros((PM.freqBins, PM.RXBUFLEN), dtype=float)
corr_PAC = np.zeros((PM.freqBins, PM.RXBUFLEN), dtype=float)
corr_ENC = np.zeros((PM.freqBins, PM.RXBUFLEN), dtype=float)

class Simulation():

   def __init__(self,t,mode):
      self.t=t
      self.hFunction = self.ACQ
      self.thread = Timer(self.t,self.handle_function)
      self.daemon = True
      self.jobDone = 0;
      self.Time = 0;
      self.acquired = 0;
      self.acquired_DAC = 0;
      self.acquired_FOLD = 0;
      self.peak = 0
      self.corr = np.zeros((PM.freqBins, PM.RXBUFLEN), dtype=float)

      self.SATs = [Satellite() for i in range(PM.NUMSAT)];
      self.RXs = [Receiver() for i in range(PM.NUMRX)];
      ######################## Setting each satellite ###############################
      mseq = np.zeros(PM.CODELEN, dtype=int);
      MSEQ(mseq);  # m-sequence generation
      Pattern = DECIMATION(mseq, PM.CODELEN, 79);
      for SATIdx in range(PM.NUMSAT):
          self.SATs[SATIdx].Pattern = Pattern;  # PAC generating Patterns

      BPSK(mseq, PM.CODELEN);  # BPSK modulation
      mseq2 = DECIMATION(mseq, PM.CODELEN, 5);  # decimated m-sequence generation
      for SATIdx in range(PM.NUMSAT):
          self.SATs[SATIdx].ID = SATIdx;  # Set SVIDs
          self.SATs[SATIdx].mode = mode;  # Set operation modes
          self.SATs[SATIdx].OSCodeBuf = SAA(mseq, mseq2, PM.CODELEN, self.SATs[SATIdx].ID);  # Allocate base codes
          self.SATs[SATIdx].DataBuf = 2 * np.random.randint(2, size=PM.DATALEN) - 1;  # Set the dataBuffers
          self.SATs[SATIdx].Key = binascii.b2a_hex(os.urandom(8));  # Set encryption key
      ###############################################################################

      ######################## Setting each receiver ################################
      for RXIdx in range(PM.NUMRX):
          self.RXs[RXIdx].delays = np.random.uniform(0, 1, size=PM.NUMSAT);  # Set propagation delays
          self.RXs[RXIdx].delays[0] = 0.005;
          # self.RXs[RXIdx].delays = np.zeros(PM.NUMSAT, dtype=int);  # Set propagation delays
          self.RXs[RXIdx].dopplers = np.random.randint(2 * PM.MAXFd, size=PM.NUMSAT) - PM.MAXFd;  # Set doppler frequencies
          # RXs[RXIdx].dopplers = np.zeros(PM.NUMSAT, dtype=float);
          self.RXs[RXIdx].SNRdB = PM.SNRdB;
      ###############################################################################

      ############################# Fill the RX buffers  ############################
      for RXIdx in range(PM.NUMRX):
          for SATIdx in range(PM.NUMSAT):
              now = PM.INITTIME - self.RXs[RXIdx].delays[SATIdx];
              FILL_ENCBUF(now, self.SATs[SATIdx])
              for BUFIdx in range(PM.RXBUFLEN):
                  # Check whether the ENCBuf should be filled with new one
                  now = PM.INITTIME + BUFIdx / PM.Fs - self.RXs[RXIdx].delays[SATIdx];
                  before = PM.INITTIME + (BUFIdx - 1) / PM.Fs - self.RXs[RXIdx].delays[SATIdx];
                  if TIME2CHIP(now) % PM.CODELEN == 0 and TIME2CHIP(before) % PM.CODELEN == PM.CODELEN - 1:
                      FILL_ENCBUF(now, self.SATs[SATIdx]);

                  Chip = GETSAMPLE(now, self.SATs[SATIdx]);
                  self.RXs[RXIdx].IBuffer[BUFIdx] += Chip * np.cos(
                      2 * np.pi * now * (PM.F + self.RXs[RXIdx].dopplers[SATIdx]));
                  self.RXs[RXIdx].QBuffer[BUFIdx] += Chip * np.sin(
                      2 * np.pi * now * (PM.F + self.RXs[RXIdx].dopplers[SATIdx]));

          sigma = np.sqrt(10 ** (-self.RXs[RXIdx].SNRdB / 10));
          self.RXs[RXIdx].IBuffer += np.random.normal(0, sigma, PM.RXBUFLEN);
          self.RXs[RXIdx].QBuffer += np.random.normal(0, sigma, PM.RXBUFLEN);

   def handle_function(self):
      self.hFunction()
      self.thread = Timer(self.t,self.handle_function)
      self.thread.start()

   def stop(self, corr):
       global corr_OS, corr_PAC, corr_ENC
       if self.SATs[0].mode == 0:
           corr_OS = corr;
       elif self.SATs[0].mode == 1:
           corr_PAC = corr;
       elif self.SATs[0].mode == 2:
           corr_ENC = corr;
           Ploting()

   def start(self):
      self.thread.start()

   def cancel(self, corr):
      fig = plt.figure()
      ax = fig.gca(projection='3d')
      df = pd.DataFrame(corr)
      x = df.columns;
      y = df.index;
      X, Y = np.meshgrid(x, y)
      Z = df;
      surf = ax.plot_surface(X, Y, Z, cmap='bwr')


   def wait(self):
       self.thread.join()

   def ACQ(self):
        if self.jobDone == 0:
            replica = np.zeros(2 * PM.RXBUFLEN, dtype=int)
            corr = np.zeros((PM.freqBins, PM.RXBUFLEN), dtype=float)
            if self.SATs[0].mode == 0:
                a = np.max(corr)
                corr = ACQUISITION(self.RXs[0], self.SATs[0], corr, self.Time)
                self.Time = self.Time + PM.RXBUFLEN / PM.Fs
                a = np.max(corr)
                T = PM.T_OS
            elif self.SATs[0].mode == 1:
                if self.acquired_DAC == 0:
                    T = PM.T_DAC
                    ACQUISITION_DAC(self.RXs[0], self.SATs[0], corr, self.Time)
                else:
                    T = PM.T_PAC
                    test = []
                    test = ACQUISITION_PAC(self.RXs[0], self.SATs[0], 0, self.codePhase, self.dopplerBin, test)

                self.Time = self.Time + PM.RXBUFLEN / PM.Fs
            elif self.SATs[0].mode == 2:
                ACQUISITION(self.RXs[0], self.SATs[0], corr, self.Time)
                self.Time = self.Time + PM.RXBUFLEN / PM.Fs
                T = PM.T_ENC
            elif self.SATs[0].mode == 3:
                if self.acquired_FOLD == 0:
                    T = PM.T_FOLD
                    # ACQUISITION_FOLDING(self.RXs[0], self.SATs[0], corr, self.Time)
                    self.Time = self.Time + PM.FOLDFACTOR * PM.RXBUFLEN / PM.Fs
                else:
                    T = PM.T_ENC
            if self.acquired_DAC == 0 or self.acquired_FOLD ==0:
                peak = np.max(corr)

            print(peak);
            if self.SATs[0].mode == 0 or self.SATs[0].mode == 2:
                if peak >= T:
                    self.jobDone = 1;
                    self.stop(corr)
                    return
            elif self.SATs[0].mode == 1:
                if self.acquired_DAC == 0 and peak >= T:
                    self.peak = peak;
                    self.corr = corr;
                    self.acquired_DAC = 1
                    self.dopplerBin, self.codePhase = FINDINARRAY(corr, peak, PM.freqBins, PM.RXBUFLEN)
                elif self.acquired_DAC == 1 and max(test) + self.peak >= T:
                    self.corr[self.dopplerBin, self.codePhase] += max(test);
                    self.jobDone = 1;
                    print(max(test) + self.peak)
                    self.stop(self.corr)
                    return
            elif self.SATs[0].mode == 3:
                if self.acquired_FOLD == 0 and peak >= T:
                    acquired_FOLD = 1
                elif self.acquired_FOLD == 1 and peak >= T:
                    self.jobDone = 1;
                    self.stop(corr)
                    return



######### Information about satellites ###############
class Satellite:
    ID = -1;
    mode = 0;                                   # 0 : Open Signal. 1 : PAC Signal, 2: Encrypted Signal
    DataBuf = np.zeros(PM.DATALEN, dtype=int);     # Data Buffer
    OSCodeBuf = np.zeros(PM.CODELEN, dtype=int);   # Open Signal Code Buffer
    ENCBuf = np.zeros(PM.CODELEN+1, dtype=int);      # Encrypted Code Buffer
    RandValue = np.random.randint(100000, 5000000);
    Pattern = np.zeros(PM.CODELEN, dtype=int);     # Watermarking pattern for PAC
    Key = 0;
######################################################

########## Information about a receiver ##############
class Receiver:
    delays = np.zeros(PM.NUMSAT, dtype=float);             # Delays for each satellite signals
    dopplers = np.zeros(PM.NUMSAT, dtype=int);             # Dopplers for each satellite signals
    IBuffer = np.zeros(PM.RXBUFLEN, dtype=float);          # Inphase signal buffer
    QBuffer = np.zeros(PM.RXBUFLEN, dtype=float);          # Quadrature Phase signal buffer
    SNRdB = 10;
######################################################

############# Functions for generating m-sequences and Gold codes ##############
def MSEQ(SEQ):
    #SEQ[0:9] = array([0, 1, 1, 1, 1, 0, 1, 0, 1, 0]);
    SEQ[0] = 0;    SEQ[1] = 1;    SEQ[2] = 1;
    SEQ[3] = 1;    SEQ[4] = 1;    SEQ[5] = 0;
    SEQ[6] = 1;    SEQ[7] = 0;    SEQ[8] = 1;
    SEQ[9] = 0;
    for i in range(PM.TAPS, PM.CODELEN):
        for j in range(0, len(PM.POS) - 1):
            SEQ[i] = SEQ[i] + SEQ[i - PM.POS[j]];

        SEQ[i] = SEQ[i] % 2;
def BPSK(SEQ, L):
    for i in range(0, L):
        SEQ[i] = 1 if (SEQ[i] == 0) else -1;
def DECIMATION(SEQ, LEN, factor):
    DECSEQ = np.zeros(LEN);
    for i in range(0, LEN):
        DECSEQ[i] = SEQ[int((factor * i) % LEN)];

    return DECSEQ;
def SAA(SEQ1, SEQ2, LEN, SHIFT):
    OUT = np.zeros(LEN, dtype=int);
    for i in range(0, LEN):
        OUT[i] = SEQ1[i % LEN] * SEQ2[(i + SHIFT) % LEN];

    return OUT;
def RCYCLESHIFT(SEQ, LEN, SHIFT):
    OUT = np.zeros(LEN, dtype=int);
    for i in range(0, LEN):
        OUT[i] = SEQ[(i + SHIFT) % LEN];

    return OUT;
################################################################################

################# generate code chip and data at the given time ################
def TIME2CHIP(time):
    return int(math.floor(time*PM.Fc));
def TIME2SAMPLE(time):
    return int(math.ceil(time*PM.Fs))
def TIME2DATA(time):
    return int(math.floor(time*PM.R));
def GETCHIP(TX, ChipIdx):
    if TX.mode == 0:        # Open signal
        return TX.OSCodeBuf[ChipIdx];
    elif TX.mode == 1:      # PAC signal
        if(TX.Pattern[ChipIdx] == 1 ):
            return TX.OSCodeBuf[ChipIdx];
        else:
            return TX.ENCBuf[ChipIdx];
    else:
        return TX.ENCBuf[ChipIdx];
def GETDATA(TX, DataIdx):
    return TX.DataBuf[DataIdx];
def FILL_ENCBUF(time, SAT):
    CODEIdx = int(math.floor(time * PM.Fc / PM.CODELEN))*10;
    if PM.TRUE_ENC == 1:
        cipher = AES.new(SAT.Key, AES.MODE_ECB);
        for i in range(8):
            msg = format(CODEIdx + i, 'x');
            msg = msg.zfill(15);
            msg = format(SAT.ID, 'x') + msg;
            out = cipher.encrypt(msg);
            out = BitArray(out);
            for CIdx in range(PM.MSGLEN):
                SAT.ENCBuf[i * PM.MSGLEN + CIdx] = int(out.bin[CIdx]);
    else:
        np.random.seed(CODEIdx+SAT.RandValue);
        SAT.ENCBuf = np.random.randint(2, size=PM.ENCBUFLEN);

    BPSK(SAT.ENCBuf, PM.CODELEN+1);
def GETSAMPLE(time, SAT):
    return GETDATA(SAT, TIME2DATA(time) % PM.DATALEN) * GETCHIP(SAT, TIME2CHIP(time) % PM.CODELEN);
################################################################################
################################################################################
def GETREPLICA(BUF, SAT, time):
    now = time;
    FILL_ENCBUF(now, SAT);
    for BUFIdx in range(2*PM.RXBUFLEN):
        # Check whether the ENCBuf should be filled with new one
        now = time + BUFIdx / PM.Fs;
        before = time + (BUFIdx - 1) / PM.Fs;
        a = TIME2CHIP(now) % PM.CODELEN;
        b = TIME2CHIP(before) % PM.CODELEN;
        if TIME2CHIP(now) % PM.CODELEN == 0 and TIME2CHIP(before) % PM.CODELEN == 1022:
            FILL_ENCBUF(now, SAT);

        BUF[BUFIdx] = GETCHIP(SAT, TIME2CHIP(now) % PM.CODELEN);
    return BUF;
def GETREPLICA_DAC(BUF, SAT, time):
    now = time;
    FILL_ENCBUF(now, SAT);
    for BUFIdx in range(2*PM.RXBUFLEN):
        # Check whether the ENCBuf should be filled with new one
        now = time + BUFIdx / PM.Fs;
        before = time + (BUFIdx - 1) / PM.Fs;
        CIdx = TIME2CHIP(now) % PM.CODELEN;
        CIdx_pref = TIME2CHIP(before) % PM.CODELEN;
        if CIdx == 0 and CIdx_pref == 1022:
            FILL_ENCBUF(now, SAT);

        if SAT.Pattern[CIdx] == 1:
            BUF[BUFIdx] = GETCHIP(SAT, TIME2CHIP(now) % PM.CODELEN);
        else:
            BUF[BUFIdx] = 0;
def GETREPLICA_PAC(BUF, SAT, time):
    now = time;
    FILL_ENCBUF(now, SAT);
    for BUFIdx in range(PM.RXBUFLEN):
        # Check whether the ENCBuf should be filled with new one
        now = time + BUFIdx / PM.Fs;
        before = time + (BUFIdx - 1) / PM.Fs;
        CIdx = TIME2CHIP(now) % PM.CODELEN;
        CIdx_pref = TIME2CHIP(before) % PM.CODELEN;
        if CIdx == 0 and CIdx_pref == 1022:
            FILL_ENCBUF(now, SAT);

        BUF[BUFIdx] = GETCHIP(SAT, TIME2CHIP(now) % PM.CODELEN);
def GETREPLICA_FOLD(BUF, SAT, time):
    now = time;
    FILL_ENCBUF(now, SAT);
    tmp = np.zeros((1 + PM.FOLDFACTOR) * PM.RXBUFLEN);
    for BUFIdx in range((1 + PM.FOLDFACTOR) * PM.RXBUFLEN):
        # Check whether the ENCBuf should be filled with new one
        now = time + BUFIdx / PM.Fs;
        before = time + (BUFIdx - 1) / PM.Fs;
        a = TIME2CHIP(now) % PM.CODELEN;
        b = TIME2CHIP(before) % PM.CODELEN;
        if TIME2CHIP(now) % PM.CODELEN == 0 and TIME2CHIP(before) % PM.CODELEN == 1022:
            FILL_ENCBUF(now, SAT);

        tmp[BUFIdx] = GETCHIP(SAT, TIME2CHIP(now) % PM.CODELEN);

    for i in range(2 * PM.RXBUFLEN):
        BUF[i] = tmp[BUFIdx];
        for j in range(PM.FOLDFACTOR - 1):
            BUF[i] += tmp[j * PM.RXBUFLEN + i];

def FFTSEARCH(IBUF, QBUF, replica, FdIdx):
    t = np.arange(0, PM.RXBUFLEN / PM.Fs, 1 / PM.Fs);
    I_comp = np.cos(2 * np.pi * (PM.F - PM.MAXFd + PM.STEPFd * FdIdx) * t) * np.transpose(IBUF);
    Q_comp = np.sin(2 * np.pi * (PM.F - PM.MAXFd + PM.STEPFd * FdIdx) * t) * np.transpose(QBUF);

    x = I_comp + 1j * Q_comp;
    x = np.pad(x, (0, PM.RXBUFLEN), 'constant');
    X = np.fft.fft(x);
    F_replica = np.fft.fft(replica + 1j * 0);
    F_replica = F_replica.conj();

    tmp = abs(np.fft.ifft(X * F_replica));
    return tmp[0:PM.RXBUFLEN];
def MATCHEDFILTER(BUF, replica):
    return abs(np.inner(BUF, replica));

def FINDINARRAY(array, value, rows, cols):
    for i in range(0,rows):
        for j in range(0,cols):
            if array[i][j]==value:
                return i, j;
def FINDINVECTOR(vector, value, length):
    for i in range(length):
        if vector[i] == value:
            return i;

def ACQUISITION(RX, SAT, corr, startTime):
    isAcquired = 0;
    time = startTime;
    cnt = 0;
    if SAT.mode == 0:
        T = PM.T_OS;
    else:
        T = PM.T_ENC;

    replica = np.zeros(2 * PM.RXBUFLEN, dtype=int);

    replica=GETREPLICA(replica, SAT, time)
    for FdIdx in range(PM.freqBins):
        corr[FdIdx, :] = FFTSEARCH(RX.IBuffer, RX.QBuffer, replica, FdIdx);

    return corr;
def ACQUISITION_DAC(RX, SAT, corr, startTime):

    time = startTime;
    ## Acquisition procedure with DAC
    replica = np.zeros(2 * PM.RXBUFLEN, dtype=int);

    GETREPLICA_DAC(replica, SAT, time);
    for FdIdx in range(PM.freqBins):
        corr[FdIdx, :] = FFTSEARCH(RX.IBuffer, RX.QBuffer, replica, FdIdx);

    return corr;
def ACQUISITION_PAC(RX, SAT, startTime, codePhase, dopplerBin, test):
    ## Acquisition procedure with PAC
    replica_PAC = np.zeros(PM.RXBUFLEN, dtype=int);
    t = np.arange(0, PM.RXBUFLEN / PM.Fs, 1 / PM.Fs);
    IBUF = np.cos(2 * np.pi * (PM.F - PM.MAXFd + PM.STEPFd * dopplerBin) * t) * np.transpose(RX.IBuffer);
    QBUF = np.sin(2 * np.pi * (PM.F - PM.MAXFd + PM.STEPFd * dopplerBin) * t) * np.transpose(RX.QBuffer);
    BUF = IBUF + 1j * QBUF;
    time = startTime + codePhase / PM.Fs;
    while time < 1.0:
        GETREPLICA_PAC(replica_PAC, SAT, time);
        test.append(MATCHEDFILTER(BUF, replica_PAC))
        time = time + PM.RXBUFLEN / PM.Fs;

    return test;
def ACQUISITION_FOLD(RX, SAT, corr, startTime):
    isAcquired_FOLD = 0;
    isAcquired = 0;
    time = startTime;
    cnt = 0;
    replica = np.zeros(2 * PM.RXBUFLEN, dtype=int);

    ## acquisition with folded replica
    while isAcquired == 0:
        GETREPLICA_FOLD(replica, SAT, time)
        for FdIdx in range(PM.freqBins):
            corr[FdIdx, :] = FFTSEARCH(RX.IBuffer, RX.QBuffer, replica, FdIdx);

        peak = np.max(corr);

        if peak >= PM.T_FOLD:
            isAcquired_FOLD = 1;
        else:
            time = time + PM.FOLDFACTOR*PM.RXBUFLEN / PM.Fs;
            cnt += 1;

        dopplerBin, codePhase = FINDINARRAY(corr, peak, PM.freqBins, PM.RXBUFLEN);

        ## acquisition without folded replica
        if isAcquired_FOLD == 1:
            time = time + codePhase / PM.Fs;
            t = np.arange(0, PM.RXBUFLEN / PM.Fs, 1 / PM.Fs);
            IBUF = np.cos(2 * np.pi * (PM.F - PM.MAXFd + PM.STEPFd * dopplerBin) * t) * np.transpose(RX.IBuffer);
            QBUF = np.sin(2 * np.pi * (PM.F - PM.MAXFd + PM.STEPFd * dopplerBin) * t) * np.transpose(RX.QBuffer);
            BUF = IBUF + 1j * QBUF;
            replica2 = np.zeros(PM.RXBUFLEN);
            for i in range(PM.FOLDFACTOR):
                GETREPLICA_PAC(replica2, SAT, time + 2 * i * PM.RXBUFLEN / PM.Fs);
                test = MATCHEDFILTER(BUF, replica2);

                if test >= PM.T_PAC:
                    isAcquired = 1;
                    break;

    time = time + (i * PM.RXBUFLEN - 1) / PM.Fs;
    codePhase = TIME2SAMPLE(time) % PM.CODELEN;
    peak = test;

    return dopplerBin, codePhase, peak, time, cnt;


def Ploting():
    global corr_OS, corr_PAC, corr_ENC
    index = '0'

    fig1 = plt.figure()
    ax1 = fig1.gca(projection='3d');
    df = pd.DataFrame(corr_OS);
    x = df.columns;
    y = df.index;
    X, Y = np.meshgrid(x, y);
    Z = df;
    surf1 = ax1.plot_surface(X, Y, Z, cmap='bwr')
    plt.savefig('../static/img/CA_'+index+'.png')

    fig2 = plt.figure()
    ax2 = fig2.gca(projection='3d');
    df = pd.DataFrame(corr_PAC);
    x = df.columns;
    y = df.index;
    X, Y = np.meshgrid(x, y);
    Z = df;
    surf2 = ax2.plot_surface(X, Y, Z, cmap='bwr')
    plt.savefig('../static/img/PAC_'+index+'.png')


    fig3 = plt.figure()
    ax3 = fig3.gca(projection='3d');
    df = pd.DataFrame(corr_ENC);
    x = df.columns;
    y = df.index;
    X, Y = np.meshgrid(x, y);
    Z = df;
    surf3 = ax3.plot_surface(X, Y, Z, cmap='bwr')
    plt.savefig('../static/img/EC_'+index+'.png')

    # plt.show();