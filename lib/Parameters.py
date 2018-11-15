import math
import numpy as np

TAPS = 10;                          # length of the shift register
CODELEN = int(2**TAPS) - 1;         # length of the spreading code
DATALEN = int(1e5);
POS = np.array([10, 9, 4, 1, 0]);   # Feedback loginc for the LFSR

F = int(1575.42*1e6);                   # carrier frequency
Fc = int(1.023e5);                  # chipping rate
Fs = 5e5;                           # sampling rate
#Fs = 5.01*Fc;
R = int(Fc/CODELEN);                # data rate

MAXFd = 500;                       # maximum magnitude of doppler frequency
STEPFd = 10;                        # step for finding doppler frequency
NUMSAT = 1;                         # number of satellite
NUMRX = 1;                          # number of recievers
INTEGRATIONS = 1;                   # number of integeration at the acquisition process

RXBUFLEN = int(math.floor(Fs*CODELEN/Fc*INTEGRATIONS));     # receiver buffer size
ENCBUFLEN = int(2**TAPS);
MSGLEN = 128;
KEYLEN = 128;

INITTIME = 1;


SNRdB = 1e10;

ttt = 31*5;

freqBins = int( 2*MAXFd/STEPFd + 1 );

T_OS = 2000;
T_ENC = 1000;
T_DAC = 1000;
T_PAC = 2000;
T_FOLD = 1500;

TRUE_ENC = 0;

FOLDFACTOR = 4;

mode = 1;       # 0: Open signal, 1: PAC signal, 2: Encrypted Code, 3: Encrypted Code with Folding