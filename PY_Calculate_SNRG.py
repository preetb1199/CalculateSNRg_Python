import numpy as np
import matplotlib.pyplot as plt
import math

# Settings (0 = off, 1 = on)
ParsevalSNR = 1
Animation = 0
NequalNseg = 0
PlotSNRvsParam = 1
PlotLastTimeSegment = 0
PlotTSvsg_log = 0
OutputFile1 = 0
OutputFile2 = 0
OutputFile3 = 0
OutputFile4 = 0

if OutputFile3 == 1:
    global StepSize_in
    global StepSize_invec

TS = 'fdot'  # Choose 'f' or 'fdot' ### Changes to 'f' or 'epsilon'

# Generate Data Spectrogram

# {SNRg | M<=g<=N} will be calculated
M = 1
N = 1
Ntrial = 10
if OutputFile1 == 0 and OutputFile4 == 0:
    Ntrial = 1

# Set information on noise samples
nNoiseSample = 10
noiseOffset = 100
nBinSide = 4

# Sampling frequency
fsamp = 32.0

# Noise ASD ( / sqrt(Hz) )
hnoise = 12
noiseamp = hnoise

# Signal amplitude (no antenna pattern)
hamp = 1

# Signal initial frequency (Hz)
f_sig = 10.0

# Signal frequency derivative (Hz/s)    ### Any particular epsilon? 
fdot_sig = -5.0e-6      

# Length of observation (hr)                 ### Any changes to observation time/ Coherence time? First fdot -> Coh time
Tobs_hr = 24.0
Tobs = Tobs_hr * 3600.0

# Coherence time (hr) - choose so that signal drifts 0.5 bin per coherence time
if abs(fdot_sig) > 0:
    Tcoh_hr = 1.0 / 3600.0 * np.sqrt(0.5 / abs(fdot_sig))
else:
    Tcoh_hr = 1.0 / 3600.0 * np.sqrt(0.5 / abs(-5.0e-6))

Tcoh = int(Tcoh_hr * 3600.0)

# Adjust coherence time to be an integer number of seconds
Tcoh = math.floor(Tcoh)
Tcoh_hr = Tcoh / 3600.0

# Adjust observation time to be exactly an integer number of coherence times to avoid binning headaches
Nseg = np.floor(Tobs/Tcoh)
if NequalNseg == 1:
    N = int(Nseg)
    
Tobs = Nseg * Tcoh
Tobs_hr = Tobs / 3600.0
print(f'Observation time = {Tobs} sec ({Tobs_hr} hr)')
print(f'Coherence time per segment = {Tcoh} sec ({Tcoh_hr} hr)')
assert M <= N, 'Error: M>N'

if OutputFile1 == 1:
    if ParsevalSNR == 1:
        filename1 = f'SNRgfdot_{fdot_sig:.2e}Tobs_{Tobs_hr:.0f}hrnoiseamp_{hnoise:.2e}sqrt(s)sigamp_{hamp:.2e}P.csv'
    else:
        filename1 = f'SNRgfdot_{fdot_sig:.2e}Tobs_{Tobs_hr:.0f}hrnoiseamp_{hnoise:.2e}sqrt(s)sigamp_{hamp:.2e}.csv'

if OutputFile4 == 1:
    if ParsevalSNR == 1:
        filename4 = f'SNRgDenominatorfdot_{fdot_sig:.2e}Tobs_{Tobs_hr:.0f}hrnoiseamp_{hnoise:.2e}sqrt(s)sigamp_{hamp:.2e}P.csv'
    else:
        filename4 = f'SNRgDenominatorfdot_{fdot_sig:.2e}Tobs_{Tobs_hr:.0f}hrnoiseamp_{hnoise:.2e}sqrt(s)sigamp_{hamp:.2e}.csv'


# Read model parameters from CSV file
SNRgModelParams = np.loadtxt('SNRgModelParams.csv', delimiter=',')

# Template spacing model function                      ### Any base that I should start with? 
def TS_model(a, g, Tobs, fdot):
    return 10**a[3] * Tobs**a[0] * g**a[1] * abs(fdot)**a[2]

# Create vector of f/fdot values (for search)
searchScale = 10
fStepSize = TS_model(SNRgModelParams[0, :], (M + N) / 2, Tobs_hr, fdot_sig)

if abs(fdot_sig) > 0:
    fdotStepSize = TS_model(SNRgModelParams[1, :], (M + N) / 2, Tobs_hr, fdot_sig)
else:
    fdotStepSize = TS_model(SNRgModelParams[1, :], (M + N) / 2, Tobs_hr, -5.e-6)

if TS == 'f':
    fvec = np.arange(f_sig - searchScale * fStepSize, f_sig + searchScale * fStepSize, fStepSize)
    fdotvec = np.array([fdot_sig])
    SNRgArray = np.zeros((N - M + 1, len(fvec)))
else:
    fvec = np.array([f_sig])
    fdotvec = np.arange(fdot_sig - searchScale * fdotStepSize, fdot_sig + searchScale * fdotStepSize, fdotStepSize)
    SNRgArray = np.zeros((N - M + 1, len(fdotvec)))

# Set noise power mean from Parseval's Theorem
noiseMeanPower = Tcoh * fsamp * noiseamp**2

# Define time series to hold raw data stream of signal plus noise
deltat = 1.0 / fsamp
t = np.arange(0, Tobs+deltat, deltat)
#t = np.arange(0, Tobs+deltat, deltat)
Nsample = len(t)

# Search bandwidth (Hz)
bandscale = 115  # Increase if fIndex is too high/low
f_siglo = min(f_sig, f_sig + fdot_sig * Tobs)
f_sighi = max(f_sig, f_sig + fdot_sig * Tobs)

freqlo_approx = (min(fvec) + min(fdotvec) * Tobs - 
                 (noiseOffset + nNoiseSample * (2 * nBinSide + 1)) / Tcoh - 
                 bandscale / Tcoh)
freqhi_approx = (max(fvec) + max(fdotvec) * Tobs * (max(fdotvec) > 0) + 
                 (noiseOffset + nNoiseSample * (2 * nBinSide + 1)) / Tcoh + 
                 bandscale / Tcoh)
freqhi = np.ceil(freqhi_approx * Tcoh) / Tcoh
freqlo = np.floor(freqlo_approx * Tcoh) / Tcoh
bandwidth = freqhi - freqlo

print(f'Low/high frequencies of band shown: {freqlo}-{freqhi} Hz')

# Number of bins signal drifts per coherence time
Nbin_drift = abs(fdot_sig) * Tcoh * Tcoh
Nbin_drift_total = abs(fdot_sig) * Tobs * Tcoh
print(f'Number of bins signal drifts per coherence time = {Nbin_drift:.2f}')
print(f'Total number of bins signal drifts = {Nbin_drift_total:.2f}')


###Funtionss

# Calculate root mean square h_0 for pure signal spectrogram
def h0rms(S, fIndex, Nseg, nBinSide):
    h_0sqr = 0
    for tIndex in range(Nseg):
        h_0sqr += np.sum(np.abs(S[fIndex[tIndex] - nBinSide : fIndex[tIndex] + nBinSide + 1, tIndex])**2)
    h_0 = np.sqrt(h_0sqr / Nseg)
    return h_0

# Calculate excess segments
def gbar(g, Nseg):
    return Nseg - np.floor((Nseg / g)) * g

# Calculate mean noise detection statistic from weights and signal spectrogram
def noise_mean_DS(A, M, N, Nseg):
    g = np.arange(M, N + 1)
    return (np.floor(Nseg / g) + gbar(g, Nseg)) * A

# Calculate standard deviation of noise detection statistic from weights and signal spectrogram
def noise_std_DS(A, M, N, Nseg):
    g = np.arange(M, N + 1)
    var_n = (np.floor(Nseg / g) * g**2 + gbar(g, Nseg)**2) * A**2
    sigma_n = np.sqrt(var_n)
    return sigma_n

# Calculate standard deviation of noise detection statistic from weights and signal spectrogram
def pred_signal_DS(h_0, A, M, N, Nseg):
    g = np.arange(M, N + 1)
    D_s = (np.floor(Nseg / g) * g**2 + gbar(g, Nseg)**2) * h_0**2 + noise_mean_DS(A, M, N, Nseg)
    return D_s




for trial in range(Ntrial):
    #print(Ntrial)

    # Generate noise
    noise = noiseamp * 0.5
    #noise = noiseamp * np.random.normal(0., 1., Nsample)

    #print(noise.shape)

    # Generate signal in time domain
    signal = hamp * np.sin(2 * np.pi * (f_sig * t + 0.5 * fdot_sig * t**2))
    #print(signal.shape)
    data = signal + noise

    # Generate spectra for each coherence time and extract band of interest to make spectrogram
    indbandlo = int(np.floor(freqlo * Tcoh)) 
    #print(f'indbandlo = {indbandlo}')
    indbandhi = int(np.floor(freqhi * Tcoh)) + 1 
    #print(f'indbandhi = {indbandhi}')
    nbandbin = int(indbandhi - indbandlo)
    #print(f'nbandbin = {nbandbin}')
    Nseg = int(np.floor(Tobs/Tcoh))
    print(Nsample, Nseg)
    Nsample_coh = int(np.floor(Nsample/Nseg))
    spectrogram = np.zeros((nbandbin, Nseg))
    rawSpectrogram = np.zeros((nbandbin, Nseg), dtype=complex)
    
    for seg in range(Nseg):
        print(f'Generating segment {seg + 1}')
        indlo = seg * Nsample_coh 
        #print(f'indlo = {indlo}')
        #indhi = indlo + Nsample_coh 
        indhi = indlo + Nsample_coh 
        #print(f'indhi = {indhi}')
        segment = data[indlo:indhi]
        #segplus = seg +1 
        #if segplus <= 5:
            #print(segment)
            #np.savetxt(f"segment_{seg+1}.txt", segment)
        segment_noNoise = signal[indlo:indhi]
        rawfft = np.fft.fft(segment, n=Nsample_coh)
        
        spectrogram[:, seg] = np.abs(rawfft[indbandlo:indbandhi])
        rawSpectrogram[:, seg] = rawfft[indbandlo:indbandhi]
    
        #dummyVariable = np.max(np.abs(rawfft[indbandlo:indbandhi]))
        #print(dummyVariable)
        #phaseVariable
        fIndex_sig = np.argmax(np.abs(rawfft[indbandlo:indbandhi]))
        #print(fIndex_sig)
        dummyVariable = rawSpectrogram[fIndex_sig,seg]
              
        if seg + 1 in [round(Nseg / 5), round(Nseg / 5 * 2), round(Nseg / 5 * 3), round(Nseg / 5 * 4), Nseg]:
            print(f'Signal fIndex of segment {seg + 1} = {fIndex_sig}; magnitude = {np.abs(dummyVariable)}; phase = {np.angle(dummyVariable)}')
            print(t[indlo], t[indhi-1], segment[0], segment[-1], indhi, indlo)
            #plt.figure()
            #plt.plot(t[indbandlo:indbandhi]),np.abs(rawfft[indbandlo:indbandhi]))

            #plt.plot(np.arange(indbandlo,indbandhi),np.abs(rawfft[indbandlo:indbandhi]))
            

    #plt.scatter(np.arange(nbandbin),np.abs(rawSpectrogram[:,-1]))
    #plt.ylim(0,1000)
    
    # Create modified spectrogram for plot
    plotSpectrogram = spectrogram.copy()
    for k in range(Nseg):
        segment = plotSpectrogram[:, k]
        segment[segment < 0.1 * np.max(segment)] = 0
        plotSpectrogram[:, k] = segment

    
    # Plot spectrogram
    if Nseg > 1:
        segarray = np.arange(1, Nseg + 1)
        seghour = (segarray - 1) * Tcoh / 3600
        indarray = np.arange(indbandlo, indbandhi)
        freqplot = (indarray - indbandlo) * 1.0 / Tcoh + freqlo

        print(f'Creating spectrogram plot with {Nseg} time bins (columns) and {nbandbin} frequency bins (rows)...')
        plt.figure(1)
        plt.contourf(seghour, freqplot, plotSpectrogram, cmap='viridis')
        plt.title('Spectrogram', fontsize=20)
        plt.xlabel('Time (hr)', fontsize=20)
        plt.ylabel('Frequency (Hz)', fontsize=20)
        plt.colorbar()
        plt.show()
    

    # Calculate Detection Statistics

    # Create vector of indices for frequency trajectory in the middle of each time segment
    freqTrajIndex = np.round((np.arange(1, Nseg + 1) - 0.5) * Nsample_coh).astype(int)
    #freqTrajIndex = np.round(((np.arange(Nseg) - 0.5) * Nsample_coh)).astype(int)

    tMid = t[freqTrajIndex]

    # Calculate expected phase/frequency trajectories and signal
    freqTraj = f_sig + fdot_sig * tMid
    phaseTraj = 2 * np.pi * (f_sig * t + 0.5 * fdot_sig * t**2)
    searchSignal = hamp * np.sin(phaseTraj)

    # Create template spectrogram
    searchSpectrogram = np.zeros((nbandbin, Nseg), dtype=complex)
    for seg in range(Nseg):
        indlo = seg * Nsample_coh
        #indhi = indlo + Nsample_coh + 1    #Made a changeee!!!
        indhi = indlo + Nsample_coh
        searchSegment = searchSignal[indlo:indhi]
        searchRawfft = np.fft.fft(searchSegment, n=Nsample_coh)
        searchSpectrogram[:nbandbin, seg] = searchRawfft[indbandlo:indbandhi]

    # Calculate signal frequency indices
    fIndex = np.round((freqTraj - freqlo) * Tcoh).astype(int)
    

    #for seg in range(Nseg):
        #fIndex_sig = np.argmax(np.abs(rawSpectrogram[:,seg]))
        #print(f"fIndex_sig {fIndex_sig}")
        #print(f"fIndex[seg] {fIndex[seg]}")

    # Calculate background noise frequency indices
    fIndex_noise = np.zeros((nNoiseSample, Nseg), dtype=int)
    for k in range(nNoiseSample):
        fIndex_noise[k, :] = fIndex + (2 * nBinSide + 2) * (k + 1) + noiseOffset

    # Calculate weightedFFTs array
    weight = np.zeros(Nseg)
    weightedFFT = np.zeros((nbandbin, Nseg), dtype=complex)
    for tIndex in range(Nseg):
        weight[tIndex] = 1 / np.sqrt(np.sum(np.abs(searchSpectrogram[(fIndex[tIndex] - nBinSide):(fIndex[tIndex] + nBinSide + 1), tIndex])**2))

        #weight[tIndex] = 1 / np.sqrt(np.sum(np.abs(searchSpectrogram[fIndex[tIndex] - nBinSide:fIndex[tIndex] + nBinSide + 1, tIndex])**2))
        weightedFFT[:, tIndex] = weight[tIndex] * searchSpectrogram[:, tIndex]

    # Calculate detection statistics
    signalDSg = np.zeros(N - M + 1)
    grouper_sig = np.zeros(N - M + 1, dtype=complex)
    noiseDSg = np.zeros((N - M + 1, nNoiseSample), dtype=complex)
    grouper_noise = np.zeros((N - M + 1, nNoiseSample), dtype=complex)

    signalDSContribution1 = []
    signalDSContribution2 = []
    signalDSContribution3 = []
    signalDSContribution4 = []

    noiseDSContribution1 = []
    noiseDSContribution2 = []
    noiseDSContribution3 = []
    noiseDSContribution4 = []

    for tIndex in range(Nseg):
        grouper_sig += np.sum(np.conj(weightedFFT[(fIndex[tIndex] - nBinSide):(fIndex[tIndex] + nBinSide + 1), tIndex]) *
                              rawSpectrogram[(fIndex[tIndex] - nBinSide):(fIndex[tIndex] + nBinSide + 1), tIndex])
        #grouper_sig += np.sum(np.conj(weightedFFT[fIndex[tIndex] - nBinSide:fIndex[tIndex] + nBinSide + 1, tIndex]) * rawSpectrogram[fIndex[tIndex] - nBinSide:fIndex[tIndex] + nBinSide + 1, tIndex])
        for g in range(M, N + 1):
            contribution = np.abs(grouper_sig[g - M])**2
            if tIndex % g == 0 or tIndex == Nseg - 1:
                signalDSg[g - M] += contribution
                if g == 1:
                    signalDSContribution1.append(contribution)
                elif g == 2:
                    signalDSContribution2.append(contribution)
                elif g == 3:
                    signalDSContribution3.append(contribution)
                elif g == 4:
                    signalDSContribution4.append(contribution)
                grouper_sig[g - M] = 0

        for k in range(nNoiseSample):
            grouper_noise[:, k] += np.sum(np.conj(weightedFFT[(fIndex[tIndex] - nBinSide):(fIndex[tIndex] + nBinSide+1), tIndex]) *
                                          rawSpectrogram[(fIndex_noise[k, tIndex] - nBinSide):(fIndex_noise[k, tIndex] + nBinSide + 1), tIndex])
        
            #grouper_noise[:, k] += np.sum(np.conj(weightedFFT[fIndex[tIndex] - nBinSide:fIndex[tIndex] + nBinSide + 1, tIndex]) * rawSpectrogram[fIndex_noise[k, tIndex] - nBinSide:fIndex_noise[k, tIndex] + nBinSide + 1, tIndex])
        for g in range(M, N + 1):
            contribution = np.abs(grouper_noise[g - M, :])**2
            if tIndex % g == 0 or tIndex == Nseg - 1:
                noiseDSg[g - M, :] += contribution
                if g == 1:
                    noiseDSContribution1.append([np.abs(grouper_noise[g - M, 0])**2, np.abs(grouper_noise[g - M, nNoiseSample // 2])**2, np.abs(grouper_noise[g - M, -1])**2])
                elif g == 2:
                    noiseDSContribution2.append([np.abs(grouper_noise[g - M, 0])**2, np.abs(grouper_noise[g - M, nNoiseSample // 2])**2, np.abs(grouper_noise[g - M, -1])**2])
                elif g == 3:
                    noiseDSContribution3.append([np.abs(grouper_noise[g - M, 0])**2, np.abs(grouper_noise[g - M, nNoiseSample // 2])**2, np.abs(grouper_noise[g - M, -1])**2])
                elif g == 4:
                    noiseDSContribution4.append([np.abs(grouper_noise[g - M, 0])**2, np.abs(grouper_noise[g - M, nNoiseSample // 2])**2, np.abs(grouper_noise[g - M, -1])**2])
                grouper_noise[g - M, :] = 0


    
                
    # Calculate SNRs and Plot
    SNRg = np.zeros(N - M + 1)
    noiseMeanDS = noise_mean_DS(noiseMeanPower, M, N, Nseg)
    noiseStdDS = noise_std_DS(noiseMeanPower, M, N, Nseg)
    for g in range(M, N + 1):
        if ParsevalSNR == 0:
            SNRg[g - M] = np.abs(signalDSg[g - M] - np.mean(noiseDSg[g - M, :])) / np.std(noiseDSg[g - M, :], ddof = 1)
        elif ParsevalSNR == 1:
            SNRg[g - M] = np.abs(signalDSg[g - M] - noiseMeanDS[g - M]) / noiseStdDS[g - M]
        print(f'SNR_{g} = {SNRg[g - M]:.2e}')

    if OutputFile1 == 1:
        assert all(np.arange(M, N + 1) == np.arange(1, Nseg + 1)), 'Error: Output File 1 can only function when M=1 and N=Nseg'
        with open(filename1, 'a') as fout1:
            fout1.write(','.join([f'{snr:.2e}' for snr in SNRg]) + '\n')
    
    if OutputFile4 == 1:
        assert all(np.arange(M, N + 1) == np.arange(1, Nseg + 1)), 'Error: Output File 4 can only function when M=1 and N=Nseg'
        with open(filename4, 'a') as fout4:
            if ParsevalSNR == 0:
                fout4.write(','.join([f'{np.std(noiseDSg[g - M, :], ddof =1):.2e}' for g in range(M, N + 1)]) + '\n')
            else:
                fout4.write(','.join([f'{np.sqrt(g) * noiseStdDS:.2e}' for g in range(M, N + 1)]) + '\n')





# Calculate predicted signal detection statistic
h0 = h0rms(searchSpectrogram, fIndex, Nseg, nBinSide)
predSignalDS = pred_signal_DS(h0, noiseMeanPower, M, N, Nseg)
print(predSignalDS)

gVals = np.arange(M, N + 1)
perfgVals = gVals[(Nseg % gVals) == 0]
fineg = np.linspace(M, N, 100 * Nseg)



# Create the plot
plt.figure(2)

# Scatter plot p1
plt.scatter(np.arange(M, N + 1), SNRg, color='black', s = 50, label='SNR$_g$', edgecolors='black')

# Scatter plot p3
predSignalDS_adjusted = (np.abs(predSignalDS[perfgVals - M] - noiseMeanPower * Nseg) / noiseMeanPower / np.sqrt(Nseg * perfgVals))    #Needs to be adjusted. 
plt.scatter(perfgVals, predSignalDS_adjusted, color='lightblue', s=200, marker='p', label='Predicted $g^*$', edgecolors='black')

#print(predSignalDS_adjusted)

# Plot p4
plt.plot(fineg, np.sqrt(fineg) * np.abs(predSignalDS[0] - noiseMeanPower * Nseg) / 
         noiseMeanPower / np.sqrt(Nseg), linestyle='--', color='lightblue', label='Predicted')

# Plot p5
plt.plot(gVals, np.abs(predSignalDS - noiseMeanDS) / noiseStdDS, color='red', linestyle='-', linewidth=2, label='Predicted')

# Customizing the plot
plt.title('SNR$_g$ vs $g$', fontsize=22)
plt.xlabel('$g$', fontsize=22)
plt.ylabel('SNR$_g$', fontsize=22)
plt.legend(loc='upper left', frameon=False, fontsize='x-large')  # Corrected legend location
plt.grid(True)

# Customize tick and line width
for axis in ['top', 'bottom', 'left', 'right']:
    plt.gca().spines[axis].set_linewidth(3)
plt.gca().tick_params(width=3)

plt.show()



# Find and print the maximum SNR_g value and its corresponding g value
gmax = np.max(SNRg)
SNR_gmax = gVals[np.argmax(SNRg)]
print(f'Max of {gmax:.2e} at g = {SNR_gmax}')

# Print Predictions of SNR from Parseval's Theorem

if M == 1:
    print('\n\t\t***\n')
    print('Signal DS_1 (empirical) = {:e}'.format(signalDSg[0]))
    print('Signal DS_1 (predicted) = {:e}'.format(predSignalDS[0])) #need to be checked
    print('Mean of Noise DS_1 (empirical) {:e}'.format(np.mean(noiseDSg[0, :]))) #need to be checked
    print('Mean of Noise DS_1 (predicted) {:e}'.format(noiseMeanDS[0]))
    #print(np.std([1,2,3], ddof=1))
    print('Standard Deviation of Noise DS_1 (empirical) = {:e}'.format(np.std(noiseDSg[0, :], ddof = 1))) #need to be checked
    print('Standard Deviation of Noise DS_1 (predicted) = {:e}'.format(noiseStdDS[0])) 
    SNRg_empirical = abs(signalDSg[0] - np.mean(noiseDSg[0, :])) / np.std(noiseDSg[0, :], ddof = 1)
    SNRg_predicted = abs(predSignalDS[0] - noiseMeanDS[0]) / noiseStdDS[0]
    print('SNRg_1 (empirical) = {:e}'.format(SNRg_empirical)) #need to check
    print('SNRg_1 (predicted) = {:e}'.format(SNRg_predicted)) #need to check
    print('\t\t***\n')



# Plot DS Contributions
if NequalNseg == 1 and M == 1 and Nseg >= 4:
    # Create segment vectors for each SNRg
    segvecs = [[], [], [], []]
    for tIndex in range(1, Nseg + 1):
        for g in range(1, 5):
            if tIndex % g == 0:
                segvecs[g - 1].append(tIndex)
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Plot Signal DS Contributions for SNRg
    axs[0, 0].plot(segvecs[0], signalDSContribution1, '-r', label='g=1')
    axs[0, 0].plot(segvecs[1], signalDSContribution2, '-g', label='g=2')
    axs[0, 0].plot(segvecs[2], signalDSContribution3, '-b', label='g=3')
    axs[0, 0].plot(segvecs[3], signalDSContribution4, '-m', label='g=4')
    axs[0, 0].set_ylim([0, 17])
    axs[0, 0].set_title('Signal DS Contributions for SNRg')
    axs[0, 0].set_xlabel('time segment')
    axs[0, 0].set_ylabel('contribution')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    axs[0, 0].tick_params(axis='both', which='major', labelsize=15)
    axs[0, 0].spines['top'].set_linewidth(3)
    axs[0, 0].spines['right'].set_linewidth(3)
    axs[0, 0].spines['left'].set_linewidth(3)
    axs[0, 0].spines['bottom'].set_linewidth(3)

    # Plot Noise DS Contributions for SNRg (first noise sample)
    axs[0, 1].plot(segvecs[0], noiseDSContribution1[0, :], '-r', label='g=1')
    axs[0, 1].plot(segvecs[1], noiseDSContribution2[0, :], '-g', label='g=2')
    axs[0, 1].plot(segvecs[2], noiseDSContribution3[0, :], '-b', label='g=3')
    axs[0, 1].plot(segvecs[3], noiseDSContribution4[0, :], '-m', label='g=4')
    axs[0, 1].set_title('Noise DS Contributions for SNRg (first noise sample)')
    axs[0, 1].set_xlabel('time segment')
    axs[0, 1].set_ylabel('contribution')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    axs[0, 1].tick_params(axis='both', which='major', labelsize=15)
    axs[0, 1].spines['top'].set_linewidth(3)
    axs[0, 1].spines['right'].set_linewidth(3)
    axs[0, 1].spines['left'].set_linewidth(3)
    axs[0, 1].spines['bottom'].set_linewidth(3)

    # Plot Noise DS Contributions for SNRg (middle noise sample)
    axs[1, 0].plot(segvecs[0], noiseDSContribution1[1, :], '-r', label='g=1')
    axs[1, 0].plot(segvecs[1], noiseDSContribution2[1, :], '-g', label='g=2')
    axs[1, 0].plot(segvecs[2], noiseDSContribution3[1, :], '-b', label='g=3')
    axs[1, 0].plot(segvecs[3], noiseDSContribution4[1, :], '-m', label='g=4')
    axs[1, 0].set_title('Noise DS Contributions for SNRg (middle noise sample)')
    axs[1, 0].set_xlabel('time segment')
    axs[1, 0].set_ylabel('contribution')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=15)
    axs[1, 0].spines['top'].set_linewidth(3)
    axs[1, 0].spines['right'].set_linewidth(3)
    axs[1, 0].spines['left'].set_linewidth(3)
    axs[1, 0].spines['bottom'].set_linewidth(3)

    # Plot Noise DS Contributions for SNRg (last noise sample)
    axs[1, 1].plot(segvecs[0], noiseDSContribution1[2, :], '-r', label='g=1')
    axs[1, 1].plot(segvecs[1], noiseDSContribution2[2, :], '-g', label='g=2')
    axs[1, 1].plot(segvecs[2], noiseDSContribution3[2, :], '-b', label='g=3')
    axs[1, 1].plot(segvecs[3], noiseDSContribution4[2, :], '-m', label='g=4')
    axs[1, 1].set_title('Noise DS Contributions for SNRg (last noise sample)')
    axs[1, 1].set_xlabel('time segment')
    axs[1, 1].set_ylabel('contribution')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=15)
    axs[1, 1].spines['top'].set_linewidth(3)
    axs[1, 1].spines['right'].set_linewidth(3)
    axs[1, 1].spines['left'].set_linewidth(3)
    axs[1, 1].spines['bottom'].set_linewidth(3)

    plt.tight_layout()
    plt.show()




SNRgArray_new = []
for r in range(len(fvec)):
    for j in range(len(fdotvec)):
        
        if TS == 'f':
            print(f'Calculating SNRgs for df = {fvec[r] - f_sig} Hz')
        else:
            print(f'Calculating SNRgs for dfdot = {fdotvec[j] - fdot_sig} Hz/s')
        
        # Calculate expected phase/frequency trajectories and signal
        freqTraj = fvec[r] + fdotvec[j] * tMid
        phaseTraj = 2 * np.pi * (fvec[r] * t + 0.5 * fdotvec[j] * t**2)
        searchSignal = hamp * np.sin(phaseTraj)
    
        # Create template spectrogram
        searchSpectrogram = np.zeros((nbandbin, Nseg), dtype=complex)
        for seg in range(Nseg):
            indlo = seg * Nsample_coh
            indhi = indlo + Nsample_coh
            searchSegment = searchSignal[indlo:indhi]
            searchRawfft = np.fft.fft(searchSegment, n=Nsample_coh)
            searchSpectrogram[:, seg] = searchRawfft[indbandlo:indbandhi]
    
        # Calculate signal frequency indices
        fIndex = np.round((freqTraj - freqlo) * Tcoh).astype(int)

        # Calculate background noise frequency indices
        if ParsevalSNR == 0:
            fIndex_noise = np.zeros((nNoiseSample, Nseg), dtype=int)
            for k in range(nNoiseSample):
                fIndex_noise[k, :] = fIndex + (2 * nBinSide + 2) * (k + 1) + noiseOffset
    
        # Calculate weightedFFTs array
        weight = np.zeros(Nseg)
        weightedFFT = np.zeros((nbandbin, Nseg), dtype=complex)
        for tIndex in range(Nseg):
            weight[tIndex] = 1 / np.sqrt(np.sum(np.abs(searchSpectrogram[fIndex[tIndex] - nBinSide:fIndex[tIndex] + nBinSide + 1, tIndex])**2))
            weightedFFT[:, tIndex] = searchSpectrogram[:, tIndex] * weight[tIndex]
        
        # Calculate detection statistics
        signalDSg = np.zeros(N - M + 1)
        grouper_sig = np.zeros(N - M + 1, dtype=complex)
        if ParsevalSNR == 0: 
            noiseDSg = np.zeros((N - M + 1, nNoiseSample), dtype=complex)
            grouper_noise = np.zeros((N - M + 1, nNoiseSample), dtype=complex)
    
        if ((j+1) * (r+1)-1)== 15:
            DScontributions15 = np.zeros(Nseg)
        elif ((j+1) * (r+1)-1) == 21:
            DScontributions21 = np.zeros(Nseg)
        elif ((j+1) * (r+1)-1) == 11:
            DScontribution11 = np.zeros(Nseg)
    
        for tIndex in range(Nseg):
            # Signal detection statistics
            grouper_sig += np.sum(np.conj(weightedFFT[fIndex[tIndex] - nBinSide:fIndex[tIndex] + nBinSide + 1, tIndex]) *
                                  rawSpectrogram[fIndex[tIndex] - nBinSide:fIndex[tIndex] + nBinSide + 1, tIndex])
            for g in range(M, N + 1):
                contribution = np.abs(grouper_sig[g - M])**2
                if tIndex % g == 0:
                    signalDSg[g - M] += contribution
                    if ((j+1) * (r+1)-1) == 15 and g == 1:
                        DScontributions15[tIndex] = contribution
                    elif ((j+1) * (r+1)-1) == 21 and g == 1:
                        DScontributions21[tIndex] = contribution
                    elif ((j+1) * (r+1)-1) == 11 and g == 1:
                        DScontribution11[tIndex] = contribution
                    grouper_sig[g - M] = 0
                elif tIndex == Nseg - 1:
                    signalDSg[g - M] += contribution
    
            # Noise detection statistics
            if ParsevalSNR == 0:
                for k in range(nNoiseSample):
                    grouper_noise[:, k] += np.sum(np.conj(weightedFFT[fIndex[tIndex] - nBinSide:fIndex[tIndex] + nBinSide + 1, tIndex]) *
                                                  rawSpectrogram[fIndex_noise[k, tIndex] - nBinSide:fIndex_noise[k, tIndex] + nBinSide + 1, tIndex])
                for g in range(M, N + 1):
                    contribution = np.sum(np.abs(grouper_noise[g - M, :])**2)
                    if tIndex % g == 0:
                        noiseDSg[g - M, :] = noiseDSg[g - M, :] + contribution
                        grouper_noise[g - M, :] = 0
                    elif tIndex == Nseg - 1:
                        noiseDSg[g - M, :] += contribution
        
        for g in range(M, N + 1):
            if ParsevalSNR == 0:
                SNRgArray[g - M, ((j+1) * (r+1)-1)] = np.abs(signalDSg[g - M] - np.mean(noiseDSg[g - M, :])) / np.std(noiseDSg[g - M, :], ddof =1)
            elif ParsevalSNR == 1:
                SNRgArray[g - M, ((j+1) * (r+1)-1)] = np.abs(signalDSg[g - M] - noiseMeanDS[g - M]) / noiseStdDS[g - M]
                #print(SNRgArray)
                #SNRgArray_new.append(SNRgArray[0,0]) ### NOTE HERE g-M =  0 ,; SPECIAL BEST CASE SCENARIO!!! WILL HAVE TO FLATTEN THE MATRIX!!! 

        
        # Compare search spectrogram and signal spectrogram of the last time segment
        if PlotLastTimeSegment == 1:
            freq_range = np.arange(fIndex[-1] - 10, fIndex[-1] + 10 + 1)

            plt.figure(1900 + j * r)
            plt.scatter(freq_range, np.abs(searchSpectrogram[freq_range, -1]), c='k', label='template')
            plt.plot(freq_range, np.abs(rawSpectrogram[freq_range, -1]), 'r', label='data')
            plt.axvline(fIndex[-1], ls='--', c='k', linewidth=1, label='fIndex')
            plt.title(f'Last Time Segment Fourier Coefficient Magnitude dfdot_sig = {fdotvec[j] - fdot_sig} Hz/s')
            plt.xlabel('frequency bin')
            plt.ylabel('magnitude')
            plt.legend()
            plt.grid(True)
            plt.show()

            plt.figure(2900 + j * r)
            plt.scatter(freq_range, np.unwrap(np.angle(searchSpectrogram[freq_range, -1])), c='k', label='template')
            plt.plot(freq_range, np.unwrap(np.angle(rawSpectrogram[freq_range, -1])), 'r', label='data')
            plt.axvline(fIndex[-1], ls='--', c='k', linewidth=1, label='fIndex')
            plt.title(f'Last Time Segment Fourier Coefficient Phase dfdot_sig = {fdotvec[j] - fdot_sig} Hz/s')
            plt.xlabel('frequency bin')
            plt.ylabel('phase')
            plt.legend()
            plt.grid(True)
            plt.show()

#print(SNRgArray_new)




if searchScale >= 20:
    plt.figure(3000)
    plt.plot(np.arange(1, Nseg + 1), DScontributions15, '-g', label='off bin with max SNR')
    plt.plot(np.arange(1, Nseg + 1), DScontributions21, '-m', label='center bin')
    plt.plot(np.arange(1, Nseg + 1), DScontribution11, '-b', label='off bin with lower SNR')
    
    plt.title('Center Bin and Off Bin DS Contributions(i == 1)')
    plt.xlabel('time segment')
    plt.ylabel('DS Contribution')
    plt.legend()
    plt.grid(True)
    
    ax = plt.gca()
    ax.tick_params(labelsize=16)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    
    plt.show()




# Create title strings
titleStrings = []
for g in range(M, N + 1):
    if TS == 'f':
        titleStrings.append('SNR$_g$ vs $\Delta f_0$')
    else:
        titleStrings.append('SNR$_g$ vs $\Delta \dot{f}$')

df = fvec - f_sig
dfdot = fdotvec - fdot_sig

# Plot SNRg_i vs i
for k in range(len(dfdot) * len(df)):
    plt.figure(3 + k)

    plt.scatter(range(M, N + 1), SNRgArray[0,k], color='black', label='SNR$_g$')

    if TS == 'f':
        plt.title(f'SNRg vs g for dfdot = {df[k]}')
    else:
        plt.title(f'SNRg vs g for dfdot = {dfdot[k]}')
    
    plt.xlabel('i')
    plt.ylabel('SNRg')
    plt.grid(True)

    ax = plt.gca()
    ax.tick_params(labelsize=16)
    for spine in ax.spines.values():
        spine.set_linewidth(3)
    
    plt.show()




from scipy.interpolate import UnivariateSpline

print(SNRgArray[0,0])
print(g-M)

if TS == 'f':
    dfp = np.linspace(np.min(df), np.max(df), int(1e4 * len(df)))
    dfdotp = dfdot
else:
    dfp = df
    dfdotp = np.linspace(np.min(dfdot), np.max(dfdot), int(1e4 * len(dfdot)))

SNRsplines = np.zeros((N - M + 1, len(dfdotp) * len(dfp)))
templateSpacings = np.zeros(N - M + 1)

for g in range(M, N + 1):
    if TS == 'f':
        spline_func = UnivariateSpline(df, SNRgArray[g-M, :], s=0)
        SNRsplines[g - M, :] = spline_func(dfp)
        indRight = np.argmin(np.abs(SNRsplines[g - M, (len(dfp) // 2):] - 0.8 * SNRgArray[g-M, searchScale]))
        indLeft = np.argmin(np.abs(SNRsplines[g - M, :len(dfp) // 2] - 0.8 * SNRgArray[g-M, searchScale]))
        TS_right = np.abs(dfp[indRight + len(dfp) // 2])
        TS_left = np.abs(dfp[indLeft])
        templateSpacings[g - M] = np.mean([TS_left, TS_right])
    else:
        spline_func = UnivariateSpline(dfdot, SNRgArray[g-M, :], s=0)
        SNRsplines[g - M, :] = spline_func(dfdotp)
        indRight = np.argmin(np.abs(SNRsplines[g - M, (len(dfdotp) // 2):] - 0.8 * SNRgArray[g-M, searchScale]))
        indLeft = np.argmin(np.abs(SNRsplines[g - M, :len(dfdotp) // 2] - 0.8 * SNRgArray[g-M, searchScale]))
        TS_right = np.abs(dfdotp[indRight + len(dfdotp) // 2])
        TS_left = np.abs(dfdotp[indLeft])
        templateSpacings[g - M] = np.mean([TS_left, TS_right])
    
    if PlotSNRvsParam == 1:
        plt.figure(3 + len(dfdot) * len(df) + g - M + 1)
        plt.subplot(1, 2, 1)
        if TS == 'f':
            plt.scatter(df, SNRgArray[g-M, :] / SNRgArray[g-M, searchScale], color='k')
            plt.xlabel(r'$\Delta f_0$ (Hz)', fontsize=14)
        else:
            plt.scatter(dfdot, SNRgArray[g-M, :] / SNRgArray[g-M, searchScale], color='k')
            plt.xlabel(r'$\Delta \dot{f}$ (Hz/s)', fontsize=14)
        plt.title(titleStrings[g - M], fontsize=14)
        plt.ylabel('SNR/SNR$_{max}$', fontsize=14)
        plt.grid(True)
        plt.gca().tick_params(width=3)
        plt.gca().tick_params(labelsize=12)

        plt.subplot(1, 2, 2)
        if TS == 'f':
            plt.scatter(df, SNRgArray[g-M, :], color='k')
            plt.xlabel(r'$\Delta f_0$ (Hz)', fontsize=14)
        else:
            plt.scatter(dfdot, SNRgArray[g-M, :], color='k')
            plt.xlabel(r'$\Delta \dot{f}$ (Hz/s)', fontsize=14)
        plt.title(titleStrings[g - M], fontsize=14)
        plt.ylabel('SNR', fontsize=14)
        plt.grid(True)
        plt.gca().tick_params(width=3)
        plt.gca().tick_params(labelsize=12)

        plt.subplot(1, 2, 1)
        if TS == 'f':
            plt.plot(dfp, SNRsplines[g - M, :] / SNRgArray[g-M, searchScale], label='spline')
        else:
            plt.plot(dfdotp, SNRsplines[g - M, :] / SNRgArray[g-M, searchScale], label='spline')
        plt.legend([f'g = {g}', 'spline'], loc='lower center')
        
        plt.subplot(1, 2, 2)
        if TS == 'f':
            plt.plot(dfp, SNRsplines[g - M, :], label='spline')
        else:
            plt.plot(dfdotp, SNRsplines[g - M, :], label='spline')
        plt.legend([f'g = {g}', 'spline'], loc='lower center')
        plt.show()



# Print template spacings
for g in range(M, N + 1):
    if TS == 'f':
        print(f'For g = {g}, stepping {templateSpacings[g - M]:.2e} Hz in f causes a 20% drop in SNRg_i')
    else:
        print(f'For g = {g}, stepping {templateSpacings[g - M]:.2e} Hz/s in fdot causes a 20% drop in SNRg_i')






