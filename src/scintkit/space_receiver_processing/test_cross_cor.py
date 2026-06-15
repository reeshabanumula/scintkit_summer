import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp

#create artifical data with time difference of 25 seconds

time = np.arange(0,100,1) # in counts of 1 second

shift = np.random.randint(5,50)
startA = 35
startB = startA + shift

rA = np.ones(100) * 45
#add a scintillation or articial wave bump
rA[35:50] += np.array([0, 2, 5, 8, 10, 14, 18, 20, 18, 14, 10, 8, 5, 2, 0])

rB = np.ones(100) * 45
rB[60:75] += np.array([0, 2, 5, 8, 10, 14, 18, 20, 18, 14, 10, 8, 5, 2, 0])

noise_level = 1
rA += np.random.normal(0, noise_level, len(rA))
rB += np.random.normal(0, noise_level, len(rB))

plt.plot(time, rB)
plt.plot(time, rA)
plt.title('Reciever A and Reciever B vs Time')
plt.ylabel('signal')
plt.xlabel('time')
plt.show()

#develop cross correlation function

# x = sp.correlate(rA, rB, mode = 'full')
# print(x)

# plt.plot(x)
# plt.title('similarity vs time')
# plt.show()

# print(np.max(x)) #max correlation, needs to be normalized to 1
# print(np.argmax(x)) #index position of max correlation, how do you connect this to the shift between them

# lag = sp.correlation_lags(len(rA), len(rB), mode = 'full')

# #print(lag)
# print(lag[np.argmax(x)]) #tells me that the signal lines up at this index. now we must convert this index into a time step

# #we are getting a best lag of 0. this should not be the case because best lag would tell us how many indexs away from a proper correlation we are
# #this is because the constant value of 45 is dominating the correlation function and isnt allowing it to identify the flucuation

# print(np.mean(rA))
# print(np.mean(rB))

# # #remove the mean to emphasize the fluctuation

# cenA = rA - np.mean(rA)
# cenB = rB - np.mean(rB)
# x_cen =sp.correlate(cenA, cenB, mode = 'full')

# centered_lag = sp.correlation_lags(len(cenA), len(cenB), mode = 'full')
# print(centered_lag)
# print(centered_lag[np.argmax(x_cen)])

# #proper index was identified or proper shift
# plt.plot(centered_lag, x_cen)
# plt.xlabel('Lag (samples)')
# plt.ylabel('Correlation')
# plt.title('Cross Correlation')
# plt.show()

#shows max correlation at a left shift of 25 counts from signal B to reach singal A

# #normalize the correlation between -1 and 1
# #process: center the signal --> normalize it --> compute cross correlation --> gives you a value between -1 and 1

# #already centered

# #normalize signals
# normA = cenA/np.std(rA)
# normB = cenB/np.std(rB)

# print(normA)
# print(normB)

# #normalize correlation
# xnorm = np.correlate(normA, normB, mode = 'full')
# xnorm = xnorm/ (np.linalg.norm(normA) * np.linalg.norm(normB))


# norm_lag = sp.correlation_lags(len(normA),len(normB), mode = 'full')

# plt.plot(norm_lag, xnorm)
# plt.xlabel('lag')
# plt.ylabel('correlation  (normalized)')
# plt.title('Normalized Cross Correlation')
# plt.show()

# #now output results
# #identify : max correlation, and lag
# print(f'The max correlation is : {np.max(xnorm)}')
# print(f'The lag is: {norm_lag[np.argmax(xnorm)]}')

#create function to create normalized cross correlation

def cross_correlation(sig1, sig2):
    norm1 = (sig1 - np.mean(sig1))/np.std(sig1)
    norm2 = (sig2 - np.mean(sig2))/np.std(sig2)

    cor = np.correlate(norm1, norm2, mode = 'full')
    cor_norm = cor/ (np.linalg.norm(norm1) * np.linalg.norm(norm2))

    norm_lag = sp.correlation_lags(len(norm1),len(norm2), mode = 'full')

    plt.plot(norm_lag, cor_norm)
    plt.xlabel('lag')
    plt.ylabel('correlation  (normalized)')
    plt.title('Normalized Cross Correlation')
    plt.show()

    #identify : max correlation, and lag
    #figure out how to correlate lag to time shift

    max_corr = np.max(cor_norm)
    best_lag = norm_lag[np.argmax(cor_norm)]
    
    return max_corr, best_lag

correlation, lag_b = cross_correlation(rA, rB)

print(f'The max correlation is : {correlation}')
print(f'The lag is: {lag_b}')
error = abs(abs(lag_b) - shift)
print(f"Error: {error}")





    