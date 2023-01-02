function [amp_total_pow, amp_env_mean, amp_env_std,...
    amp_skew, amp_kurt] = amplitudeFeatures(x)

% Calculates several amplitude features
%   x - the input signal
%
% amp_total_pow - amplitude total power
% amp_env_mean - mean of the envelope
% amp_env_std - standard deviation of the envelope
% amp_skew - skewness of the signal amplitude
% amp_kurt - kurtosis of the signal amplitude


% power of the signal (amplitude)
amp_total_pow = mean(abs(x).^2);
        

% mean and standard deviation of the amplitude envelopes
env = abs(hilbert(x)).^2;       
amp_env_mean = mean(env);
amp_env_std = std(env);

% skewness and kurtosis of the signal
amp_skew = abs(skewness(x));
amp_kurt = kurtosis(x);

end

