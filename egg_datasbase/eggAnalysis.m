% eggAnalysis.m - The purpose of this code is to perform adequate
% processing of EGG signal, to calculate dominant frequencies (DF), and to
% perform statistical analysis.
%
% INPUT:    EGG signals recorded in 20 subjects (.txt file)
% OUTPUT:   dominant frequencies (.csv file)
%
%   Written by Nenad B. Popovic (nenad.pop92@gmail.com) and
%   Nadica Miljkovic (nadica.miljkovic@etf.rs)
% 
% If you use this code, please, cite relevant paper and dataset as:
%   [1] Popovic, N.B., Miljkovic, N. and Popovic M.B., 2019. Simple gastric
%   motility assessment method with a single-channel electrogastrogram.
%   Biomedical Engineering/Biomedizinsche Technik, 64(2), pp.177-185, doi:
%   10.1515/bmt-2017-0218
%   [2] Popovic, N.B., Miljkovic, N. and Popovic M.B., 2020. Three-channel
%   surface electrogastrogram (EGG) dataset recorded during fasting and
%   post-prandial states in 20 heatlhy individuals [Data set]. Zenodo, doi:
%   10.5281/zenodo.3730617.
%
% GNU Octave, version 5.2.0
% Copyright (C) 2020 John W. Eaton and others.
% This is free software; see the source code for copying conditions.
% -------------------------------------------------------------------------

close all; clear all; clc;

% load Octave packages
pkg load signal; pkg load io; pkg load nan; % comment if you use Matlab

%% dominant peak analysis

% specify parameters
fs = 2; % Hz, sampling frequency
N = 4096; % number of points for FFT analysis

% memory allocation for Dominant Frequency (DF) calculation
df = zeros(20, 6);

% bandpass filter for noise reduction
[b, a] = butter(3, [0.03 0.25] / (fs / 2), 'bandpass');

% DF calculation
for ind = 1 : 20
    % FASTING -------------------------------------------------------------
    
    % signal loading
    file_name_f = ['ID' num2str(ind) '_fasting.txt'];
    dat_f = load(file_name_f);
    
    % EGG filtering
    ch1_f = filtfilt(b, a, dat_f(1:2400, 1)); % channel 1
    ch2_f = filtfilt(b, a, dat_f(1:2400, 2)); % channel 2
    ch3_f = filtfilt(b, a, dat_f(1:2400, 3)); % channel 3
    
    % FFT
    fch1_f = abs(fft(ch1_f, N)).^2; fftch1_f = fch1_f(1:(N/2+1));
    fch2_f = abs(fft(ch2_f, N)).^2; fftch2_f = fch2_f(1:(N/2+1));
    fch3_f = abs(fft(ch3_f, N)).^2; fftch3_f = fch3_f(1:(N/2+1));
    
    % automatic DF calculation
    [~, b1_f] = max(fftch1_f); df(ind, 1) = b1_f / 2048;
    [~, b2_f] = max(fftch2_f); df(ind, 2) = b2_f / 2048;
    [~, b3_f] = max(fftch3_f); df(ind, 3) = b3_f / 2048;
    
    % POSTPRANDIAL --------------------------------------------------------
    
    % signal loading
    file_name_p = ['ID' num2str(ind) '_postprandial.txt'];
    dat_p = load(file_name_p);

    % EGG filtering
    ch1_p = filtfilt(b, a, dat_p(1:2400, 1)); % channel 1
    ch2_p = filtfilt(b, a, dat_p(1:2400, 2)); % channel 2
    ch3_p = filtfilt(b, a, dat_p(1:2400, 3)); % channel 3
    
    % FFT
    fch1_p = abs(fft(ch1_p, N)).^2; fftch1_p = fch1_p(1:(N/2+1));
    fch2_p = abs(fft(ch2_p, N)).^2; fftch2_p = fch2_p(1:(N/2+1));
    fch3_p = abs(fft(ch3_p, N)).^2; fftch3_p = fch3_p(1:(N/2+1));
    
    % automatic DF calculation
    [~, b1_p] = max(fftch1_p); df(ind, 4) = b1_p / 2048;
    [~, b2_p] = max(fftch2_p); df(ind, 5) = b2_p / 2048;
    [~, b3_p] = max(fftch3_p); df(ind, 6) = b3_p / 2048;  
    
end

% conversion from Hz to cpm (cycles per minute)
df = df.*60;

% all peaks were individually inspected by visualization and manual
% corrections were provided for false DF values in order to provide true
% dominant peaks
df(4, 4) = 3.1934; % ID4 postprandial channel 1
df(4, 5) = 3.1348; % ID4 postprandial channel 2
df(4, 6) = 3.1348; % ID4 postprandial channel 3
df(6, 3) = 2.4900; % ID6 fasting channel 3
df(15, 4) = 2.2560; % ID15 postprandial channel 1
df(17, 1) = 2.9592; % ID17 fasting channel 1
df(17, 2) = 3.0469; % ID17 fasting channel 2
df(17, 3) = 3.0762; % ID17 fasting channel 3

%% saving DF values in a file
filename = 'df.csv';
csvwrite(filename, df)

%% paired-sample t-tests

% for all subjects ID1-ID20
[h1, p1] = ttest( df(:, 1), df(:, 4) );
[h2, p2] = ttest( df(:, 2), df(:, 5) );
[h3, p3] = ttest( df(:, 3), df(:, 6) );

% for subjects with lower body mass index (BMI)
df_low(1:10, :) = [df(1:3, :); df(9, :); df(12, :); df(14, :); df(16:19, :)];
[h4, p4] = ttest( df_low(:, 1), df_low(:, 4) );
[h5, p5] = ttest( df_low(:, 2), df_low(:, 5) );
[h6, p6] = ttest( df_low(:, 3), df_low(:, 6) );

% for subjects ID11-ID20 (higher BMI)
df_high(1:10, :) = [df(4:8, :); df(10:11, :); df(13, :); df(15, :); df(20, :)];
[h7, p7] = ttest( df_high(:, 1), df_high(:, 4) );
[h8, p8] = ttest( df_high(:, 2), df_high(:, 5) );
[h9, p9] = ttest( df_high(:, 3), df_high(:, 6) );

% show p-values (p) and test decisions (h)
p = [p1, p2, p3, p4, p5, p6, p7, p8, p9]
h = [h1, h2, h3, h4, h5, h6, h7, h8, h9]