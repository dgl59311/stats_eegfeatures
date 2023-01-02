function vc_mat2eph( data, srate, file)
%vc_mat2eph( data, srate, file) converts data matrix into Cartool-readable .eph file
%   data - NxM matrix (N channels, M time points)
%   srate - sampling rate
%   file - fullpath to the new file
% Vitaly Chicherov, EPFL 2014
% 17-02-2015 corrected dimentions of the data (data' in line 13)

h = fopen(file, 'w+');

try
    fprintf (h, '%d %d %d', int16([size(data,1), size(data,2), srate]));
    dlmwrite(file, data',  '-append', 'delimiter', ' ', 'roffset', 1, 'precision', '%8.4f');
    fclose(h);
catch err
    fclose(h);
    rethrow(err);
end
