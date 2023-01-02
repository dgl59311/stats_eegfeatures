
% Modified for aging EEG Lemon data set
function [flp,fhi]=FiltLims(typeF)
if typeF==1
%     delta 
    flp=1;
    fhi=4;
    elseif typeF==2
%     theta
    flp=4;
    fhi=8;
    elseif typeF==3
%     alpha
    flp=8;
    fhi=13;
    elseif typeF==4
%     beta
    flp=13;
    fhi=30;
    elseif typeF==5
%     gamma
    flp=30;
    fhi=45;
end
end







