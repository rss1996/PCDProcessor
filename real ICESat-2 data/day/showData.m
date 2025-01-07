
clear all;
clc;

dataOri = load('ATL03_20190101040709_00570202_003_01_gt2l.Out.txt');

% [row,col]=find((dataOridata(:,5)>=0)&(dataOridata(:,5)<16000));
% dataOri=dataOridata(row,:);


%show the points according to the classified ID

[rowNoise,colNoise]=find(dataOri(:,4) == 0);
[rowSignal,colSignal]=find(dataOri(:,4) == 1);

noiseData = dataOri(rowNoise,:);
signalData = dataOri(rowSignal,:);

plot(noiseData(:,1),noiseData(:,2),'r.');
hold on;
plot(signalData(:,1),signalData(:,2),'g.');




%show the points according to the classified ID


% [rowNoise,colNoise]=find(dataOri(:,9) == 0);
% [rowSignal,colSignal]=find(dataOri(:,9) == 1);
% 
% noiseData = dataOri(rowNoise,:);
% signalData = dataOri(rowSignal,:);
% 
% plot(noiseData(:,4),noiseData(:,5),'r.');
% hold on;
% plot(signalData(:,4),signalData(:,5),'g.');