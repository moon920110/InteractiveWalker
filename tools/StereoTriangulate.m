%%
load('stereoParams_1124.mat');
dir = "../data/2022-11-24_16-34-02_testing_carpets_YH_lunge_1";

frontSub = readtable(dir + "/keypoints/keypoints2D/frontSub.csv");
sideSub = readtable(dir + "/keypoints/keypoints2D/sideSub.csv");

folderName = "/keypoints/keypoints3D/";
if not(exist(folderName, "dir"))
    mkdir(fullfile(dir, folderName))
end

for i =1:size(frontSub,1)
    X1(:,i)=table2array(frontSub(i,1:3:end))';
    Y1(:,i)=table2array(frontSub(i,2:3:end))';
end

for i =1:size(sideSub,1)
    X2(:,i)=table2array(sideSub(i,1:3:end))';
    Y2(:,i)=table2array(sideSub(i,2:3:end))';
end
%%
% frontSubTimeIdx=205;
% sideSubTimeIdx=349;
% 
% X1=X1(:,frontSubTimeIdx:end);
% Y1=Y1(:,frontSubTimeIdx:end);
% 
% X2=X2(:,sideSubTimeIdx:sideSubTimeIdx+694);
% Y2=Y2(:,sideSubTimeIdx:sideSubTimeIdx+694);

X1(16:19,:) =[];
Y1(16:19,:) =[];

X2(16:19,:) =[];
Y2(16:19,:) =[];

%%
for i = 1:size(X1,1)
    outX1 = find(X1(i,:)==0);
    X1(i,outX1)=NaN;
    X1(i,:)=fillmissing(X1(i,:),'linear');
    outY1 = find(Y1(i,:)==0);
    Y1(i,outY1)=NaN;
    Y1(i,:)=fillmissing(Y1(i,:),'linear');
    outX2 = find(X2(i,:)==0);
    X2(i,outX2)=NaN;
    X2(i,:)=fillmissing(X2(i,:),'linear');
    outY2 = find(Y2(i,:)==0);
    Y2(i,outY2)=NaN;
    Y2(i,:)=fillmissing(Y2(i,:),'linear');
end

%%
    Rx = rotx(-90);
    for i = 1:length(X1)
        wPts = triangulate([X1(:,i) Y1(:,i)], [X2(:,i) Y2(:,i)],stereoParams_1124);
        aPts = [wPts(:,1) wPts(:,2) wPts(:,3)];
        for k = 1:length(aPts)
            RPts(:,k) = Rx*aPts(k,:)';
        end
        WX(:,i) = RPts(1,:)';
        WY(:,i) = RPts(2,:)';
        WZ(:,i) = RPts(3,:)';
    end
    
  %%  
    figure;
    for i = 1:length(X1)
        clf
        hold on
        plot3(WX(:,i),WY(:,i),WZ(:,i),'bo');
        plot3(WX(13:16,i),WY(13:16,i),WZ(13:16,i),'r-');
        plot3(WX([10:12 19],i),WY([10:12 19],i),WZ([10:12 19],i),'g-');
        plot3(WX(6:8,i),WY(6:8,i),WZ(6:8,i),'r-');
        plot3(WX(3:5,i),WY(3:5,i),WZ(3:5,i),'g-');
        
        plot3(WX([1 2 9],i),WY([1 2 9],i),WZ([1 2 9],i),'b-');
        plot3(WX([3 2 6],i),WY([3 2 6],i),WZ([3 2 6],i),'b-');
        plot3(WX([9 10 13],i),WY([9 10 13],i),WZ([9 10 13],i),'b-');
        plot3(WX([21 12 19 20],i),WY([21 12 19 20],i),WZ([21 12 19 20],i),'b-');
        plot3(WX([18 15 16 17],i),WY([18 15 16 17],i),WZ([18 15 16 17],i), 'b-');

%         v1 = [-399.9818 1729.5521 487.66082];
%         v2 = [-279.5029 1494.4893 265.35904];
%         v3 = [259.6 1896.43 357.17709];
%         v4 = [99.98375, 2188.0678 619.93435];
%         
%         v=[v1;v2;v3;v4;v1];
%         plot3(v(:,1),v(:,2),v(:,3),'r')

        grid on
        ylabel('y');
        xlabel('x');
%        xlim([-200 100]);
%        ylim([0 400]);
%        zlim([-200 100]);

        view([120 30]);
        anim(i) = getframe;
        pause(0.01);
        disp(i)
    end
    
    disp("ended")