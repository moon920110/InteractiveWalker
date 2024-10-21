
load('stereoParams_1109.mat')

%%
    Rx = rotx(-90);
    X1 =[211; 278; 583; 493];
    Y1 =[762; 887; 772; 688];
    X2 =[330; 238; 559; 602];
    Y2 =[714; 801; 899; 787];
    
    for i = 1:1
        wPts = triangulate([X1(:,i) Y1(:,i)], [X2(:,i) Y2(:,i)],stereoParams_1109);
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
        disp(WX(1,1))
        clf
        hold on
        plot3(WX([1 2 3 4 1],1),WY([1 2 3 4 1],1),WZ([1 2 3 4 1],1),'-r');
 
        grid on
        ylabel('y');
        xlabel('x');
%        xlim([-200 100]);
%        ylim([0 400]);
%        zlim([-200 100]);
        view([120 30]);
        disp(i)
    end
    