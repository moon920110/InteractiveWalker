load('../calibration_data/2022-12-16/stereoParams_1216.mat');
main_dir = "../data/";
files = dir(main_dir);
dirs = files(arrayfun(@(x) x.isdir, files));

for data =3:size(dirs,1)
    try
    %%
        fprintf("%c", dirs(data).name);
        sub_dir = main_dir + dirs(data).name;
        frontSub = readtable(sub_dir + "/keypoints/keypoints2D/frontSub.csv");
        sideSub = readtable(sub_dir + "/keypoints/keypoints2D/sideSub.csv");
        folderName = "/keypoints/keypoints3D/";
        if not(exist(folderName, "dir"))
            mkdir(fullfile(sub_dir, folderName));
        end
    %%
        for i =1:size(frontSub,1)
            X1(:,i)=table2array(frontSub(i,1:3:end))';
            Y1(:,i)=table2array(frontSub(i,2:3:end))';
        end
    %%
        for i =1:size(sideSub,1)
            X2(:,i)=table2array(sideSub(i,1:3:end))';
            Y2(:,i)=table2array(sideSub(i,2:3:end))';
        end
    %%  
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
            wPts = triangulate([X1(:,i) Y1(:,i)], [X2(:,i) Y2(:,i)],stereoParams_1216);
            aPts = [wPts(:,1) wPts(:,2) wPts(:,3)];
            for k = 1:length(aPts)
                RPts(:,k) = Rx*aPts(k,:)';
            end
            WX(:,i) = RPts(1,:)';
            WY(:,i) = RPts(2,:)';
            WZ(:,i) = RPts(3,:)';
        end
    %%
        writematrix(WX, sub_dir + "/keypoints/keypoints3D/WX.csv");
        writematrix(WY, sub_dir + "/keypoints/keypoints3D/WY.csv");
        writematrix(WZ, sub_dir + "/keypoints/keypoints3D/WZ.csv");
        
        fprintf("\n");
        vars = {'WX', 'WY', 'WZ', 'X1', 'X2', 'Y1', 'Y2'};
        clear(vars{:})
     catch
         fprintf("Declined");
         fprintf("\n")
         vars = {'WX', 'WY', 'WZ', 'X1', 'X2', 'Y1', 'Y2'};
         clear(vars{:})
     end
end