clear all;
close all;

% 1. fisher face (appearance) (from Problem 5)

%Compute C = F U M

%read the female and male faces. Store two images(1 male and 1 female) in the same column
sdirectoryff = 'female_face\training';
bmpfilesff = dir([sdirectoryff '\*.bmp']);

sdirectorymf = 'male_face\training';
bmpfilesmf = dir([sdirectorymf '\*.bmp']);

C = zeros(65536,75);
mean_female = zeros(65536,1);
imageVectorff = zeros(65536,1);
imageVectormf = zeros(65536,1);
for k = 1:length(bmpfilesff)
    
    %read each image
    filenameff = [sdirectoryff '\' bmpfilesff(k).name];
    tempImageff = imread(filenameff);
  
    %for this image form a k^2 dimensional vector
    l = 1;
    for i = 1 : 256
          for j = 1 : 256
             imageVectorff(l,1) = tempImageff(i,j);
             l=l+1;
          end
    end
    mean_female(:,1) = mean_female(:,1) + imageVectorff(:,1);
    C(:,k) = imageVectorff(:,1);
end
mean_female = mean_female/length(bmpfilesff);

mean_male = zeros(65536,1);
for k = 1:length(bmpfilesmf)
    
    %read each image
    filenamemf = [sdirectorymf '\' bmpfilesmf(k).name];
    tempImagemf = imread(filenamemf);
  
    %for this image form a k^2 dimensional vector
    l = 1;
    for i = 1 : 256
          for j = 1 : 256
             imageVectormf(l,1) = tempImagemf(i,j);
             l=l+1;
          end
    end
    mean_male(:,1) = mean_male(:,1) + imageVectormf(:,1);
    C(:,k + length(bmpfilesff)) = imageVectormf(:,1);
end
mean_male = mean_male/length(bmpfilesmf);

%compute B = C'C
B = C' * C;

%find eigenvalues and eigenvectors of B
[V,L,Q] = svd(B);
%{
for i = 1 : 153
    V(:,i) = V(:,i)/norm(V(:,i));
end
%}

%find A
A = zeros(65536,153);
for i = 1 : 153
    A(:,i) = sqrt(L(i,i)) * C * V(:,i)/norm(C * V(:,i));
end

%find y = A'(Mf - Mm)
y = A' * (mean_female - mean_male);

%solve for z in the equation (L^2 * V')*z = y
L2V = L^2 * V';
z = inv(L2V) * y;

%compute w = Cz
w1 = C * z;

m = int32(256);
normalized_image = uint8(255*mat2gray(w1));
  
            for q = 1 : 65536
                col_pixel = mod(q,256) + 1;
                row_pixel = idivide(q,m,'ceil');
                mat_image(row_pixel,col_pixel) = normalized_image(q);
            end
            subplot(1,3,1);
            imshow(mat_image);
            pause;
            close;

%w0 = (w' *(mean_female) + w'*(mean_male)) / 2;

%______________________________________________________________________________________
% 2. fisher face (geometric) 
            
 %Compute C = F U M

%read the female and male faces. Store two images(1 male and 1 female) in the same column
sdirectoryff = 'female_landmark_87\training';
textfilesff = dir([sdirectoryff '\*.txt']);
mean_female = zeros(174,1);
C = zeros(174,153);
for k = 1:length(textfilesff)
    %read each female
    file_content = importdata([sdirectoryff '\' textfilesff(k).name]);
    C(:,k) = reshape(file_content,174,1);
    mean_female(:,1) = mean_female(:,1) + C(:,k);
end
mean_female = mean_female/length(textfilesff);
    
sdirectorymf = 'male_landmark_87\training';
textfilesmf = dir([sdirectorymf '\*.txt']);
mean_male =  zeros(174,1);
for k = 1:length(textfilesmf)
    %read each male
    file_content = importdata([sdirectorymf '\' textfilesmf(k).name]);
    C(:,k+length(textfilesff)) = reshape(file_content,174,1);
    mean_male(:,1) = mean_male(:,1) + C(:,k+length(textfilesff));
end
mean_male = mean_male/length(textfilesmf);

%compute B = C'C
B = C' * C;

%find eigenvalues and eigenvectors of B
[V,L,Q] = svd(B);
%{
for i = 1 : 153
    V(:,i) = V(:,i)/norm(V(:,i));
end
%}

%find A
A = zeros(174,153);
for i = 1 : 153
    A(:,i) = sqrt(L(i,i)) * C * V(:,i)/norm(C * V(:,i));
end

%find y = A'(Mf - Mm)
y = A' * (mean_female - mean_male);

%solve for z in the equation (L^2 * V')*z = y
L2V = L^2 * V';
z = inv(L2V) * y;

%compute w = Cz
w2 = C * z;

%w20 = (w' *(mean_female) + w'*(mean_male)) / 2;

%_________________________________________________________________________________

% read the test faces
% lets first read the female faces

sdirectoryff = 'female_face\test';
bmpfilesff = dir([sdirectoryff '\*.bmp']);

for k = 1:length(bmpfilesff)
    
    %read each image
    filenameff = [sdirectoryff '\' bmpfilesff(k).name];
    tempImageff = imread(filenameff);

    %for this image form a k^2 dimensional vector
    l = 1;
    for i = 1 : 256
          for j = 1 : 256
             imageVectorff(l,1) = tempImageff(i,j);
             l=l+1;
          end
    end

   
    test_image_female(:,k) = imageVectorff(:,1);
end

for i = 1 : 10
    
    f_val_female(i) = w1' * test_image_female(:,i);
    
end

sdirectoryff = 'female_landmark_87\test';
textfilesff = dir([sdirectoryff '\*.txt']);
mean_female = zeros(174,1);
for k = 1:length(textfilesff)
    %read each female
    file_content = importdata([sdirectoryff '\' textfilesff(k).name]);
    test_landmarks_female(:,k) = reshape(file_content,174,1);
end

for i = 1 : 10
    
    f_val_female_y(i) = w2' * test_landmarks_female(:,i);
    
end

sdirectorymf = 'male_face\test';
bmpfilesmf = dir([sdirectorymf '\*.bmp']);

for k = 1:length(bmpfilesmf)
    
    %read each image
    filenamemf = [sdirectorymf '\' bmpfilesmf(k).name];
    tempImagemf = imread(filenamemf);

    %for this image form a k^2 dimensional vector
    l = 1;
    for i = 1 : 256
          for j = 1 : 256
             imageVectormf(l,1) = tempImagemf(i,j);
             l=l+1;
          end
    end

   
    test_image_male(:,k) = imageVectormf(:,1);
end

for i = 1 : 10
    
    f_val_male(i) = w1' * test_image_male(:,i);
    
end

sdirectorymf = 'male_landmark_87\test';
textfilesmf = dir([sdirectorymf '\*.txt']);
mean_male = zeros(174,1);
for k = 1:length(textfilesmf)
    %read each female
    file_content = importdata([sdirectorymf '\' textfilesmf(k).name]);
    test_landmarks_male(:,k) = reshape(file_content,174,1);
end

for i = 1 : 10
    
    f_val_male_y(i) = w2' * test_landmarks_male(:,i);
    
end

hold on;
scatter(f_val_female,f_val_female_y,'.','g');
scatter(f_val_male,f_val_male_y,'b');
hold off;
