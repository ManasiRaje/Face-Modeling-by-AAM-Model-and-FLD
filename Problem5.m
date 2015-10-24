clear all;
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
w = C * z;

m = int32(256);
normalized_image = uint8(255*mat2gray(w));
  
            for q = 1 : 65536
                col_pixel = mod(q,256) + 1;
                row_pixel = idivide(q,m,'ceil');
                mat_image(row_pixel,col_pixel) = normalized_image(q);
            end
            subplot(1,3,1);
            imshow(mat_image);
            pause;
            close;
           
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

w0 = (w' *(mean_female) + w'*(mean_male)) / 2;

for i = 1 : 10
    
    f_val_female(i) = w' * test_image_female(:,i) - w0;
    
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
    
    f_val_male(i) = w' * test_image_male(:,i) - w0;
       
end

hold on;
scatter(f_val_male,zeros(10,1),'b');
scatter(f_val_female,zeros(10,1),'.','g');
hold off;


            
            