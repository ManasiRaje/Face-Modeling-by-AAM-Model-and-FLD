close all;
clear all;
clc;
m = int32(256);
samplesMatrix = zeros(65536,150);

% read the images
imageVector = zeros(65536,1);
sdirectory = 'face\training';
bmpfiles = dir([sdirectory '\*.bmp']);

for k = 1:length(bmpfiles)
    
    %read each image
    filename = [sdirectory '\' bmpfiles(k).name];
    tempImage = imread(filename);
  
    %for this image form a k^2 dimensional vector
    l = 1;
    for i = 1 : 256
          for j = 1 : 256
             imageVector(l,1) = tempImage(i,j);
             l=l+1;
          end
    end
    
    % add the vector to your matrix of all sample vectors
    samplesMatrix(:,k) = samplesMatrix(:,k) + imageVector(:,1);
end

%find the mean image
[r,c] = size(samplesMatrix);
for i=1:r
    mean(i,1) = sum(samplesMatrix(i,:)/c);
end

%display mean face
normalizedImage = uint8(255*mat2gray(mean));

for i = 1 : 65536
        col_pixel = mod(i,256) + 1;
        row_pixel = idivide(i,m,'ceil');
        mat_image(row_pixel,col_pixel) = normalizedImage(i,1);
end
imshow(mat_image);
pause;
close;

%scatter matrix 
%(we will calculate T'T instead of TT' to avoid going out of memory)
for i = 1 : 150
        diff_matrix(:,i) = samplesMatrix(:,i)-mean(:,1);
end
scatter_matrix = diff_matrix' * diff_matrix;

[U,L,V] = svd(scatter_matrix);
e = diff_matrix * U;

for i = 1 : 150
    e(:,i) = e(:,i)/norm(e(:,i));
end

%show the first 20 eigen faces
for i = 1 : 20
    normalizedImage = uint8(255*mat2gray(e(:,i)));
    for j = 1 : 65536
        col_pixel = mod(j,256) + 1;
        row_pixel = idivide(j,m,'ceil');
        mat_image(row_pixel,col_pixel) = normalizedImage(j,1);
    end
    subplot(5,4,i);
    imshow(mat_image);
end
pause;
close;

%read test images
sdirectory = 'face\test';
bmpfiles = dir([sdirectory '\*.bmp']);
testSamplesMatrix = zeros(65536,27);
for k = 1:length(bmpfiles)
    
    %read each test image
    filename = [sdirectory '\' bmpfiles(k).name];
    tempImage = imread(filename);
    
    %for this test image form a k^2 dimensional vector
    l = 1;
    for i = 1 : 256
          for j = 1 : 256
             imageVector(l,1) = tempImage(i,j);
             l=l+1;
          end
    end
    
    % add the vector to your matrix of all test sample vectors
    testSamplesMatrix(:,k) = testSamplesMatrix(:,k) + imageVector(:,1);   
end

%diff_matrix for test images
for i = 1 : 27
    diff_test(:,i) = testSamplesMatrix(:,i) - mean(:,1);
end

% matrix of principal components
a = e' * diff_test;

%reconstructed test images
for i = 1 : 27 %test images
    reconstructed_image = (mean);
    for j = 1 : 20 %for each image i, k=20 eigen_vectors are used for reconstruction
        reconstructed_image = reconstructed_image + ( a(j,i)*(e(:,j)));
    end
    reconstructed_image = uint8(255*mat2gray(reconstructed_image));
    for q = 1 : 65536
        col_pixel = mod(q,256) + 1;
        row_pixel = idivide(q,m,'ceil');
        mat_image(row_pixel,col_pixel) = reconstructed_image(q);
    end
    subplot(5,6,i);
    imshow(mat_image);
           
end

pause;
close;

reconstructed_image = zeros(65536,27);
%reconstruction error
for k = 1 : 150 %number of eigen vectors used for reconstruction
    error_image = zeros(27,k);
    for i = 1 : 27 %test images
        reconstructed_image(:,i) = double(mean(:,1));
        for j = 1 : k %for each image i, k eigen_vectors are used for reconstruction
            reconstructed_image(:,i) = (reconstructed_image(:,i) + a(j,i)*e(:,j));
        end
        %reconstructed_image(:,i) = uint8(255*mat2gray(reconstructed_image(:,i)));
        A = ((reconstructed_image(:,i)) - (testSamplesMatrix(:,i))).^2;
        error_image(i,k) = (sum((double(A))))/65536;
    end
    total_error(k) = (sum(error_image(:,k)))/27;
end

plot(total_error(1:end));
