close all;
clear all;
clc;
m = int32(256);
%align the training images using their landmarks into the mean position
% i.e. align the training images from their current landmarks into the
% mean landmark

% 1. read the training landmarks and compute the mean
sdirectory1 = 'landmark_87\training';
datfiles1 = dir([sdirectory1 '\*.dat']);
for k = 1:length(datfiles1)
    
    %read each image
    file_content = importdata([sdirectory1 '\' datfiles1(k).name]);
    file_content = file_content(2:end);
    j=1;
    for i = 1 :2: 174
        file_x(j,1) = file_content(i);
        file_y(j,1) = file_content(i+1);
        j=j+1;
    end
    landmarks(:,k) = vertcat(file_x,file_y);
end

%compute the mean landmark
[r,c] = size(landmarks);
for i=1:r
    mean_landmarks(i,1) = sum(landmarks(i,:)/c);
end
mean_landmarks_reshaped = reshape(mean_landmarks,87,2);

%diff_matrix for the landmarks
for i = 1 : c
    diff_matrix_landmarks(:,i) = landmarks(:,i) - mean_landmarks(:,1);
end

%scatter_matrix 
scatter_matrix = diff_matrix_landmarks * diff_matrix_landmarks';

%eigenvectors and eigenvalues
[U,L,V] = svd(scatter_matrix);
eigen_vectors_landmarks = U;
for  i = 1 : 174
    eigen_vectors_landmarks(:,i) = (eigen_vectors_landmarks(:,i)+mean_landmarks(:,1))/norm(eigen_vectors_landmarks(:,i));
end

% 2. read each training image and warp it
sdirectory = 'face\training';
bmpfiles = dir([sdirectory '\*.bmp']);
warped_images = zeros(65536,150);
imageVector = zeros(65536,1);
for k = 1 : 150
    
    filename = [sdirectory '\' bmpfiles(k).name];
    original_image = imread(filename);
    
    landmarks_reshaped = reshape(landmarks(:,k),87,2);
    [warped_image] = warpImage_kent(original_image,landmarks_reshaped,mean_landmarks_reshaped);
    %{
    subplot(10,15,k);
    imshow(warped_image);
    %}
    
    %for this image form a k^2 dimensional vector
    l = 1;
    for i = 1 : 256
          for j = 1 : 256
             imageVector(l,1) = warped_image(i,j);
             l=l+1;
          end
    end
    
    % add the vector to your matrix of all sample vectors
    warped_images(:,k) = warped_images(:,k) + imageVector(:,1);
    
end

% 3. find eigen faces with these aligned images

% 3.1 find the mean-warped image
[r,c] = size(warped_images);
for i=1:r
    mean_warped_image(i,1) = sum(warped_images(i,:)/c);
end


% 3.2 display the mean-warped face
normalizedImage = uint8(255*mat2gray(mean_warped_image));
m = int32(256);
for i = 1 : 65536
        col_pixel = mod(i,256) + 1;
        row_pixel = idivide(i,m,'ceil');
        mat_image(row_pixel,col_pixel) = normalizedImage(i,1);
end
imshow(mat_image);
pause;
close;


% 3.3 scatter matrix
%(we will calculate T'T instead of TT' to avoid going out of memory)
for i = 1 : 150
        diff_matrix_warped(:,i) = warped_images(:,i)-mean_warped_image(:,1);
end

scatter_matrix = diff_matrix_warped' * diff_matrix_warped;
[U,L,V] = svd(scatter_matrix);
eigen_vectors_warped = diff_matrix_warped * V;
for i = 1 : 150
    eigen_vectors_warped(:,i) = eigen_vectors_warped(:,i)/norm(eigen_vectors_warped(:,i));
end


% 3.4 show the first 20 eigen faces
for i = 1 : 20
    normalizedImage = uint8(255*mat2gray(eigen_vectors_warped(:,i)));
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

% 4. test landmarks reconstruction

%(i) project test landmarks to the top 10 eigen-warpings, you get the reconstructed landmarks.

%read the test landmarks
sdirectory2 = 'landmark_87\test';
datfiles2 = dir([sdirectory2 '\*.dat']);
for k = 1:length(datfiles2)
    
    %read each image
    file_content = importdata([sdirectory2 '\' datfiles2(k).name]);
    file_content = file_content(2:end);
    j=1;
    for i = 1 :2: 174
        file_x(j,1) = file_content(i);
        file_y(j,1) = file_content(i+1);
        j=j+1;
    end
    test_landmarks(:,k) = vertcat(file_x,file_y);
end

%diff_matrix for test warpings
for i = 1: 27
    diff_test_landmarks(:,i) = test_landmarks(:,i) - mean_landmarks(:,1);
end

% matrix of principal components
a_test_landmarks = eigen_vectors_landmarks' * diff_test_landmarks;

%reconstruction
for k = 1 : 27   %test images
    reconstructed_landmarks(:,k) = mean_landmarks(:,1);
    for i = 1 : 10   %number of eigen vectors used for reconstruction
        reconstructed_landmarks(:,k) = reconstructed_landmarks(:,k) + a_test_landmarks(i,k)*eigen_vectors_landmarks(:,i);
    end
end

% 5. test images + test landmarks ----mean landmark----> warped_test_images
% (ii) warp the face image to the mean position and then project to the top k (say k=10) eigen-faces

%read test landmarks
%alreay read in 'test_landmarks'

%read eah test image and wwarp/align it using the 'mean_landmarks'
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

warped_test_images = zeros(65536,27);
for k = 1:length(bmpfiles)
    
    %read each test image
    filename = [sdirectory '\' bmpfiles(k).name];
    original_test_image = imread(filename);
    
    %warp this test image
    test_landmarks_reshaped = reshape(test_landmarks(:,k),87,2);
    [warped_test_image] = warpImage_kent(original_test_image,test_landmarks_reshaped,mean_landmarks_reshaped);
    
    %for this image form a k^2 dimensional vector
    l = 1;
    for i = 1 : 256
          for j = 1 : 256
             imageVector(l,1) = warped_test_image(i,j);
             l=l+1;
          end
    end
    
    % add the vector to your matrix of all sample vectors
    warped_test_images(:,k) = warped_test_images(:,k) + imageVector(:,1);
end

%diff matrix
for i = 1 : 27
    diff_test_warped(:,i) = warped_test_images(:,i) - mean_warped_image(:,1);
end
% reconstruct these warped_test_images using the warped-eigen-faces

a_warped_test = eigen_vectors_warped' * diff_test_warped;

%{
%reconstruction of warped test images 
for k = 1 : 27   %test images
    reconstructed_warped_image(:,k) = mean_warped_image(:,1);
    for i = 1 : 150   %number of eigen vectors used for reconstruction
        reconstructed_warped_image(:,k) = reconstructed_warped_image(:,k) + (a_warped_test(i,k)*(eigen_vectors_warped(:,i)));
    end
end

%normalize these warped-reconstructed-test images
reconstructed_warped_image = uint8(255*mat2gray(reconstructed_warped_image));

%display these 27 images
for j = 1 : 27
    for i = 1 : 65536
        col_pixel = mod(i,256) + 1;
        row_pixel = idivide(i,m,'ceil');
        mat_image(row_pixel,col_pixel) = reconstructed_warped_image(i,j);
    end
    subplot(6,5,j);
    imshow(mat_image); 
end

%reconstruction error
% let's do the unwarping first
for i = 1 : 27
     [unwarped_image] = warpImage_kent(reshape(reconstructed_warped_image(:,i),256,256),mean_landmarks_reshaped,reshape(reconstructed_landmarks(:,i),87,2));
     unwarped_test_images(:,i) = reshape(unwarped_image,65536,1);
end
%}

%reconstruction error

for k = 1 : 30 %number of eigen vectors used for reconstruction
    error_image = zeros(27,k);
    for i = 1 : 27 %test images
        reconstructed_warped_image(:,i) = mean_warped_image(:,1);
        for j = 1 : k %for each image i, k eigen_vectors are used for reconstruction
            reconstructed_warped_image(:,i) = reconstructed_warped_image(:,i) + (a_warped_test(j,i)*(eigen_vectors_warped(:,j)));
        end
        %reconstructed_image(:,i) = uint8(255*mat2gray(reconstructed_image(:,i)));
        [unwarped_image] = warpImage_kent(reshape(reconstructed_warped_image(:,i),256,256),mean_landmarks_reshaped,reshape(reconstructed_landmarks(:,i),87,2));
        unwarped_test_images(:,i) = reshape(unwarped_image,65536,1);
        A = (double(unwarped_test_images(:,i)) - (testSamplesMatrix(:,i))).^2;
        error_image(i,k) = (sum((double(A))))/65536;
    end
    total_error(k) = (sum(error_image(:,k)))/27;
end

plot(total_error(1:end));