close all;
clear all;
clc;
m = int32(256);

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
[U,L1,V] = svd(scatter_matrix);
eigen_vectors_landmarks = U;
for  i = 1 : 174
    eigen_vectors_landmarks(:,i) = (eigen_vectors_landmarks(:,i) + mean_landmarks(:,1))/norm(eigen_vectors_landmarks(:,i));
end
L1 = diag(L1);

%reconstruction of landmarks
for k = 1 : 20   %new synthesized images
    reconstructed_landmarks(:,k) = mean_landmarks(:,1);
    for i = 1 : 10   %number of eigen vectors used for reconstruction
        reconstructed_landmarks(:,k) = reconstructed_landmarks(:,k) + (normrnd(0,(sqrt(L1(k))),[1,1]))*eigen_vectors_landmarks(:,i);
    end
    reconstructed_landmarks(:,k) = uint8(255*mat2gray(reconstructed_landmarks(:,k)));
    %subplot(4,5,k);
    %scatter(reconstructed_landmarks(1:87,k),reconstructed_landmarks(88:174,k),'.');
    
    
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

% 3.3 scatter matrix
%(we will calculate T'T instead of TT' to avoid going out of memory)
for i = 1 : 150
        diff_matrix_warped(:,i) = warped_images(:,i)-mean_warped_image(:,1);
end

scatter_matrix = diff_matrix_warped' * diff_matrix_warped;
[U,L2,V] = svd(scatter_matrix);
eigen_vectors_warped = diff_matrix_warped * U;

for i = 1 : 150
    eigen_vectors_warped(:,i) = eigen_vectors_warped(:,i)/norm(eigen_vectors_warped(:,i));
end
L2 = diag(L2);

%reconstruction of warped test images 
for k = 1 : 20   %test images
    reconstructed_warped_image(:,k) = mean_warped_image(:,1);
    for i = 1 : 10   %number of eigen vectors used for reconstruction
        reconstructed_warped_image(:,k) = reconstructed_warped_image(:,k) + ((normrnd(0,sqrt(L2(k)),[1,1]))*(eigen_vectors_warped(:,i)));
    end
    reconstructed_warped_image(:,k) = uint8(255*mat2gray(reconstructed_warped_image(:,k)));
    %{
    reconstructed_image = uint8(255*mat2gray(reconstructed_warped_image(:,k) ));
           
            for q = 1 : 65536
                col_pixel = mod(q,256) + 1;
                row_pixel = idivide(q,m,'ceil');
                mat_image(row_pixel,col_pixel) = reconstructed_image(q);
            end
            imshow(mat_image);
            pause;
    %}
end

for i = 1 : 20
    [final_image] = warpImage_kent(reshape(reconstructed_warped_image(:,i),256,256),mean_landmarks_reshaped,reshape(reconstructed_landmarks(:,i),87,2));
    subplot(7,7,i);
    imshow(uint8(255*mat2gray(final_image')));
end


