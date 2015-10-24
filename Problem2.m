close all;
clear all;
clc;

sdirectory1 = 'landmark_87\training';
datfiles1 = dir([sdirectory1 '\*.dat']);
sdirectory2 = 'landmark_87\test';
datfiles2 = dir([sdirectory2 '\*.dat']);

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
    warpings(:,k) = vertcat(file_x,file_y);
end

%compute the mean warpmeing
[r,c] = size(warpings);
for i=1:r
    mean(i,1) = sum(warpings(i,:)/c);
end

%display the mean_warping
plot(255-mean(1:87,1),255-mean(88:174,1),'.','Color','b');
pause;
close;

%diff_matrix for the warpings
for i = 1 : c
    diff_matrix(:,i) = warpings(:,i) - mean(:,1);
end

%scatter_matrix 
scatter_matrix = diff_matrix * diff_matrix';

%eigenvectors and eigenvalues
[U,L,V] = svd(scatter_matrix);
eigen_vectors = U;

for i = 1 : 174
    eigen_vectors(:,i) = eigen_vectors(:,i)/norm(eigen_vectors(:,i));
end

%didplay the first five eigrn warpings
for i = 1 : 5
    eigen_warping = eigen_vectors(:,i) + mean(:,1);
    subplot(3,3,i);
    plot(255-eigen_warping(1:87,1),255-eigen_warping(88:174,1),'.','Color','g');
end
pause;
close;

%read the test warpings
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
    test_warpings(:,k) = vertcat(file_x,file_y);
    
end

%diff_matrix for test warpings
for i = 1: 27
    diff_test(:,i) = test_warpings(:,i) - mean(:,1);
end

% matrix of principal components
a = eigen_vectors' * diff_test;

%reconstruction error
for k = 1 : 150 %number of eigen vectors used for reconstruction
    for i = 1 : 27 %test images
       
        reconstructed_warping(:,i) = mean(:,1);
        for j = 1 : k %for each image i, k eigen_vectors are used for reconstruction
            reconstructed_warping(:,i) = reconstructed_warping(:,i) + a(j,i)*eigen_vectors(:,j);
        end
        A = ((reshape(reconstructed_warping(:,i),87,2)) - (reshape(test_warpings(:,i),87,2))).^2;
        B = sum(A,2);
        error_image(i,k) = ((sum(sqrt(B))))/87;
    end
    total_error(k) = sum(error_image(:,k))/27;
end

plot(total_error);

    



