
%% define variables
clc
close all
clear all
square_size = 250;
rand_factor = 0.9;
number_of_frames = 1000;
seed = 10;
rng(seed)
x = 0:square_size-1;
y = 0:square_size-1;
radius = 16;
x_shift = round(square_size/2);
y_shift = round(square_size/2);


%% generate a circle

[xx, yy] = meshgrid(x,y);
u = zeros(size(xx));
u(((xx-x_shift).^2+(yy-y_shift).^2)<=radius^2)=1;   % radius 10, center at the origin
len = sum(u(:)==1);
% imshow(u);
%% generate random numbers with amplitude = 1
comp_vec =  exp(i*(2*pi*rand(len,1)-pi));

%% Simulation of time integrated dynamic speckle image
for ff = 1:number_of_frames
    k= 1;
    for ii=1:length(x)
      for jj = 1:length(y)
        if(abs(u(ii,jj))>0)
          u(ii,jj) = comp_vec(k);
          k = k+1;
        end
      end
    end
    u_fft=  fft2(u);
    UU_abs(:,:,ff) = abs(u_fft);
    UU_speckle(:,:,ff) = u_fft.*conj(u_fft);
    rho = abs(comp_vec);
    theta = angle(comp_vec) + (2*pi*rand(len,1)-pi)*rand_factor;
    comp_vec = rho.*exp(i*theta);
    %ff;
end
 % implay(abs(UU_speckle))

%% gif
% v = VideoWriter('test.avi');
% open(v);
% %v.fps = 20
% %v.Quality = 100
% 
% for i = 1:number_of_frames
%    
%     figure(1)
%     fig = imshow(UU_speckle(:,:,i),[]);
%     colormap('gray')
%     frame(i) = getframe(gcf);
%     writeVideo(v,frame(i));
%     
% end


%% correlation coeffitient 
corr_co = [];
first_frame = UU_speckle(:,:,1);
for ff = 1:number_of_frames
    
    corr_co(ff) = corr2(first_frame,UU_speckle(:,:,ff));
    
end
histogram(corr_co, 100)
disp ('Done synthesis of dynamic speckle');
% figure ('name', 'L/D = 13.9')
% imshow (abs(UU_speckle(:,:,107)))
%% FT
% u_fft=  fft2(u);
% speckle1 = u_fft.*conj(u_fft);
% % hard boundary
% figure;imshow(abs(u_fft),[])

%% CDF calculation
% 
% % Compute the histogram of A and B.
% [countsA, binsA] = hist(Z);
% 
% % Compute the cumulative distribution function of A and B.
% cdfA = cumsum(countsA) / sum(countsA);

%% blurring

uu_blurring_25 = zeros(square_size, square_size);
uu_blurring_5 = zeros(square_size, square_size);
uu_blurring_1 = zeros(square_size, square_size);

for k = 1 : 25
    uu_blurring_25(:,:) = uu_blurring_25(:,:) + UU_speckle(:,:,k);
    if k == 25
        uu_blurring_25(:,:) = uu_blurring_25(:,:) ./ k;
    end
end
for k = 1 : 5
    uu_blurring_5(:,:) = uu_blurring_5(:,:) + UU_speckle(:,:,k);
    if k == 5
        uu_blurring_5(:,:) = uu_blurring_5(:,:) ./ k;
    end
end
% for k = 1 : 1
%     uu_blurring_1(:,:) = uu_blurring_1(:,:) + UU_speckle(:,:,k);
%     if k == 1
%         uu_blurring_1(:,:) = uu_blurring_1(:,:) ./ k;
%     end
% end
uu_blurring_1(:,:) = UU_speckle(:,:,1);

figure('name', 'blurring of all of them')
subplot(1,3,1)
imshow(abs(uu_blurring_1),[0 square_size*radius]);
subplot(1,3,2)
imshow(abs(uu_blurring_5),[0 square_size*radius]);
subplot(1,3,3)
imshow(abs(uu_blurring_25),[0 square_size*radius]);
colormap ('gray')
figure('name', 'contrast of all of them')

% LASCA Contrast
kernel = ones(15,15);
Nk=sum(kernel(:));
mu_img=filter2(kernel, uu_blurring_1, 'valid')/Nk;
img_sq=filter2(kernel, uu_blurring_1.^2, 'valid');
sig_img=sqrt((img_sq-Nk*mu_img.^2)/(Nk-1))
C=0.6*sig_img./mu_img;
mu_img5=filter2(kernel, uu_blurring_5, 'valid')/Nk;
img_sq5=filter2(kernel, uu_blurring_5.^2, 'valid');
sig_img5=sqrt((img_sq5-Nk*mu_img5.^2)/(Nk-1))
C5=0.6*sig_img5./mu_img5;
mu_img25=filter2(kernel, uu_blurring_25, 'valid')/Nk;
img_sq25=filter2(kernel, uu_blurring_25.^2, 'valid');
sig_img25=sqrt((img_sq25-Nk*mu_img25.^2)/(Nk-1))
C25=0.6*sig_img25./mu_img25;
subplot(1,3,1)
imshow(abs(C));
subplot(1,3,2)
imshow(abs(C5));
subplot(1,3,3)
imshow(abs(C25));

colormap jet

%%
i=1;
while corr_co(i)>=0.1
    decorrelation_time = i;
    i=i+1;
end   
%% Contrast to varying decorelation/exposure ratio vs Gaussian velocity distribution

uu_blurring_general = zeros(square_size, square_size);
tic
for i = 1:number_of_frames
    for k = 1:i
        uu_blurring_general(:,:) = uu_blurring_general(:,:) + UU_speckle(:,:,k);
        mu_img = filter2(kernel, uu_blurring_general, 'valid')/Nk;
        img_sq = filter2(kernel, uu_blurring_general.^2, 'valid');
        sig_img = sqrt((img_sq-Nk*mu_img.^2)/(Nk-1));
        C = sig_img./mu_img;
    end 
    Contrast(i) = mean(mean(C));
end
toc
figure()
C_for_Gaus_vel_dist=semilogx(Contrast);
logx=log(x);
hold on
Alt_Gaus_vel_dist = zeros(number_of_frames);
for i = 1:number_of_frames
    Gaus_vel_dist(i) = sqrt(0.5*sqrt(pi)*(decorrelation_time/i)*erf(i/decorrelation_time));
    Alt_Gaus_vel_dist(i) = sqrt(0.5*(decorrelation_time/i)*erf((sqrt(pi)*i)/decorrelation_time));
end
plot(Gaus_vel_dist)
hold on
plot(Alt_Gaus_vel_dist)
legend('C for Gaussian velocity distribution','Gaussian velocity distribution','Alternative Gaussian velocity distribution')
hold off

%% LSI Contrast
% cube is the 3-D spatio-temporal speckle image cube to be filtered

Ns = 1;% spatial dimension of region of % interest
Nt = 15;% temporal dimension of region of % interest
kernel = ones(Ns,Ns,Nt);
Nk = sum(kernel(:));
mu_cube = imfilter(UU_speckle,kernel)/Nk;
cube_sq = imfilter(UU_speckle.^2,kernel);
sig_cube=sqrt((cube_sq-Nk*mu_cube.^2)/(Nk-1));
C_LSI = sig_cube./mu_cube;
implay(abs(C_LSI));
figure
imshow(abs(C_LSI(:,:,20)),[]);colormap('jet');

colormap jet

%% Simulation of time integrated dynamic speckle image
sum_blur = 5; %number of frames in one blurred image
UU_blurring_5 = zeros(square_size, square_size,number_of_frames/sum_blur);
for iii = 1 : number_of_frames/sum_blur
    kkk = 1;
    for k = 1 : 5
        UU_blurring_5(:,:,iii) = UU_blurring_5(:,:,iii) + UU_speckle(:,:,k+kkk+iii);
    end
    kkk = kkk + sum_blur;
end
video = implay(abs(UU_blurring_5), 5);
video;

%%
% uu_blurring_mean = mean(UU_speckle(:,:,:))
% figure('name', 'func mean blurring of 200 ms temporal speckle')
% imshow(abs(uu_blurring_mean),[]);
% imshow(abs(UU_speckle(:,:,1)),[])
% uu_blurring = zeros(square_size, square_size);
% for k = 1 : number_of_frames_calc
%     uu_blurring(:,:) = uu_blurring(:,:) + UU_speckle(:,:,k);
% end
% %uu_blur_med = uu_blur ./ number_of_frames_calc;
% figure('name', 'blurring of few ms (!) temporal speckle')
% imshow(abs(uu_blurring),[]);

%% Histogram of speckles
figure ('name', 'histogram of temporal speckle')
hist = histogram(mean(UU_speckle(:,:,1:200),3));
figure('name','histogram of spatial speckle')
hist2 = histogram((UU_speckle(:,:,1)));

%% Median Filter
% J1 = medfilt2(UU_blurring_5(:,:,1), [7 7])
% J2 = ordfilt2(UU_blurring_5(:,:,1), )
% imshow(abs(J1),[])
 
%% 3D 
%function [] = video3dfigure(UU_speckle,number_of_frames,'jet')
% AntiShock Technologies LTD Copyrights
% Date  : 08/01/2020 , Author : Yokhai Dan
%[n m t] = size(video);
for i=1:number_of_frames
    
    surface([-0.1 0.1; -0.1 0.1], [i i; i i], [-1 -1; 1 1], ...
    'FaceColor', 'texturemap', 'CData', UU_speckle(:,:,i) );
   i
end
view(3)
colormap('jet')
