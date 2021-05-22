%% define variables

close all
clear all
x = 0:255;
y = 0:255;
radius = 32;
x_shift = 95;
y_shift = 50;

%% generate a circle

[xx yy] = meshgrid(x,y);
u = zeros(size(xx));

u(((xx-x_shift).^2+(yy-y_shift).^2)<=radius^2)=1;   % radius 100, center at the origin
u1 = u;
len = sum(u(:)==1);

%% generate random numbers with amplitude = 1
comp_vec =  exp(i*(2*pi*rand(len,1)-pi));

%% replace circle values with random values
k= 1;
for i=1:length(x)
  for j = 1:length(y)
    if(abs(u(i,j))>0)
      u(i,j)
      u(i,j) = comp_vec(k);
      %comp_vec(k)
      u(i,j)
      k = k+1;
    end
  end
end

%% FT
u_fft=  fft2(u);
speckle1 = u_fft.*conj(u_fft);
% hard boundary
figure(2)
imshow(abs(u_fft),[])
%imshow(abs(speckle1),[])
