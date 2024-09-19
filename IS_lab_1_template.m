clc;
% Classification using perceptron

% Reading apple images
A1=imread('apple_04.jpg');
A2=imread('apple_05.jpg');
A3=imread('apple_06.jpg');
A4=imread('apple_07.jpg');
A5=imread('apple_11.jpg');
A6=imread('apple_12.jpg');
A7=imread('apple_13.jpg');
A8=imread('apple_17.jpg');
A9=imread('apple_19.jpg');

% Reading pears images
P1=imread('pear_01.jpg');
P2=imread('pear_02.jpg');
P3=imread('pear_03.jpg');
P4=imread('pear_09.jpg');

% Calculate for each image, colour and roundness
% For Apples
% 1st apple image(A1)
hsv_value_A1=spalva_color(A1); %color
metric_A1=apvalumas_roundness(A1); %roundness
% 2nd apple image(A2)
hsv_value_A2=spalva_color(A2); %color
metric_A2=apvalumas_roundness(A2); %roundness
% 3rd apple image(A3)
hsv_value_A3=spalva_color(A3); %color
metric_A3=apvalumas_roundness(A3); %roundness
% 4th apple image(A4)
hsv_value_A4=spalva_color(A4); %color
metric_A4=apvalumas_roundness(A4); %roundness
% 5th apple image(A5)
hsv_value_A5=spalva_color(A5); %color
metric_A5=apvalumas_roundness(A5); %roundness
% 6th apple image(A6)
hsv_value_A6=spalva_color(A6); %color
metric_A6=apvalumas_roundness(A6); %roundness
% 7th apple image(A7)
hsv_value_A7=spalva_color(A7); %color
metric_A7=apvalumas_roundness(A7); %roundness
% 8th apple image(A8)
hsv_value_A8=spalva_color(A8); %color
metric_A8=apvalumas_roundness(A8); %roundness
% 9th apple image(A9)
hsv_value_A9=spalva_color(A9); %color
metric_A9=apvalumas_roundness(A9); %roundness

%For Pears
%1st pear image(P1)
hsv_value_P1=spalva_color(P1); %color
metric_P1=apvalumas_roundness(P1); %roundness
%2nd pear image(P2)
hsv_value_P2=spalva_color(P2); %color
metric_P2=apvalumas_roundness(P2); %roundness
%3rd pear image(P3)
hsv_value_P3=spalva_color(P3); %color
metric_P3=apvalumas_roundness(P3); %roundness
%2nd pear image(P4)
hsv_value_P4=spalva_color(P4); %color
metric_P4=apvalumas_roundness(P4); %roundness

%selecting features(color, roundness, 3 apples and 2 pears)
%A1,A2,A3,P1,P2
%building matrix 2x5
x1=[hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_P1 hsv_value_P2];
x2=[metric_A1 metric_A2 metric_A3 metric_P1 metric_P2];
% estimated features are stored in matrix P:
P=[x1;x2];

%Desired output vector
T=[1;1;1;-1;-1]; % <- ČIA ANKSČIAU BUVO KLAIDA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

%% train single perceptron with two inputs and one output

% generate random initial values of w1, w2 and b
w1 = randn(1);
w2 = randn(1);
b = randn(1);

% calculate weighted sum with randomly generated parameters
% calculate current output of the perceptron 
for i= 1:5
    v(i) = x1(i)*w1 + x2(i)*w2 + b;
    if v(i) > 0
        y = 1;
    else
        y = -1;
    end
    e_laikinas(i) = T(i) - y;
end

% calculate the total error for these 5 inputs 
e = abs(e_laikinas(1)) + abs(e_laikinas(2)) + abs(e_laikinas(3)) + abs(e_laikinas(4)) + abs(e_laikinas(5));

% write training algorithm
eta = 0.1;
while e ~= 0 % executes while the total error is not 0

for i= 1:5
    v(i) = x1(i)*w1 + x2(i)*w2 + b;
    if v(i) > 0
        y = 1;
    else
        y = -1;
    end
    e_laikinas(i) = T(i) - y;
    w1 = w1 + eta*e_laikinas(i)*x1(i);
    w2 = w2 + eta*e_laikinas(i)*x2(i);
    b = b + eta*e_laikinas(i);
end

e = abs(e_laikinas(1)) + abs(e_laikinas(2)) + abs(e_laikinas(3)) + abs(e_laikinas(4)) + abs(e_laikinas(5));
disp(['Total error: ', num2str(e)]);

end
%Final w1, w2, b
disp(['Final w1: ', num2str(w1)]);
disp(['Final w2: ', num2str(w2)]);
disp(['Final b: ', num2str(b)]);

%% Testing
x1=[hsv_value_A4 hsv_value_A5 hsv_value_A6 hsv_value_A7 hsv_value_A8 hsv_value_A9 hsv_value_P3 hsv_value_P4];
x2=[metric_A4 metric_A5 metric_A6 metric_A7 metric_A8 metric_A9 metric_P3 metric_P4];
T=[1;1;1;1;1;1;-1;-1];
for i= 1:8
    v(i) = x1(i)*w1 + x2(i)*w2 + b;
    if v(i) > 0
        y = 1;
    else
        y = -1;
    end

    if y==T(i) && y==1
        disp('Obuolis');
    elseif y==T(i) && y==-1
        disp('Kriause');
    else
        disp('Error');
    end
    e_laikinas(i) = T(i) - y;
end
e = abs(e_laikinas(1)) + abs(e_laikinas(2)) + abs(e_laikinas(3)) + abs(e_laikinas(4)) + abs(e_laikinas(5));
disp(['Total error: ', num2str(e)]);

%% Classification task using Naive Bayes Classifier
clc;

%Learning
x1=[hsv_value_A1 hsv_value_A2 hsv_value_A3 hsv_value_A4 hsv_value_P1 hsv_value_P2];
x2=[metric_A1 metric_A2 metric_A3 metric_A4 metric_P1 metric_P2];

P=[x1;x2]';

T=[1;1;1;1;-1;-1];

NBModel = fitcnb(P, T);

%Testing
x1_test=[hsv_value_A5 hsv_value_A6 hsv_value_A7 hsv_value_A8 hsv_value_A9 hsv_value_P3 hsv_value_P4];
x2_test=[metric_A5 metric_A6 metric_A7 metric_A8 metric_A9 metric_P3 metric_P4];

P_test=[x1_test;x2_test]';

T_test=[1;1;1;1;1;-1;-1];


predicted_labels = predict(NBModel, P_test);

for i = 1:length(predicted_labels)
    if predicted_labels(i) == 1
        disp('Obuolis');
    elseif predicted_labels(i) == -1
        disp('Kriause');
    else
        disp('Error');
    end
end

accuracy = sum(predicted_labels == T_test) / length(T_test) * 100;
disp(['Tikslumas: ', num2str(accuracy), '%']);
