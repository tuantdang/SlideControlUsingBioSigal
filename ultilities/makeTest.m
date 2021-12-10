clear;
e1 = csvread("clean_data\eyeblink1.csv");
e2 = csvread("clean_data\eyeblink2.csv");
e3 = csvread("clean_data\eyeblink3.csv");
m = csvread("clean_data\Grindteeth.csv");
n = csvread("clean_data\normal.csv");

fs = 250;
m13 = m(1:3*fs);
n13 = n(1:3*fs);
out = [e1; e2; e3;  m13; n13];
csvwrite("test.csv", out);