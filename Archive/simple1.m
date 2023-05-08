%% run 'placeinmat.m' with matlab data in 'simple1.mat'
%outputfile='out.mat';
load('simple1.mat');
[mat,vec,metind,expind]=placeinmat(tracer,mea,exp,alltracers,emumets(2:end),lmid,-1);  
writematrix(mat);
writematrix(vec);
writematrix(metind);
writematrix(expind);
%save(outputfile, 'mat', 'vec', 'metind', 'expind');
%load(outputfile);
%csvwrite('out.csv', data);
