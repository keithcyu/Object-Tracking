% demo_VBMF.m
%
% (c) Masashi Sugiyama, Department of Compter Science, Tokyo Institute of Technology, Japan.
%     sugi@cs.titech.ac.jp,     http://sugiyama-www.cs.titech.ac.jp/~sugi/software/uLSIF/

clear all

rand('state',0); randn('state',0);

d=100;         % Entire dimensionality
n=200;         % Number of samples
r=50;          % True rank
Z=randn(r,n);  % True samples in dimension r
B=randn(d,r);  % Loading matrix
U=B*Z; U=U/std(U(:)); % Noiseless samples in dimension d
sigma=0.1;
V=U+sigma*randn(d,n); % Noisy samples in dimension d

[L,Shat,R]=VBMF(V);
%Uhat=L*diag(Shat)*R';

[L,Strue,R]=svd(U); [L,S,R]=svd(V);
Strue=diag(Strue); S=diag(S);

figure(2); clf; hold on
set(gca,'FontName','Helvetica'); set(gca,'FontSize',12)
plot([1:d],Strue,'k:','LineWidth',5)
plot([1:d],S,'r--','LineWidth',2)
plot([1:d],Shat,'b-','LineWidth',2)
legend(sprintf('True (rank=%g)',sum(Strue>0.001)),...
       sprintf('Noisy (rank=%g)',sum(S>0.001)),...
       sprintf('Denoised (rank=%g)',sum(Shat>0.001)))
xlabel('Eigenvalue index')
title(sprintf('Eigenspectrum of d by n matrix (d=%g, n=%g)',d,n))
set(gcf,'PaperUnits','centimeters');
set(gcf,'PaperPosition',[0 0 12 9]);
print('-dpng','eigenspectrum')
