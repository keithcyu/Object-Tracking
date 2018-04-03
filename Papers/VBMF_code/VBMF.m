function [omega_left,gamma_hat,omega_right]=VBMF(V)
%
% Variational Bayesian Matrix Factorization
%
% Given a noisy (fully observed) matrix V,
% its low-dimensional denoised matrix Uhat is computed,
% where Uhat=omega_left*diag(gamma_hat)*omega_right'.
%
% Usage:
%       [omega_left,gamma_hat,omega_right]=VBMF(V)
%
% Input:
%    V           :  L by M matrix to be denoised (L<=M is assumed!)
%
% Output:
%    omega_left  : Matrix consisting of left singular vectors
%    gamma_hat   : Vector consisting of singular values
%    omega_right : Matrix consisting of right singular vectors
%
% (c) Masashi Sugiyama, Department of Compter Science, Tokyo Institute of Technology, Japan.
%     sugi@cs.titech.ac.jp,     http://sugiyama-www.cs.titech.ac.jp/~sugi/software/uLSIF/

  [L,M]=size(V);
  if M<L
    error('L by M input matrix V should be L<=M !!!')
  end
  LM=L*M;
  [omega_left,tmp,omega_right]=svd(V);
  tmp=diag(tmp);
  H=sum(tmp>0.001);
  gamma=tmp(1:H);
  omega_left=omega_left(1:H,:);
  omega_right=omega_right(:,1:H);

  sigma_candidate=linspace(0.01,1,100);

  FE_best=inf;
  for sigma_index=1:length(sigma_candidate)

    % Compute gamma_EVB for a given sigma
    sigma2=sigma_candidate(sigma_index)^2;
    sigma4=sigma2^2;
    gamma2=gamma.^2;
    tmp=gamma2-(L+M)*sigma2;
    ca2cb2=max(eps,(tmp+sqrt(max(eps,tmp.^2-4*LM*sigma4)))/(2*LM));
    cacb=sqrt(max(eps,ca2cb2));
    ca2=cacb; cb2=cacb;
    sigma4_ca2cb2=sigma4./ca2cb2;
    tmp=(L+M)*sigma2/2+sigma4_ca2cb2/2;
    gamma_tilde=sqrt(max(eps,tmp+sqrt(max(eps,tmp.^2-LM*sigma4))));
    eta2=(1-sigma2*L./gamma2).*(1-sigma2*M./gamma2).*gamma2;
    gamma_EVB=zeros(H,1);
    for h=1:H
      if gamma(h)>gamma_tilde(h)
        xi(1)=1;
        xi(2)=(L-M)^2*gamma(h)/LM;
        xi(3)=-xi(2)*gamma(h)-(L^2+M^2)*eta2(h)/LM-2*sigma4_ca2cb2(h);
        xi(5)=(eta2(h)-sigma4_ca2cb2(h))^2;
        xi(4)=xi(2)*eta2(h)-sigma4_ca2cb2(h);
        sol=sort(roots(xi),'descend');
        gamma_EVB(h)=sol(2);
      end
    end %h
    tmp=gamma.*gamma_EVB;
    delta=M*log(tmp/(M*sigma2)+1)+L*log(tmp/(L*sigma2)+1)+(-2*tmp+LM*ca2cb2)/sigma2;
    gamma_underbar=(sqrt(L)+sqrt(M))*sqrt(sigma2);
    tmp=~((gamma>gamma_underbar) & (delta<=0));
    gamma_EVB(tmp)=0;

    %Compute free energy
    tmp=(M-L)*(gamma-gamma_EVB);
    delta_hat=(tmp.*ca2+sqrt(max(eps,(tmp.*ca2).^2+4*LM*sigma4)))/(2*sigma2*M);
    eta2_hat=sigma2./cacb;
    tmp=(gamma>gamma_tilde);
    eta2_hat(tmp)=eta2(tmp);
    mu_a2=gamma_EVB.*delta_hat;
    mu_b2=gamma_EVB./delta_hat;
    mu_b2(isnan(mu_b2))=0;
    tmp=eta2_hat-sigma2*(M-L);
    sigma_a2=max(eps,(-tmp+sqrt(max(eps,tmp.^2+4*M*sigma2.*eta2_hat)))./...
            (2*M*(gamma_EVB./delta_hat+sigma2./ca2)));
    tmp=eta2_hat+sigma2*(M-L);
    sigma_b2=max(eps,(-tmp+sqrt(max(eps,tmp.^2+4*L*sigma2.*eta2_hat)))./...
            (2*L*(gamma_EVB.*delta_hat+sigma2./cb2)));
    alpha=mu_a2+M*sigma_a2; 
    beta=mu_b2+L*sigma_b2;
    tmp=M*log(ca2./sigma_a2)/2+alpha./(2*ca2)+L*log(cb2./sigma_b2)/2+beta./(2*cb2);
    FE=LM*log(sigma2)/2+sum(tmp)+sum(gamma2-2*gamma.*gamma_EVB+alpha.*beta)/(2*sigma2);

    if FE<FE_best
      FE_best=FE;
      gamma_hat=gamma_EVB;
    end
  
  end % sigma_index

