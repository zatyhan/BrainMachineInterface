function P = PCA(x,x_avg,p)
    %PCA Calculates the principal components 
    % x - preprocessed firing rate in bins
    % x_avg - trial average of preprocessed firing rate in bins
    % p - number of components
    % P - principal components matrix
    
    T = size(x,1);
    A=x'-x_avg';
    S=A'*A/T;
    [P,L]=eig(S);
    p=min(p,size(P,2));
    [~,ind]=maxk(diag(L),p);
    P=A*P(:,ind);
    P=P./sqrt(sum(P.^2));
end

