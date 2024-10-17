function [l,g] = nll(k,dks,p,Y_q,M,Y_o,f_o,u_o)
    n_o = size(Y_o,2);
    n_q = size(Y_q,2);
    
    % Compute covariance matrix
    S_oo = zeros(n_o,n_o);
    for i = 1:n_o
        for j = 1:n_o
            S_oo(i,j) = k(Y_o(:,i), Y_o(:,j), p);
        end
    end
    S_oq = zeros(n_o,n_q);
    for i = 1:n_o
        for j = 1:n_q
            S_oq(i,j) = k(Y_o(:,i), Y_q(:,j), p);
        end
    end
    S_qq = zeros(n_q,n_q);
    for i = 1:n_q
        for j = 1:n_q
            S_qq(i,j) = k(Y_q(:,i), Y_q(:,j), p);
        end
    end
    
    S = [S_oo S_oq*M'; M*S_oq' M*S_qq*M'];
    S = S + p(end) * diag(ones(length(f_o) + length(u_o)));
    
    % Compute negative log likelihood
    logdet = 2*sum(log(diag(chol(S))));
    l      = logdet + dot([f_o; u_o], S\[f_o; u_o]);
    
    % Compute gradient
    if nargout > 1
        g = zeros(length(p),1);
        for k = 1:length(p)
%             if k == length(p) 
%                 % Compute nugget derivative covariance matrix
%                 dS = diag(ones(length(f_o) + length(u_o),1));
%             else
                % Compute non-nugget derivative covariance matrix
                dk = dks{k};
                dS_oo = zeros(n_o,n_o);
                for i = 1:n_o
                    for j = 1:n_o
                        dS_oo(i,j) = dk(Y_o(:,i), Y_o(:,j), p);
                    end
                end
                dS_oq = zeros(n_o,n_q);
                for i = 1:n_o
                    for j = 1:n_q
                        dS_oq(i,j) = dk(Y_o(:,i), Y_q(:,j), p);
                    end
                end
                dS_qq = zeros(n_q,n_q);
                for i = 1:n_q
                    for j = 1:n_q
                        dS_qq(i,j) = dk(Y_q(:,i), Y_q(:,j), p);
                    end
                end

                dS = [dS_oo dS_oq*M'; M*dS_oq' M*dS_qq*M'];
%             end
            g(k) = trace(S\dS) - dot([f_o; u_o], S\(dS*(S\[f_o; u_o])));
        end
    end
    disp(p)
    fprintf(" : ")
    disp(l)
%     disp(g)
end

