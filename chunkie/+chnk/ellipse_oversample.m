function nleg = ellipse_oversample(rho,npoly,tol)
%CHNK.ELLIPSE_OVERSAMPLE get oversampling factors based on bernstein
% ellipses
%
% input 
%
% rho - bernstein ellipse parameter
% npoly - degree of polynomial*log + polynomial to integrate
% tol - tolerance for integral 
%
% output
%
% nleg - the order of the legendre rule to use to get sufficient accuracy
% outside of the image of the bernstein ellipse under the boundary 
% parameterization.

[iprecs,npolys,rhos,nlegtab] = loadtab();

precs = 10.^(-iprecs);

if (tol < min(precs))
    warning('requested tolerance out of range, oversampling may be insufficient');
end
if (rho < min(rhos))
    warning('requested bernstein parameter out of range, oversampling may be insufficient');
end
if (npoly > max(npolys))
    warning('requested poly degree out of range, oversampling may be insufficient');
end

iprec = min(nnz(precs>tol)+1,length(iprecs));
ipol = min(nnz(npolys < npoly) + 1,length(npolys));
irho = max(length(rhos)-nnz(rhos > rho),1);

nleg = nlegtab(irho,ipol,iprec);


end


function [iprecs,npolys,rhos,nlegtab] = loadtab()
iprecs = zeros(5,1);
npolys = zeros(2,1);
rhos = zeros(5,1);
nlegtab = zeros(5,20,5);
      iprecs(  1) =    3;
      iprecs(  2) =    6;
      iprecs(  3) =    9;
      iprecs(  4) =   12;
      iprecs(  5) =   15;
      npolys(  1) =    4;
      npolys(  2) =    8;
      npolys(  3) =   12;
      npolys(  4) =   16;
      npolys(  5) =   20;
      npolys(  6) =   24;
      npolys(  7) =   28;
      npolys(  8) =   32;
      npolys(  9) =   36;
      npolys( 10) =   40;
      npolys( 11) =   44;
      npolys( 12) =   48;
      npolys( 13) =   52;
      npolys( 14) =   56;
      npolys( 15) =   60;
      npolys( 16) =   64;
      npolys( 17) =   68;
      npolys( 18) =   72;
      npolys( 19) =   76;
      npolys( 20) =   80;
      rhos(  1) =0.12000000000000000000E+01;
      rhos(  2) =0.14000000000000000000E+01;
      rhos(  3) =0.16000000000000000000E+01;
      rhos(  4) =0.18000000000000000000E+01;
      rhos(  5) =0.20000000000000000000E+01;
      nlegtab(   1,   1,   1) =   14;
      nlegtab(   2,   1,   1) =    9;
      nlegtab(   3,   1,   1) =    7;
      nlegtab(   4,   1,   1) =    6;
      nlegtab(   5,   1,   1) =    5;
      nlegtab(   1,   2,   1) =   14;
      nlegtab(   2,   2,   1) =    9;
      nlegtab(   3,   2,   1) =    7;
      nlegtab(   4,   2,   1) =    6;
      nlegtab(   5,   2,   1) =    6;
      nlegtab(   1,   3,   1) =   14;
      nlegtab(   2,   3,   1) =    9;
      nlegtab(   3,   3,   1) =    8;
      nlegtab(   4,   3,   1) =    7;
      nlegtab(   5,   3,   1) =    7;
      nlegtab(   1,   4,   1) =   14;
      nlegtab(   2,   4,   1) =    9;
      nlegtab(   3,   4,   1) =    8;
      nlegtab(   4,   4,   1) =    8;
      nlegtab(   5,   4,   1) =    8;
      nlegtab(   1,   5,   1) =   14;
      nlegtab(   2,   5,   1) =   10;
      nlegtab(   3,   5,   1) =    9;
      nlegtab(   4,   5,   1) =    9;
      nlegtab(   5,   5,   1) =    8;
      nlegtab(   1,   6,   1) =   14;
      nlegtab(   2,   6,   1) =   10;
      nlegtab(   3,   6,   1) =   10;
      nlegtab(   4,   6,   1) =    9;
      nlegtab(   5,   6,   1) =    9;
      nlegtab(   1,   7,   1) =   14;
      nlegtab(   2,   7,   1) =   11;
      nlegtab(   3,   7,   1) =   10;
      nlegtab(   4,   7,   1) =   10;
      nlegtab(   5,   7,   1) =   10;
      nlegtab(   1,   8,   1) =   14;
      nlegtab(   2,   8,   1) =   11;
      nlegtab(   3,   8,   1) =   11;
      nlegtab(   4,   8,   1) =   10;
      nlegtab(   5,   8,   1) =   10;
      nlegtab(   1,   9,   1) =   14;
      nlegtab(   2,   9,   1) =   12;
      nlegtab(   3,   9,   1) =   11;
      nlegtab(   4,   9,   1) =   11;
      nlegtab(   5,   9,   1) =   11;
      nlegtab(   1,  10,   1) =   14;
      nlegtab(   2,  10,   1) =   12;
      nlegtab(   3,  10,   1) =   12;
      nlegtab(   4,  10,   1) =   11;
      nlegtab(   5,  10,   1) =   11;
      nlegtab(   1,  11,   1) =   14;
      nlegtab(   2,  11,   1) =   13;
      nlegtab(   3,  11,   1) =   12;
      nlegtab(   4,  11,   1) =   12;
      nlegtab(   5,  11,   1) =   12;
      nlegtab(   1,  12,   1) =   14;
      nlegtab(   2,  12,   1) =   13;
      nlegtab(   3,  12,   1) =   12;
      nlegtab(   4,  12,   1) =   12;
      nlegtab(   5,  12,   1) =   12;
      nlegtab(   1,  13,   1) =   15;
      nlegtab(   2,  13,   1) =   13;
      nlegtab(   3,  13,   1) =   13;
      nlegtab(   4,  13,   1) =   13;
      nlegtab(   5,  13,   1) =   12;
      nlegtab(   1,  14,   1) =   15;
      nlegtab(   2,  14,   1) =   14;
      nlegtab(   3,  14,   1) =   13;
      nlegtab(   4,  14,   1) =   13;
      nlegtab(   5,  14,   1) =   13;
      nlegtab(   1,  15,   1) =   15;
      nlegtab(   2,  15,   1) =   14;
      nlegtab(   3,  15,   1) =   14;
      nlegtab(   4,  15,   1) =   13;
      nlegtab(   5,  15,   1) =   13;
      nlegtab(   1,  16,   1) =   16;
      nlegtab(   2,  16,   1) =   14;
      nlegtab(   3,  16,   1) =   14;
      nlegtab(   4,  16,   1) =   14;
      nlegtab(   5,  16,   1) =   13;
      nlegtab(   1,  17,   1) =   16;
      nlegtab(   2,  17,   1) =   15;
      nlegtab(   3,  17,   1) =   14;
      nlegtab(   4,  17,   1) =   14;
      nlegtab(   5,  17,   1) =   14;
      nlegtab(   1,  18,   1) =   16;
      nlegtab(   2,  18,   1) =   15;
      nlegtab(   3,  18,   1) =   15;
      nlegtab(   4,  18,   1) =   14;
      nlegtab(   5,  18,   1) =   14;
      nlegtab(   1,  19,   1) =   17;
      nlegtab(   2,  19,   1) =   15;
      nlegtab(   3,  19,   1) =   15;
      nlegtab(   4,  19,   1) =   15;
      nlegtab(   5,  19,   1) =   14;
      nlegtab(   1,  20,   1) =   17;
      nlegtab(   2,  20,   1) =   16;
      nlegtab(   3,  20,   1) =   15;
      nlegtab(   4,  20,   1) =   15;
      nlegtab(   5,  20,   1) =   15;
      nlegtab(   1,   1,   2) =   31;
      nlegtab(   2,   1,   2) =   18;
      nlegtab(   3,   1,   2) =   13;
      nlegtab(   4,   1,   2) =   11;
      nlegtab(   5,   1,   2) =   10;
      nlegtab(   1,   2,   2) =   31;
      nlegtab(   2,   2,   2) =   18;
      nlegtab(   3,   2,   2) =   13;
      nlegtab(   4,   2,   2) =   11;
      nlegtab(   5,   2,   2) =   10;
      nlegtab(   1,   3,   2) =   31;
      nlegtab(   2,   3,   2) =   18;
      nlegtab(   3,   3,   2) =   14;
      nlegtab(   4,   3,   2) =   12;
      nlegtab(   5,   3,   2) =   11;
      nlegtab(   1,   4,   2) =   31;
      nlegtab(   2,   4,   2) =   18;
      nlegtab(   3,   4,   2) =   14;
      nlegtab(   4,   4,   2) =   13;
      nlegtab(   5,   4,   2) =   12;
      nlegtab(   1,   5,   2) =   31;
      nlegtab(   2,   5,   2) =   18;
      nlegtab(   3,   5,   2) =   15;
      nlegtab(   4,   5,   2) =   13;
      nlegtab(   5,   5,   2) =   12;
      nlegtab(   1,   6,   2) =   31;
      nlegtab(   2,   6,   2) =   19;
      nlegtab(   3,   6,   2) =   15;
      nlegtab(   4,   6,   2) =   14;
      nlegtab(   5,   6,   2) =   13;
      nlegtab(   1,   7,   2) =   31;
      nlegtab(   2,   7,   2) =   19;
      nlegtab(   3,   7,   2) =   16;
      nlegtab(   4,   7,   2) =   15;
      nlegtab(   5,   7,   2) =   14;
      nlegtab(   1,   8,   2) =   31;
      nlegtab(   2,   8,   2) =   19;
      nlegtab(   3,   8,   2) =   16;
      nlegtab(   4,   8,   2) =   15;
      nlegtab(   5,   8,   2) =   15;
      nlegtab(   1,   9,   2) =   31;
      nlegtab(   2,   9,   2) =   20;
      nlegtab(   3,   9,   2) =   17;
      nlegtab(   4,   9,   2) =   16;
      nlegtab(   5,   9,   2) =   15;
      nlegtab(   1,  10,   2) =   31;
      nlegtab(   2,  10,   2) =   20;
      nlegtab(   3,  10,   2) =   18;
      nlegtab(   4,  10,   2) =   17;
      nlegtab(   5,  10,   2) =   16;
      nlegtab(   1,  11,   2) =   31;
      nlegtab(   2,  11,   2) =   21;
      nlegtab(   3,  11,   2) =   18;
      nlegtab(   4,  11,   2) =   17;
      nlegtab(   5,  11,   2) =   17;
      nlegtab(   1,  12,   2) =   31;
      nlegtab(   2,  12,   2) =   21;
      nlegtab(   3,  12,   2) =   19;
      nlegtab(   4,  12,   2) =   18;
      nlegtab(   5,  12,   2) =   18;
      nlegtab(   1,  13,   2) =   31;
      nlegtab(   2,  13,   2) =   21;
      nlegtab(   3,  13,   2) =   19;
      nlegtab(   4,  13,   2) =   19;
      nlegtab(   5,  13,   2) =   18;
      nlegtab(   1,  14,   2) =   31;
      nlegtab(   2,  14,   2) =   22;
      nlegtab(   3,  14,   2) =   20;
      nlegtab(   4,  14,   2) =   19;
      nlegtab(   5,  14,   2) =   19;
      nlegtab(   1,  15,   2) =   31;
      nlegtab(   2,  15,   2) =   22;
      nlegtab(   3,  15,   2) =   20;
      nlegtab(   4,  15,   2) =   20;
      nlegtab(   5,  15,   2) =   20;
      nlegtab(   1,  16,   2) =   31;
      nlegtab(   2,  16,   2) =   23;
      nlegtab(   3,  16,   2) =   21;
      nlegtab(   4,  16,   2) =   20;
      nlegtab(   5,  16,   2) =   20;
      nlegtab(   1,  17,   2) =   31;
      nlegtab(   2,  17,   2) =   23;
      nlegtab(   3,  17,   2) =   21;
      nlegtab(   4,  17,   2) =   21;
      nlegtab(   5,  17,   2) =   21;
      nlegtab(   1,  18,   2) =   31;
      nlegtab(   2,  18,   2) =   23;
      nlegtab(   3,  18,   2) =   22;
      nlegtab(   4,  18,   2) =   22;
      nlegtab(   5,  18,   2) =   21;
      nlegtab(   1,  19,   2) =   31;
      nlegtab(   2,  19,   2) =   24;
      nlegtab(   3,  19,   2) =   22;
      nlegtab(   4,  19,   2) =   22;
      nlegtab(   5,  19,   2) =   22;
      nlegtab(   1,  20,   2) =   32;
      nlegtab(   2,  20,   2) =   24;
      nlegtab(   3,  20,   2) =   23;
      nlegtab(   4,  20,   2) =   23;
      nlegtab(   5,  20,   2) =   22;
      nlegtab(   1,   1,   3) =   48;
      nlegtab(   2,   1,   3) =   28;
      nlegtab(   3,   1,   3) =   20;
      nlegtab(   4,   1,   3) =   16;
      nlegtab(   5,   1,   3) =   14;
      nlegtab(   1,   2,   3) =   48;
      nlegtab(   2,   2,   3) =   28;
      nlegtab(   3,   2,   3) =   20;
      nlegtab(   4,   2,   3) =   17;
      nlegtab(   5,   2,   3) =   15;
      nlegtab(   1,   3,   3) =   48;
      nlegtab(   2,   3,   3) =   28;
      nlegtab(   3,   3,   3) =   21;
      nlegtab(   4,   3,   3) =   17;
      nlegtab(   5,   3,   3) =   16;
      nlegtab(   1,   4,   3) =   48;
      nlegtab(   2,   4,   3) =   28;
      nlegtab(   3,   4,   3) =   21;
      nlegtab(   4,   4,   3) =   18;
      nlegtab(   5,   4,   3) =   16;
      nlegtab(   1,   5,   3) =   48;
      nlegtab(   2,   5,   3) =   28;
      nlegtab(   3,   5,   3) =   22;
      nlegtab(   4,   5,   3) =   19;
      nlegtab(   5,   5,   3) =   17;
      nlegtab(   1,   6,   3) =   48;
      nlegtab(   2,   6,   3) =   28;
      nlegtab(   3,   6,   3) =   22;
      nlegtab(   4,   6,   3) =   19;
      nlegtab(   5,   6,   3) =   18;
      nlegtab(   1,   7,   3) =   48;
      nlegtab(   2,   7,   3) =   28;
      nlegtab(   3,   7,   3) =   23;
      nlegtab(   4,   7,   3) =   20;
      nlegtab(   5,   7,   3) =   18;
      nlegtab(   1,   8,   3) =   48;
      nlegtab(   2,   8,   3) =   29;
      nlegtab(   3,   8,   3) =   23;
      nlegtab(   4,   8,   3) =   20;
      nlegtab(   5,   8,   3) =   19;
      nlegtab(   1,   9,   3) =   48;
      nlegtab(   2,   9,   3) =   29;
      nlegtab(   3,   9,   3) =   24;
      nlegtab(   4,   9,   3) =   21;
      nlegtab(   5,   9,   3) =   20;
      nlegtab(   1,  10,   3) =   48;
      nlegtab(   2,  10,   3) =   30;
      nlegtab(   3,  10,   3) =   24;
      nlegtab(   4,  10,   3) =   22;
      nlegtab(   5,  10,   3) =   20;
      nlegtab(   1,  11,   3) =   48;
      nlegtab(   2,  11,   3) =   30;
      nlegtab(   3,  11,   3) =   25;
      nlegtab(   4,  11,   3) =   22;
      nlegtab(   5,  11,   3) =   21;
      nlegtab(   1,  12,   3) =   48;
      nlegtab(   2,  12,   3) =   30;
      nlegtab(   3,  12,   3) =   25;
      nlegtab(   4,  12,   3) =   23;
      nlegtab(   5,  12,   3) =   22;
      nlegtab(   1,  13,   3) =   48;
      nlegtab(   2,  13,   3) =   31;
      nlegtab(   3,  13,   3) =   26;
      nlegtab(   4,  13,   3) =   24;
      nlegtab(   5,  13,   3) =   23;
      nlegtab(   1,  14,   3) =   48;
      nlegtab(   2,  14,   3) =   31;
      nlegtab(   3,  14,   3) =   26;
      nlegtab(   4,  14,   3) =   24;
      nlegtab(   5,  14,   3) =   23;
      nlegtab(   1,  15,   3) =   48;
      nlegtab(   2,  15,   3) =   31;
      nlegtab(   3,  15,   3) =   27;
      nlegtab(   4,  15,   3) =   25;
      nlegtab(   5,  15,   3) =   24;
      nlegtab(   1,  16,   3) =   48;
      nlegtab(   2,  16,   3) =   32;
      nlegtab(   3,  16,   3) =   27;
      nlegtab(   4,  16,   3) =   25;
      nlegtab(   5,  16,   3) =   25;
      nlegtab(   1,  17,   3) =   48;
      nlegtab(   2,  17,   3) =   32;
      nlegtab(   3,  17,   3) =   28;
      nlegtab(   4,  17,   3) =   26;
      nlegtab(   5,  17,   3) =   26;
      nlegtab(   1,  18,   3) =   48;
      nlegtab(   2,  18,   3) =   32;
      nlegtab(   3,  18,   3) =   28;
      nlegtab(   4,  18,   3) =   27;
      nlegtab(   5,  18,   3) =   26;
      nlegtab(   1,  19,   3) =   49;
      nlegtab(   2,  19,   3) =   33;
      nlegtab(   3,  19,   3) =   29;
      nlegtab(   4,  19,   3) =   27;
      nlegtab(   5,  19,   3) =   27;
      nlegtab(   1,  20,   3) =   49;
      nlegtab(   2,  20,   3) =   33;
      nlegtab(   3,  20,   3) =   29;
      nlegtab(   4,  20,   3) =   28;
      nlegtab(   5,  20,   3) =   28;
      nlegtab(   1,   1,   4) =   66;
      nlegtab(   2,   1,   4) =   37;
      nlegtab(   3,   1,   4) =   27;
      nlegtab(   4,   1,   4) =   22;
      nlegtab(   5,   1,   4) =   19;
      nlegtab(   1,   2,   4) =   66;
      nlegtab(   2,   2,   4) =   37;
      nlegtab(   3,   2,   4) =   27;
      nlegtab(   4,   2,   4) =   22;
      nlegtab(   5,   2,   4) =   20;
      nlegtab(   1,   3,   4) =   66;
      nlegtab(   2,   3,   4) =   37;
      nlegtab(   3,   3,   4) =   28;
      nlegtab(   4,   3,   4) =   23;
      nlegtab(   5,   3,   4) =   20;
      nlegtab(   1,   4,   4) =   66;
      nlegtab(   2,   4,   4) =   37;
      nlegtab(   3,   4,   4) =   28;
      nlegtab(   4,   4,   4) =   24;
      nlegtab(   5,   4,   4) =   21;
      nlegtab(   1,   5,   4) =   66;
      nlegtab(   2,   5,   4) =   38;
      nlegtab(   3,   5,   4) =   29;
      nlegtab(   4,   5,   4) =   24;
      nlegtab(   5,   5,   4) =   22;
      nlegtab(   1,   6,   4) =   66;
      nlegtab(   2,   6,   4) =   38;
      nlegtab(   3,   6,   4) =   29;
      nlegtab(   4,   6,   4) =   25;
      nlegtab(   5,   6,   4) =   22;
      nlegtab(   1,   7,   4) =   66;
      nlegtab(   2,   7,   4) =   38;
      nlegtab(   3,   7,   4) =   30;
      nlegtab(   4,   7,   4) =   25;
      nlegtab(   5,   7,   4) =   23;
      nlegtab(   1,   8,   4) =   66;
      nlegtab(   2,   8,   4) =   39;
      nlegtab(   3,   8,   4) =   30;
      nlegtab(   4,   8,   4) =   26;
      nlegtab(   5,   8,   4) =   24;
      nlegtab(   1,   9,   4) =   66;
      nlegtab(   2,   9,   4) =   39;
      nlegtab(   3,   9,   4) =   31;
      nlegtab(   4,   9,   4) =   27;
      nlegtab(   5,   9,   4) =   24;
      nlegtab(   1,  10,   4) =   66;
      nlegtab(   2,  10,   4) =   39;
      nlegtab(   3,  10,   4) =   31;
      nlegtab(   4,  10,   4) =   27;
      nlegtab(   5,  10,   4) =   25;
      nlegtab(   1,  11,   4) =   66;
      nlegtab(   2,  11,   4) =   40;
      nlegtab(   3,  11,   4) =   31;
      nlegtab(   4,  11,   4) =   28;
      nlegtab(   5,  11,   4) =   26;
      nlegtab(   1,  12,   4) =   66;
      nlegtab(   2,  12,   4) =   40;
      nlegtab(   3,  12,   4) =   32;
      nlegtab(   4,  12,   4) =   28;
      nlegtab(   5,  12,   4) =   26;
      nlegtab(   1,  13,   4) =   66;
      nlegtab(   2,  13,   4) =   40;
      nlegtab(   3,  13,   4) =   32;
      nlegtab(   4,  13,   4) =   29;
      nlegtab(   5,  13,   4) =   27;
      nlegtab(   1,  14,   4) =   66;
      nlegtab(   2,  14,   4) =   41;
      nlegtab(   3,  14,   4) =   33;
      nlegtab(   4,  14,   4) =   30;
      nlegtab(   5,  14,   4) =   28;
      nlegtab(   1,  15,   4) =   66;
      nlegtab(   2,  15,   4) =   41;
      nlegtab(   3,  15,   4) =   33;
      nlegtab(   4,  15,   4) =   30;
      nlegtab(   5,  15,   4) =   28;
      nlegtab(   1,  16,   4) =   66;
      nlegtab(   2,  16,   4) =   41;
      nlegtab(   3,  16,   4) =   34;
      nlegtab(   4,  16,   4) =   31;
      nlegtab(   5,  16,   4) =   29;
      nlegtab(   1,  17,   4) =   66;
      nlegtab(   2,  17,   4) =   42;
      nlegtab(   3,  17,   4) =   34;
      nlegtab(   4,  17,   4) =   31;
      nlegtab(   5,  17,   4) =   30;
      nlegtab(   1,  18,   4) =   66;
      nlegtab(   2,  18,   4) =   42;
      nlegtab(   3,  18,   4) =   35;
      nlegtab(   4,  18,   4) =   32;
      nlegtab(   5,  18,   4) =   31;
      nlegtab(   1,  19,   4) =   67;
      nlegtab(   2,  19,   4) =   42;
      nlegtab(   3,  19,   4) =   35;
      nlegtab(   4,  19,   4) =   33;
      nlegtab(   5,  19,   4) =   31;
      nlegtab(   1,  20,   4) =   67;
      nlegtab(   2,  20,   4) =   43;
      nlegtab(   3,  20,   4) =   36;
      nlegtab(   4,  20,   4) =   33;
      nlegtab(   5,  20,   4) =   32;
      nlegtab(   1,   1,   5) =   85;
      nlegtab(   2,   1,   5) =   47;
      nlegtab(   3,   1,   5) =   34;
      nlegtab(   4,   1,   5) =   28;
      nlegtab(   5,   1,   5) =   24;
      nlegtab(   1,   2,   5) =   85;
      nlegtab(   2,   2,   5) =   47;
      nlegtab(   3,   2,   5) =   34;
      nlegtab(   4,   2,   5) =   28;
      nlegtab(   5,   2,   5) =   24;
      nlegtab(   1,   3,   5) =   85;
      nlegtab(   2,   3,   5) =   47;
      nlegtab(   3,   3,   5) =   35;
      nlegtab(   4,   3,   5) =   29;
      nlegtab(   5,   3,   5) =   25;
      nlegtab(   1,   4,   5) =   85;
      nlegtab(   2,   4,   5) =   47;
      nlegtab(   3,   4,   5) =   35;
      nlegtab(   4,   4,   5) =   29;
      nlegtab(   5,   4,   5) =   26;
      nlegtab(   1,   5,   5) =   85;
      nlegtab(   2,   5,   5) =   47;
      nlegtab(   3,   5,   5) =   36;
      nlegtab(   4,   5,   5) =   30;
      nlegtab(   5,   5,   5) =   26;
      nlegtab(   1,   6,   5) =   85;
      nlegtab(   2,   6,   5) =   48;
      nlegtab(   3,   6,   5) =   36;
      nlegtab(   4,   6,   5) =   30;
      nlegtab(   5,   6,   5) =   27;
      nlegtab(   1,   7,   5) =   85;
      nlegtab(   2,   7,   5) =   48;
      nlegtab(   3,   7,   5) =   37;
      nlegtab(   4,   7,   5) =   31;
      nlegtab(   5,   7,   5) =   28;
      nlegtab(   1,   8,   5) =   85;
      nlegtab(   2,   8,   5) =   48;
      nlegtab(   3,   8,   5) =   37;
      nlegtab(   4,   8,   5) =   32;
      nlegtab(   5,   8,   5) =   28;
      nlegtab(   1,   9,   5) =   85;
      nlegtab(   2,   9,   5) =   49;
      nlegtab(   3,   9,   5) =   38;
      nlegtab(   4,   9,   5) =   32;
      nlegtab(   5,   9,   5) =   29;
      nlegtab(   1,  10,   5) =   85;
      nlegtab(   2,  10,   5) =   49;
      nlegtab(   3,  10,   5) =   38;
      nlegtab(   4,  10,   5) =   33;
      nlegtab(   5,  10,   5) =   30;
      nlegtab(   1,  11,   5) =   85;
      nlegtab(   2,  11,   5) =   49;
      nlegtab(   3,  11,   5) =   39;
      nlegtab(   4,  11,   5) =   33;
      nlegtab(   5,  11,   5) =   30;
      nlegtab(   1,  12,   5) =   85;
      nlegtab(   2,  12,   5) =   50;
      nlegtab(   3,  12,   5) =   39;
      nlegtab(   4,  12,   5) =   34;
      nlegtab(   5,  12,   5) =   31;
      nlegtab(   1,  13,   5) =   85;
      nlegtab(   2,  13,   5) =   50;
      nlegtab(   3,  13,   5) =   39;
      nlegtab(   4,  13,   5) =   35;
      nlegtab(   5,  13,   5) =   32;
      nlegtab(   1,  14,   5) =   85;
      nlegtab(   2,  14,   5) =   50;
      nlegtab(   3,  14,   5) =   40;
      nlegtab(   4,  14,   5) =   35;
      nlegtab(   5,  14,   5) =   32;
      nlegtab(   1,  15,   5) =   85;
      nlegtab(   2,  15,   5) =   51;
      nlegtab(   3,  15,   5) =   40;
      nlegtab(   4,  15,   5) =   36;
      nlegtab(   5,  15,   5) =   33;
      nlegtab(   1,  16,   5) =   85;
      nlegtab(   2,  16,   5) =   51;
      nlegtab(   3,  16,   5) =   41;
      nlegtab(   4,  16,   5) =   36;
      nlegtab(   5,  16,   5) =   34;
      nlegtab(   1,  17,   5) =   85;
      nlegtab(   2,  17,   5) =   52;
      nlegtab(   3,  17,   5) =   41;
      nlegtab(   4,  17,   5) =   37;
      nlegtab(   5,  17,   5) =   34;
      nlegtab(   1,  18,   5) =   85;
      nlegtab(   2,  18,   5) =   52;
      nlegtab(   3,  18,   5) =   42;
      nlegtab(   4,  18,   5) =   37;
      nlegtab(   5,  18,   5) =   35;
      nlegtab(   1,  19,   5) =   85;
      nlegtab(   2,  19,   5) =   52;
      nlegtab(   3,  19,   5) =   42;
      nlegtab(   4,  19,   5) =   38;
      nlegtab(   5,  19,   5) =   36;
      nlegtab(   1,  20,   5) =   85;
      nlegtab(   2,  20,   5) =   53;
      nlegtab(   3,  20,   5) =   43;
      nlegtab(   4,  20,   5) =   39;
      nlegtab(   5,  20,   5) =   37;

end
