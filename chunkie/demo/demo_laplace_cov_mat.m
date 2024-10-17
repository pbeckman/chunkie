%DEMO_SCATTER
%
% Define an exterior scattering problem on a starfish-shaped domain and 
% solve
%

%% Set up geometry

% clearvars; close all;
iseed = 8675309;
rng(iseed);
addpaths_loc();

% discretize domain

tol = 1e-12;

cparams = [];
cparams.eps = tol;

pref = []; 
pref.k = 16; 
pref.dim = 2;

start = tic;

% % point source boundary data
% x0 = [0; 0] + 0.1*[randn(); randn()];
% x1 = [-0.5; 0.5] + 0.1*[randn(); randn()];
% ffunc  = @(x) log(norm(x-x0)) - log(norm(x-x1));

% starfish
cparams.nover = 0;
narms = 5;
amp = 0.25;
chnkr = chunkerfunc(@(t) starfish(t,narms,amp),cparams,pref);

t1 = toc(start);

fprintf('%5.2e s : time to build geo\n',t1)

[~,~,info] = sortinfo(chnkr);
assert(info.ier == 0);

% plot geometry and data

% figure(1)
% clf
% plot(chnkr,'-x')
% hold on
% quiver(chnkr)
% axis equal

% solve and visualize the solution

%% Compute covariance matrix of boundary data

S = eye(chnkr.npt)/chnkr.npt;

% start = tic;
% k = @(xi, xj) exp(-norm(xi-xj));
% % k = @(xi, xj) exp(-norm(xi-xj)^2/100.0);
% for i = 1:chnkr.npt
%     for j = 1:chnkr.npt
%         S(i,j) = k(...
%             chnkr.r(:, rem(i-1,chnkr.k)+1, fix((i-1)/chnkr.k)+1), ...
%             chnkr.r(:, rem(j-1,chnkr.k)+1, fix((j-1)/chnkr.k)+1)  ...
%         );
% %         if i == j % numerical nugget
% %             S(i,j) = S(i,j) + 1e-12;
% %         end
%     end
% end
% t1 = toc(start);
% fprintf('%5.2e s : time to form covariance matrix of boundary data\n',t1)

% % plot samples
% figure(1)
% clf
% samples = chol(S)'*randn(chnkr.npt, 1);
% hold on
% for ch = 1:chnkr.nch
%     plot3(chnkr.r(1,:,ch), chnkr.r(2,:,ch), zeros(chnkr.k), 'color', 'blue');
%     plot3(chnkr.r(1,:,ch), chnkr.r(2,:,ch), samples((ch-1)*chnkr.k+1:ch*chnkr.k), 'color', 'red');
% end
% hold off
% view(45,45);

%% Build IE

fkern = @(s,t) chnk.lap2d.kern(s,t,'D')+1;
opdims(1) = 1; opdims(2) = 1;
% opts = struct('l2scale', true, 'eps', tol);
opts = struct('eps', tol);
start = tic; sysmat = chunkermat(chnkr,fkern,opts);
t1 = toc(start);

fprintf('%5.2e s : time to assemble matrix\n',t1)

K = 0.5*eye(chnkr.k*chnkr.nch) + sysmat;

rhs = chol(S)'*randn(chnkr.npt, 1);
% rhs = zeros(chnkr.nch*chnkr.k,1);
% bndpts = chnkr.r(:,:);
% for i=1:chnkr.nch*chnkr.k
%     rhs(i) = ffunc(bndpts(:,i));
% end
% adjust for l2 scaling
% wts   = weights(chnkr);
% l2rhs = rhs .* sqrt(wts(:));
% run GMRES
start = tic; sig = gmres(K,rhs,[],1e-13,100); t1 = toc(start);
% adjust for l2 scaling
% sig = sol ./ sqrt(wts(:));

fprintf('%5.2e s : time for dense gmres\n',t1)

% evaluate at targets and plot

rmin = min(chnkr); rmax = max(chnkr);
xl = rmax(1)-rmin(1);
yl = rmax(2)-rmin(2);
nplot = 20;
xtarg = linspace(rmin(1)-xl,rmax(1)+xl,nplot); 
ytarg = linspace(rmin(2)-yl,rmax(2)+yl,nplot);
[xxtarg,yytarg] = meshgrid(xtarg,ytarg);
targets = zeros(2,length(xxtarg(:)));
targets(1,:) = xxtarg(:); targets(2,:) = yytarg(:);

start = tic; in = chunkerinterior(chnkr,targets); t1 = toc(start);
out = ~in;

fprintf('%5.2e s : time to find points in domain\n',t1)

% compute layer potential based on oversample boundary

start = tic;
u = chunkerkerneval(chnkr,fkern,sig,targets(:,out),opts); t1 = toc(start);
fprintf('%5.2e s : time for kernel eval (for plotting)\n',t1)
start = tic;
A = chunkerkernevalmat(chnkr,fkern,targets(:,out),opts); t1 = toc(start);
fprintf('%5.2e s : time to form kernel eval matrix (for covariance)\n',t1)

%% Compute full dense covariance matrix of solution

start = tic;
% D = diag(sqrt(wts(:)));
% M = (D*((K')\(D\(A'))))';
M = ((K')\(A'))';
C = M*S*M';
t1 = toc(start);
fprintf('%5.2e s : time to form covariance matrix of solution\n',t1)

% fprintf('|u - A*D\\K\\f)| = %5.2e\n', norm(u - A*(D\sol)))

%% Plot everything

i0s = [
    nplot*round(nplot/10) + round(nplot/10),
    nplot*round(nplot/3) + round(2*nplot/3),
    nplot*round(3*nplot/4) + round(nplot/3)
    ];

maxu = max(abs(u));

figure(2)
clf
t = tiledlayout(2,3,'TileSpacing','Compact','Padding','Compact');
% nexttile;
% zztarg = nan(size(xxtarg));
% zztarg(out) = f;
% h=pcolor(xxtarg,yytarg,zztarg);
% set(h,'EdgeColor','none')
% hold on
% plot(chnkr,'LineWidth',2)
% axis equal
% axis tight
% colormap(redblue)
% caxis([-maxu,maxu])
% title('$f$','Interpreter','latex','FontSize',24)

nexttile;
zztarg = nan(size(xxtarg));
zztarg(out) = u;
h=pcolor(xxtarg,yytarg,zztarg);
set(h,'EdgeColor','none')
hold on
plot(chnkr,'k','LineWidth',0.5)
axis equal
axis tight
colormap(redblue)
caxis([-maxu,maxu])
title('$E(u)$','Interpreter','latex','FontSize',24)
colorbar

% vec = [100; 90; 50; 10; 0];
% hex = ['#661fa3'; '#661fa3'; '#ffffff'; '#219e1c'; '#219e1c'];
% raw = sscanf(hex','#%2x%2x%2x',[3,size(hex,1)]).' / 255;
% N = 128;
% greenpurple = interp1(vec,raw,linspace(100,0,N),'pchip');

uvarmax  = max(diag(C)); 
u0covmax = max(max(abs(C(:,i0s))));

nexttile;
zztarg = nan(size(xxtarg));
zztarg(out) = diag(C);
h=pcolor(xxtarg,yytarg,zztarg);
set(h,'EdgeColor','none')
hold on
plot(chnkr,'k','LineWidth',0.5)
axis equal
axis tight
colormap(redblue)
caxis([-uvarmax,uvarmax])
title('$Var(u)$','Interpreter','latex','FontSize',20)
colorbar

nexttile;
zztarg = nan(size(xxtarg));
zztarg(out) = log(diag(C));
h=pcolor(xxtarg,yytarg,zztarg);
set(h,'EdgeColor','none')
hold on
plot(chnkr,'k','LineWidth',0.5)
axis equal
axis tight
colormap(redblue)
title('$\log Var(u)$','Interpreter','latex','FontSize',20)
colorbar

tout = targets(:,out);

for j = 1:3
    nexttile;
    zztarg = nan(size(xxtarg));
    zztarg(out) = C(:,i0s(j));
    h=pcolor(xxtarg,yytarg,zztarg);
    set(h,'EdgeColor','none')
    hold on
    plot(chnkr,'k','LineWidth',0.5)
    scatter(tout(1,i0s(j)),tout(2,i0s(j)),'filled','SizeData',50,'MarkerFaceColor','black','LineWidth',1,'MarkerEdgeColor','white')
    axis equal
    axis tight
    colormap(redblue)
    caxis([-u0covmax,u0covmax])
    if j == 2
        title('$Cov(u(x_0),u(x))$','Interpreter','latex','FontSize',20)
    end
    colorbar
end

% exportgraphics(t,'/Users/beckman/Documents/Research/IE-UQ/output/mat1-wn-tol10.pdf','backgroundcolor','none')

