%DEMO_SCATTER
%
% Define an exterior scattering problem on a starfish-shaped domain and 
% solve
%

%% Set up geometry

% clearvars; close all;
iseed = 198123;
% rng(iseed);
addpaths_loc();

% discretize domain

tol = 1e-10;

cparams = [];
cparams.eps = tol;

pref = []; 
pref.k = 16; 
pref.dim = 2;

start = tic;

% starfish
cparams.nover = 0;
narms = 5;
amp = 0.25;
chnkr = chunkerfunc(@(t) starfish(t,narms,amp),cparams,pref);

t1 = toc(start);

fprintf('%5.2e s : time to build geo\n',t1)

[~,~,info] = sortinfo(chnkr);
assert(info.ier == 0);

%% Compute covariance matrix of boundary data

S = eye(chnkr.npt)/chnkr.npt;

start = tic;
k = @(xi, xj, p) p(1) * exp(-norm(xi-xj)/p(2));
dks = {
    @(xi, xj, p) exp(-norm(xi-xj)/p(2)), ...
    @(xi, xj, p) p(1) * norm(xi-xj)/p(2)^2 * exp(-norm(xi-xj)/p(2))
    };
p   = [1.0, 1.0];
nug = 1e-12;
for i = 1:chnkr.npt
    for j = 1:chnkr.npt
        S(i,j) = k(...
            chnkr.r(:, rem(i-1,chnkr.k)+1, fix((i-1)/chnkr.k)+1), ...
            chnkr.r(:, rem(j-1,chnkr.k)+1, fix((j-1)/chnkr.k)+1), ...
            p ...
        );
        if i == j
            S(i,j) = S(i,j) + nug;
        end
    end
end
t1 = toc(start);
fprintf('%5.2e s : time to form covariance matrix of boundary data\n',t1)

% plot samples
figure(1)
clf
samples = chol(S)'*randn(chnkr.npt, 1);
hold on
for ch = 1:chnkr.nch
    plot3(chnkr.r(1,:,ch), chnkr.r(2,:,ch), zeros(chnkr.k), 'color', 'blue');
    plot3(chnkr.r(1,:,ch), chnkr.r(2,:,ch), samples((ch-1)*chnkr.k+1:ch*chnkr.k), 'color', 'red');
end
hold off
view(45,45);

%% Build IE

fkern = @(s,t) chnk.lap2d.kern(s,t,'D')+1;
opdims(1) = 1; opdims(2) = 1;
opts = struct('eps', tol);
start = tic; sysmat = chunkermat(chnkr,fkern,opts);
t1 = toc(start);

fprintf('%5.2e s : time to assemble matrix\n',t1)

K = 0.5*eye(chnkr.k*chnkr.nch) + sysmat;

rhs = chol(S)'*randn(chnkr.npt, 1);
% run GMRES
start = tic; sig = gmres(K,rhs,[],1e-13,100); t1 = toc(start);

fprintf('%5.2e s : time for dense gmres\n',t1)

% evaluate at targets and plot

rmin = min(chnkr); rmax = max(chnkr);
xl = rmax(1)-rmin(1);
yl = rmax(2)-rmin(2);
nplot = 100;
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

%% Collect some observations

% choose some observations from the boundary
% rng(123);
n_f_obs = 10;
f_obs_ind = randsample(chnkr.npt, n_f_obs);
f_unk_ind = setdiff(1:chnkr.npt, f_obs_ind);

% choose some observations from the exterior
% rng(456);
n_u_obs = 0;
u_obs_ind = randsample(1:sum(out), n_u_obs);
u_unk_ind = setdiff(1:sum(out), u_obs_ind);
M = ((K')\(A(u_obs_ind,:)'))';

%% Fit using maximum likelihood

% Y_q = chnkr.r(:,:);
% Y_o = Y_q(:,f_obs_ind);
% f_o = rhs(f_obs_ind) + nug * randn(n_f_obs,1);
% u_o = u(u_obs_ind) + nug * randn(n_u_obs,1);
% obj = @(p) nll(k,dks,p,Y_q,M,Y_o,f_o,u_o);
% 
% p0  = [10*rand(), 10*rand()];
% lb  = [0.0, 0.0];
% ub  = [Inf, Inf];
% 
% options = optimoptions('fmincon','MaxIterations',100); % 'SpecifyObjectiveGradient',true,
% mle = fmincon(obj, p0, [], [], [], [], lb, ub, [], options)

mle = [1, 1];

%% Compute full dense covariance matrix of solution at MLE

start = tic;

S_mle = zeros(chnkr.npt,chnkr.npt);
for i = 1:chnkr.npt
    for j = 1:chnkr.npt
        S_mle(i,j) = k(...
            chnkr.r(:, rem(i-1,chnkr.k)+1, fix((i-1)/chnkr.k)+1), ...
            chnkr.r(:, rem(j-1,chnkr.k)+1, fix((j-1)/chnkr.k)+1), ...
            mle ...
        );
    end
end

S11 = S_mle(f_obs_ind, f_obs_ind);
S12 = S_mle(f_obs_ind, f_unk_ind);
S22 = S_mle(f_unk_ind, f_unk_ind);

if n_u_obs == 0
    C0 = S12;
    C  = S11 + nug * diag(ones(n_f_obs,1));
    S0 = S22;
    y  = rhs(f_obs_ind);
elseif n_f_obs == 0
    C0 = M*S22;
    C  = M*S22*M' + nug * diag(ones(n_u_obs,1));
    S0 = S22;
    y  = u(u_obs_ind) + sqrt(nug) * randn(n_u_obs,1);
else
    S1 = S_mle(f_obs_ind,:);
    C0 = [S1; M*S_mle];
    C  = [S11 S1*M'; M*S1' M*S_mle*M'] + nug * diag(ones(n_f_obs + n_u_obs,1));
    S0 = S_mle;
    y  = [rhs(f_obs_ind); u(u_obs_ind) + sqrt(nug) * randn(n_u_obs,1)];
end

% compute conditional distribution of unknown boundary data
f_unk_cond = C0'*(C\y);
S_unk_cond = S0 - C0'*(C\C0);

if n_u_obs > 0 && n_f_obs > 0
    f_unk_cond = f_unk_cond(f_unk_ind);
    S_unk_cond = S_unk_cond(f_unk_ind,f_unk_ind);
end

% construct full conditional boundary data
f_cond = zeros(chnkr.npt, 1);
f_cond(f_obs_ind) = rhs(f_obs_ind);
f_cond(f_unk_ind) = f_unk_cond;

S_cond = zeros(chnkr.npt, chnkr.npt);
S_cond(f_unk_ind, f_unk_ind) = S_unk_cond;

% compute conditional distribution of solution
u_cond = A*(K\f_cond);
C_cond = A*(K\S_cond)*((K')\(A'));

t1 = toc(start);
fprintf('%5.2e s : time to form conditional distribution of solution\n',t1)

%% Plot everything

figure(2)
set(gcf, 'Position', [10 100 900 600]);
clf
t = tiledlayout(2,3,'TileSpacing','Compact','Padding','Compact');

% pick base points for conditional covariance plots
i0s = [
    nplot*round(nplot/10) + round(nplot/10),
    nplot*round(nplot/3) + round(2*nplot/3),
    nplot*round(2*nplot/3) + round(nplot/4)
    ];
tout = targets(:,out);
bndpts = chnkr.r(:,:);

% compute solution and covariance bounds
maxu = max(abs(u));
uvarmax  = max(diag(C_cond)); 
u0covmax = max(max(abs(C_cond(:,i0s))));

% set default colormap
colormap(redblue);

% plot solution
nexttile;
zztarg = nan(size(xxtarg));
zztarg(out) = u;
h=pcolor(xxtarg,yytarg,zztarg);
set(h,'EdgeColor','none')
hold on
plot(chnkr,'k','LineWidth',0.5)
axis equal
axis tight
caxis([-maxu,maxu])
title('$u$','Interpreter','latex','FontSize',24)
colorbar

% plot conditional mean
nexttile;
zztarg = nan(size(xxtarg));
zztarg(out) = u_cond;
h=pcolor(xxtarg,yytarg,zztarg);
set(h,'EdgeColor','none')
hold on
plot(chnkr,'k','LineWidth',0.5)
if n_f_obs > 0
    scatter(bndpts(1,f_obs_ind),bndpts(2,f_obs_ind),'filled','SizeData',20,'MarkerFaceColor','black','LineWidth',1,'MarkerEdgeColor','black')
end
if n_u_obs > 0
    scatter(tout(1,u_obs_ind),tout(2,u_obs_ind),'filled','SizeData',20,'MarkerFaceColor','black','LineWidth',1,'MarkerEdgeColor','black')
end
axis equal
axis tight
caxis([-maxu,maxu])
title('$E(u|\tilde{f})$','Interpreter','latex','FontSize',24)
colorbar

% plot conditional variance
nexttile;
zztarg = nan(size(xxtarg));
zztarg(out) = diag(C_cond);
h=pcolor(xxtarg,yytarg,zztarg);
set(h,'EdgeColor','none')
hold on
plot(chnkr,'k','LineWidth',0.5)
if n_f_obs > 0
    scatter(bndpts(1,f_obs_ind),bndpts(2,f_obs_ind),'filled','SizeData',20,'MarkerFaceColor','black','LineWidth',1,'MarkerEdgeColor','black')
end
if n_u_obs > 0
    scatter(tout(1,u_obs_ind),tout(2,u_obs_ind),'filled','SizeData',20,'MarkerFaceColor','black','LineWidth',1,'MarkerEdgeColor','black')
end
axis equal
axis tight
caxis([-uvarmax,uvarmax])
title('$Var(u|\tilde{f})$','Interpreter','latex','FontSize',20)
colorbar

% nexttile;
% zztarg = nan(size(xxtarg));
% zztarg(out) = real(log10(diag(C_cond)));
% h=pcolor(xxtarg,yytarg,zztarg);
% set(h,'EdgeColor','none')
% hold on
% plot(chnkr,'k','LineWidth',0.5)
% if n_f_obs > 0
%     scatter(bndpts(1,f_obs_ind),bndpts(2,f_obs_ind),'filled','SizeData',10,'MarkerFaceColor','black','LineWidth',1,'MarkerEdgeColor','black')
% end
% if n_u_obs > 0
%     scatter(tout(1,u_obs_ind),tout(2,u_obs_ind),'filled','SizeData',10,'MarkerFaceColor','black','LineWidth',1,'MarkerEdgeColor','black')
% end
% axis equal
% axis tight
% colormap(redblue)
% % caxis([-7,-0.5])
% title('$\log_{10} Var(u_*|f,u)$','Interpreter','latex','FontSize',20)
% colorbar

% plot conditional covariances
for j = 1:3
    nexttile;
    zztarg = nan(size(xxtarg));
    zztarg(out) = C_cond(:,i0s(j));
    h=pcolor(xxtarg,yytarg,zztarg);
    set(h,'EdgeColor','none')
    hold on
    plot(chnkr,'k','LineWidth',0.5)
    if n_f_obs > 0
        scatter(bndpts(1,f_obs_ind),bndpts(2,f_obs_ind),'filled','SizeData',20,'MarkerFaceColor','black','LineWidth',1,'MarkerEdgeColor','black')
    end
    if n_u_obs > 0
        scatter(tout(1,u_obs_ind),tout(2,u_obs_ind),'filled','SizeData',20,'MarkerFaceColor','black','LineWidth',1,'MarkerEdgeColor','black')
    end
    scatter(tout(1,i0s(j)),tout(2,i0s(j)),'filled','SizeData',80,'MarkerFaceColor','green','LineWidth',1,'MarkerEdgeColor','white')
    axis equal
    axis tight
    caxis([-u0covmax,u0covmax])
    if j == 2
        title('$Cov(u,u(x_0)|\tilde{f})$','Interpreter','latex','FontSize',20)
    end
    colorbar
end

% set solution plots to different colorscheme
ax = nexttile(1);
colormap(ax, jet);
ax = nexttile(2);
colormap(ax, jet);

% exportgraphics(t,'/Users/beckman/Documents/Research/NYU/IE-UQ/output/u10-nug4-2.pdf','backgroundcolor','none')

