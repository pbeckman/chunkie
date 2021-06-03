%DEMO_SCATTER
%
% Define an exterior scattering problem on a starfish-shaped domain and 
% solve
%

%% Set up geometry

clearvars; close all;
iseed = 8675309;
rng(iseed);
addpaths_loc();

% discretize domain

tol = 1e-5;

cparams = [];
cparams.eps = tol;

pref = []; 
pref.k = 16; 
pref.dim = 2;

start = tic;

% point source boundary data
x0 = [0; 0] + 0.1*[randn(); randn()]
x1 = [-0.5; 0.5] + 0.1*[randn(); randn()]
ffunc  = @(x) log(norm(x-x0)) - log(norm(x-x1));

% % linear boundary data
% t = pi/8*(2*rand() - 1);
% ffunc  = @(x) dot(x, [cos(t), sin(t)]);

% starfish
cparams.nover = 0;
narms = 5;
amp = 0.25;
chnkr = chunkerfunc(@(t) starfish(t,narms,amp),cparams,pref);

% % barbell
% verts = chnk.demo.barbell(1.0,1.0,0.5,0.5);
% nv    = size(verts,2);
% edgevals = zeros(nv,1);
% cparams.widths = 0.1*ones(size(verts,2),1);
% cparams.rounded = true;
% cparams.nover = 0;
% chnkr = chunkerpoly(verts,cparams,pref,edgevals);
% % chnkr = chnkr.refine(); 
% chnkr = chnkr.sort();

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

%% Build CFIE

fkern = @(s,t) chnk.lap2d.kern(s,t,'D')+1;
opdims(1) = 1; opdims(2) = 1;
% opts = [];
opts = struct('eps', tol);
start = tic; sysmat = chunkermat(chnkr,fkern,opts);
t1 = toc(start);

fprintf('%5.2e s : time to assemble matrix\n',t1)

sys = 0.5*eye(chnkr.k*chnkr.nch) + sysmat;

bndpts = chnkr.r(:,:);
rhs = zeros(chnkr.nch*chnkr.k,1);
for i=1:chnkr.nch*chnkr.k
    rhs(i) = ffunc(bndpts(:,i));
end
start = tic; sol = gmres(sys,rhs,[],tol,100); t1 = toc(start); % 1e-13

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

%

start = tic; in = chunkerinterior(chnkr,targets); t1 = toc(start);
out = ~in;

fprintf('%5.2e s : time to find points in domain\n',t1)

% compute layer potential based on oversample boundary

start = tic;
opts = struct("flam", 0, "forcesmooth", 1); % uncomment to get nontrivial kernmat
[u, kernmat] = chunkerkerneval(chnkr,fkern,sol,targets(:,out),opts); t1 = toc(start);
fprintf('%5.2e s : time for kernel eval (for plotting)\n',t1)

extargets = targets(:,out);
nout = size(extargets,2);
f = zeros(nout,1);
for i=1:nout
    f(i) = ffunc(extargets(:,i));
end

% compute variance of solution with iid Gaussian noise

%%

S = eye(chnkr.npt);

% k = @(xi, xj) exp(-norm(xi-xj)^2);
% for i = 1:chnkr.npt
%     for j = 1:chnkr.npt
%         S(i,j) = k(...
%             chnkr.r(:, rem(i-1,chnkr.k)+1, fix((i-1)/chnkr.k)+1), ...
%             chnkr.r(:, rem(j-1,chnkr.k)+1, fix((j-1)/chnkr.k)+1)  ...
%         );
%         if i == j
%             S(i,j) = S(i,j) + 1e-12;
%         end
%     end
% end
% 
% clf
% samples = chol(S)'*randn(chnkr.npt, 1);
% hold on
% for ch = 1:chnkr.nch
%     plot3(chnkr.r(1,:,ch), chnkr.r(2,:,ch), zeros(chnkr.k), 'color', 'blue');
%     plot3(chnkr.r(1,:,ch), chnkr.r(2,:,ch), samples((ch-1)*chnkr.k+1:ch*chnkr.k), 'color', 'red');
% end
% % zlim([-0.2, 0.2]);
% hold off

%%

nt = length(u);
i0s = [
    nplot*round(nplot/4) + round(nplot/4),
    nplot*round(nplot/3) + round(2*nplot/3),
    nplot*round(5*nplot/6) + round(nplot/6)
    ];
uvar  = zeros(nt,1);
u0cov = zeros(3,nt);
v0  = zeros(3,length(sys));
for j = 1:3
    v0(j,:)  = (sys.')\(kernmat(i0s(j),:).');
end
for i = 1:nt
    v = (sys.')\(kernmat(i,:).');
    uvar(i) = dot(v, S*v);
    for j=1:3
        u0cov(j,i) = dot(v0(j,:), S*v);
    end
end
% save('udist.mat', 'uvar', 'u0cov');
% load('udist.mat');

%% Plot everything

maxu = max(abs(f(:)));

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
plot(chnkr,'LineWidth',2)
axis equal
axis tight
colormap(redblue)
caxis([-maxu,maxu])
title('$E(u)$','Interpreter','latex','FontSize',24)

% vec = [100; 90; 50; 10; 0];
% hex = ['#661fa3'; '#661fa3'; '#ffffff'; '#219e1c'; '#219e1c'];
% raw = sscanf(hex','#%2x%2x%2x',[3,size(hex,1)]).' / 255;
% N = 128;
% greenpurple = interp1(vec,raw,linspace(100,0,N),'pchip');

uvarmax  = 10000.0; % max(uvar(:)); 
u0covmax = 10000.0; % max(max(abs([min(u0cov(:)), max(u0cov(:))])));

nexttile;
zztarg = nan(size(xxtarg));
zztarg(out) = uvar;
h=pcolor(xxtarg,yytarg,zztarg);
set(h,'EdgeColor','none')
hold on
plot(chnkr,'LineWidth',2)
axis equal
axis tight
colormap(redblue)
caxis([-uvarmax,uvarmax])
title('$Var(u)$','Interpreter','latex','FontSize',20)

nexttile;
zztarg = nan(size(xxtarg));
zztarg(out) = log(uvar);
h=pcolor(xxtarg,yytarg,zztarg);
set(h,'EdgeColor','none')
hold on
plot(chnkr,'LineWidth',2)
axis equal
axis tight
colormap(redblue)
title('$\log Var(u)$','Interpreter','latex','FontSize',20)

tout = targets(:,out);

for j = 1:3
    nexttile;
    zztarg = nan(size(xxtarg));
    zztarg(out) = u0cov(j,:);
    h=pcolor(xxtarg,yytarg,zztarg);
    set(h,'EdgeColor','none')
    hold on
    plot(chnkr,'LineWidth',2)
    scatter(tout(1,i0s(j)),tout(2,i0s(j)),'filled','SizeData',50,'MarkerFaceColor','black','LineWidth',1,'MarkerEdgeColor','white')
    axis equal
    axis tight
    colormap(redblue)
    caxis([-u0covmax,u0covmax])
    if j == 2
        title('$Cov(u(x_0),u(x))$','Interpreter','latex','FontSize',20)
    end
end
